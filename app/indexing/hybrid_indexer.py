from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from app.indexing.dense_indexer import DenseIndexer
from app.indexing.metadata_store import read_json, write_json
from app.indexing.schemas import ChildChunk, ParentChunk, SearchResult
from app.indexing.sparse_indexer import SparseIndexer
from app.llm.embeddings import HashEmbeddingModel


class HybridIndexer:
    def __init__(self, embedding_dimensions: int = 256) -> None:
        self.embedding_model = HashEmbeddingModel(embedding_dimensions)
        self.parents_by_id: dict[str, ParentChunk] = {}
        self.children_by_id: dict[str, ChildChunk] = {}
        self.dense = DenseIndexer(self.embedding_model)
        self.sparse = SparseIndexer()

    def index(self, parents: list[ParentChunk], children: list[ChildChunk]) -> None:
        self.parents_by_id = {parent.parent_id: parent for parent in parents}
        self.children_by_id = {child.child_id: child for child in children}
        self.dense.index(children)
        self.sparse.index(children)

    def dense_search(self, query: str, top_k: int) -> list[SearchResult]:
        return self.dense.search(query, self.children_by_id, top_k)

    def sparse_search(self, query: str, top_k: int) -> list[SearchResult]:
        return self.sparse.search(query, self.children_by_id, top_k)

    def save(self, directory: Path) -> None:
        write_json(directory / "parents.json", [asdict(parent) for parent in self.parents_by_id.values()])
        write_json(directory / "children.json", [asdict(child) for child in self.children_by_id.values()])

    @classmethod
    def load_or_create(cls, directory: Path, embedding_dimensions: int = 256) -> "HybridIndexer":
        indexer = cls(embedding_dimensions=embedding_dimensions)
        parents_data = read_json(directory / "parents.json", [])
        children_data = read_json(directory / "children.json", [])
        if parents_data and children_data:
            parents = [ParentChunk(**item) for item in parents_data]
            children = [ChildChunk(**item) for item in children_data]
            indexer.index(parents, children)
        return indexer
