from __future__ import annotations

from app.indexing.schemas import ChildChunk, SearchResult
from app.llm.embeddings import HashEmbeddingModel, cosine_similarity


class DenseIndexer:
    def __init__(self, embedding_model: HashEmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.vectors: dict[str, list[float]] = {}

    def index(self, chunks: list[ChildChunk]) -> None:
        for chunk in chunks:
            self.vectors[chunk.child_id] = self.embedding_model.embed(chunk.text)

    def search(self, query: str, chunks_by_id: dict[str, ChildChunk], top_k: int) -> list[SearchResult]:
        query_vector = self.embedding_model.embed(query)
        scored: list[SearchResult] = []
        for child_id, vector in self.vectors.items():
            chunk = chunks_by_id.get(child_id)
            if chunk is None:
                continue
            score = cosine_similarity(query_vector, vector)
            scored.append(
                SearchResult(
                    chunk_id=chunk.child_id,
                    parent_id=chunk.parent_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                    source="dense",
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

