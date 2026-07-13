"""Knowledge base indexing pipeline.

The indexer writes child chunks into Qdrant for dense retrieval and stores parent
chunks in a local docstore for full-context expansion at answer time. Sparse terms
are stored in the Qdrant payload so the retrieval layer can later implement
Dense+Sparse hybrid search without changing the point schema.
"""

from __future__ import annotations

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk
from agentic_rag_kb.indexing.docstore import ParentDocStore
from agentic_rag_kb.indexing.embeddings import EmbeddingModel, SentenceTransformerEmbeddingModel
from agentic_rag_kb.indexing.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    QdrantStore,
    VectorPoint,
    VectorSearchResult,
    VectorStore,
)
from agentic_rag_kb.indexing.sparse import SparsePayloadBuilder


class KnowledgeBaseIndexer:
    """Index parent and child chunks for Agentic RAG retrieval."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_model: EmbeddingModel | None = None,
        parent_docstore: ParentDocStore | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.vector_store = vector_store or QdrantStore()
        self.embedding_model = embedding_model or SentenceTransformerEmbeddingModel()
        self.parent_docstore = parent_docstore or ParentDocStore()
        self.collection_name = collection_name
        self.sparse_payload_builder = SparsePayloadBuilder()

    def build_index(self, parents: list[ParentChunk], children: list[ChildChunk]) -> None:
        """Build or rebuild all indexes.

        Repeated runs intentionally recreate the Qdrant collection, then upsert all
        child chunks and rewrite the parent docstore.
        """

        self.parent_docstore.save(parents)
        self.vector_store.recreate_collection(self.collection_name, self.embedding_model.vector_size)
        vectors = self.embedding_model.embed_documents([child.text for child in children])
        points = [
            VectorPoint(
                point_id=child.child_id,
                vector=vector,
                payload=self._build_child_payload(child),
            )
            for child, vector in zip(children, vectors)
        ]
        if points:
            self.vector_store.upsert(self.collection_name, points)

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks and metadata for one document."""

        raise NotImplementedError("Document deletion will be implemented with Qdrant payload filters.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search indexed child chunks and attach parent context when available."""

        query_vector = self.embedding_model.embed_query(query)
        results = self.vector_store.search(self.collection_name, query_vector, top_k=top_k)
        return [self._hydrate_search_result(result) for result in results]

    def _build_child_payload(self, child: ChildChunk) -> dict:
        sparse_payload = self.sparse_payload_builder.build(child.text)
        metadata = {
            **child.metadata,
            "doc_id": child.doc_id,
            "parent_id": child.parent_id,
            "source_path": child.source_path,
            "section_title": child.metadata.get("section_title"),
            "chunk_index": child.chunk_index,
        }
        return {
            "child_chunk_id": child.child_id,
            "parent_id": child.parent_id,
            "doc_id": child.doc_id,
            "text": child.text,
            "metadata": metadata,
            **sparse_payload,
        }

    def _hydrate_search_result(self, result: VectorSearchResult) -> dict:
        payload = result.payload
        parent_id = payload.get("parent_id")
        return {
            "child_chunk_id": payload.get("child_chunk_id"),
            "parent_id": parent_id,
            "doc_id": payload.get("doc_id"),
            "text": payload.get("text"),
            "metadata": payload.get("metadata", {}),
            "score": result.score,
            "parent": self.parent_docstore.get_parent(parent_id) if parent_id else None,
        }
