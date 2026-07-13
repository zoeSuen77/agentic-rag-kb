"""Dense semantic retrieval.

Dense retrieval embeds the query and searches child chunk vectors in Qdrant.
"""

from __future__ import annotations

from agentic_rag_kb.indexing.embeddings import EmbeddingModel
from agentic_rag_kb.indexing.qdrant_store import VectorStore


class DenseRetriever:
    """Semantic vector retriever for child chunks."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        collection_name: str,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Return dense retrieval candidates."""

        query_vector = self.embedding_model.embed_query(query)
        results = self.vector_store.search(self.collection_name, query_vector, top_k)
        return [
            {
                "child_id": result.payload.get("child_chunk_id"),
                "parent_id": result.payload.get("parent_id"),
                "text": result.payload.get("text", ""),
                "score_dense": result.score,
                "metadata": result.payload.get("metadata", {}),
                "payload": result.payload,
            }
            for result in results
        ]
