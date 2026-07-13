"""Knowledge base indexing interfaces.

TODO:
- Create Qdrant collections for child and parent chunks.
- Generate sentence-transformer embeddings.
- Add sparse index support with rank-bm25 or fastembed sparse vectors.
- Persist parent-child mappings and metadata payloads.
"""

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk


class KnowledgeBaseIndexer:
    """Index parent and child chunks for Agentic RAG retrieval."""

    def build_index(self, parents: list[ParentChunk], children: list[ChildChunk]) -> None:
        """Build or update all indexes."""

        raise NotImplementedError("Index building will be implemented in the indexing phase.")

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks and metadata for one document."""

        raise NotImplementedError("Document deletion will be implemented with Qdrant payload filters.")

