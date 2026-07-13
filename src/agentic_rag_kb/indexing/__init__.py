"""Indexing layer.

This module writes child chunks into Qdrant and parent chunks into a local docstore.
The point payload keeps sparse terms for future Dense+Sparse hybrid retrieval.
"""

from agentic_rag_kb.indexing.docstore import ParentDocStore
from agentic_rag_kb.indexing.embeddings import DeterministicEmbeddingModel, SentenceTransformerEmbeddingModel
from agentic_rag_kb.indexing.indexer import KnowledgeBaseIndexer
from agentic_rag_kb.indexing.qdrant_store import InMemoryVectorStore, QdrantStore

__all__ = [
    "DeterministicEmbeddingModel",
    "InMemoryVectorStore",
    "KnowledgeBaseIndexer",
    "ParentDocStore",
    "QdrantStore",
    "SentenceTransformerEmbeddingModel",
]
