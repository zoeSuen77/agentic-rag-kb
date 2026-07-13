"""Indexing layer.

This module writes parent and child chunks into vector stores and sparse indexes.
The intended production backend is Qdrant with dense vectors, sparse vectors or
BM25-compatible signals, and metadata payloads for filtering and citations.
"""

from agentic_rag_kb.indexing.indexer import KnowledgeBaseIndexer

__all__ = ["KnowledgeBaseIndexer"]

