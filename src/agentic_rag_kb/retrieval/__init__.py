"""Retrieval layer.

The retrieval module performs dense retrieval, sparse retrieval, hybrid fusion,
and parent expansion before reranking.
"""

from agentic_rag_kb.retrieval.hybrid import HybridRetriever

__all__ = ["HybridRetriever"]

