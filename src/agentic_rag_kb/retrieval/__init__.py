"""Retrieval layer.

The retrieval module performs dense retrieval, sparse retrieval, RRF hybrid fusion,
debug tracing, and parent context expansion before reranking.
"""

from agentic_rag_kb.retrieval.hybrid import HybridRetriever
from agentic_rag_kb.retrieval.models import RetrievedChunk, RetrievedParentContext, RetrievalDebugInfo

__all__ = ["HybridRetriever", "RetrievedChunk", "RetrievedParentContext", "RetrievalDebugInfo"]
