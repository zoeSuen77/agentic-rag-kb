"""Reranking layer.

The rerank module applies a cross-encoder to retrieval candidates so the final
context has higher precision before answer generation.
"""

from agentic_rag_kb.rerank.cross_encoder import (
    CrossEncoderReranker,
    LexicalReranker,
    RerankConfig,
    rerank_parent_contexts,
)

__all__ = ["CrossEncoderReranker", "LexicalReranker", "RerankConfig", "rerank_parent_contexts"]
