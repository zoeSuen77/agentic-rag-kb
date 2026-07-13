"""Conversation memory layer.

The memory module stores recent conversation turns, compressed summaries,
confirmed facts, unresolved questions, and retrieval traces needed for long
technical troubleshooting sessions.
"""

from agentic_rag_kb.memory.compression import (
    CompressionResult,
    CompressionStats,
    CompressionTrigger,
    ConversationCompressor,
    ConversationMemory,
    StructuredSummary,
    estimate_tokens,
    estimate_turn_tokens,
)
from agentic_rag_kb.memory.store import ConversationMemoryStore, LocalJSONMemoryStore

__all__ = [
    "CompressionResult",
    "CompressionStats",
    "CompressionTrigger",
    "ConversationCompressor",
    "ConversationMemory",
    "ConversationMemoryStore",
    "LocalJSONMemoryStore",
    "StructuredSummary",
    "estimate_tokens",
    "estimate_turn_tokens",
]
