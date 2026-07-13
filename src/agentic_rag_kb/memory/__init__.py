"""Conversation memory layer.

The memory module stores recent conversation turns, compressed summaries,
confirmed facts, unresolved questions, and retrieval traces needed for long
technical troubleshooting sessions.
"""

from agentic_rag_kb.memory.store import ConversationMemoryStore

__all__ = ["ConversationMemoryStore"]

