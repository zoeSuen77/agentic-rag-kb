"""Chunking layer.

The chunking module builds parent-child hierarchical chunks. Child chunks support
high-precision retrieval, while parent chunks provide complete generation context.
"""

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk
from agentic_rag_kb.chunking.parent_child import ParentChildChunker

__all__ = ["ChildChunk", "ParentChunk", "ParentChildChunker"]

