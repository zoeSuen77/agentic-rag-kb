"""Chunking layer.

The chunking module builds parent-child hierarchical chunks. Child chunks support
high-precision retrieval, while parent chunks provide complete generation context.
"""

from agentic_rag_kb.chunking.models import ChildChunk, ChunkingReport, ParentChunk
from agentic_rag_kb.chunking.parent_child import ParentChildChunker, build_title_path
from agentic_rag_kb.chunking.text_splitter import RecursiveCharacterSplitter, estimate_tokens

__all__ = [
    "ChildChunk",
    "ChunkingReport",
    "ParentChunk",
    "ParentChildChunker",
    "RecursiveCharacterSplitter",
    "build_title_path",
    "estimate_tokens",
]
