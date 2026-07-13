"""Chunk data models for hierarchical indexing.

Parent chunks preserve larger semantic context for answer generation. Child chunks
are smaller retrieval units and always point back to a parent chunk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ParentChunk:
    """Large context block used for answer generation after child-level retrieval."""

    parent_id: str
    doc_id: str
    source_path: str
    title: str
    title_path: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        """Backward-compatible alias for early skeleton code."""

        return self.doc_id

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(slots=True)
class ChildChunk:
    """Small retrievable block indexed by dense and sparse retrievers."""

    child_id: str
    parent_id: str
    doc_id: str
    source_path: str
    title: str
    title_path: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        """Backward-compatible alias for early skeleton code."""

        return self.doc_id

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(slots=True)
class ChunkingReport:
    """Summary statistics emitted after parent-child chunking."""

    document_count: int
    parent_chunk_count: int
    child_chunk_count: int
    average_parent_tokens: float
    average_child_tokens: float
    min_parent_tokens: int
    max_parent_tokens: int
    min_child_tokens: int
    max_child_tokens: int

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)
