"""Chunk data models for hierarchical indexing."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParentChunk:
    """Large context block used for answer generation after child-level retrieval."""

    parent_id: str
    document_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ChildChunk:
    """Small retrievable block indexed by dense and sparse retrievers."""

    child_id: str
    parent_id: str
    document_id: str
    text: str
    metadata: dict = field(default_factory=dict)

