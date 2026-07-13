"""Retrieval data models.

`RetrievedChunk` carries both first-stage retrieval scores and parent-expanded
context used by answer generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class RetrievedChunk:
    """Hybrid retrieval result after child-level fusion and parent expansion."""

    child_id: str
    parent_id: str
    text: str
    score_dense: float | None
    score_sparse: float | None
    score_fused: float
    metadata: dict = field(default_factory=dict)
    parent: dict | None = None
    rerank_score: float | None = None

    def to_json_dict(self) -> dict:
        """Return JSON-serializable data."""

        return asdict(self)


RetrievedParentContext = RetrievedChunk


@dataclass(slots=True)
class RetrievalDebugInfo:
    """Debug trace for a retrieval call."""

    dense_hits: list[dict] = field(default_factory=list)
    sparse_hits: list[dict] = field(default_factory=list)
    fused_ranking: list[dict] = field(default_factory=list)
    parent_contexts: list[dict] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        """Return JSON-serializable data."""

        return asdict(self)
