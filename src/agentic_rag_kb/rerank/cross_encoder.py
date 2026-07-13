"""Cross-encoder reranking for parent-expanded retrieval contexts.

Hybrid retrieval is optimized for recall. This module performs the second stage:
take the highest hybrid-score parent contexts, score each `(query, context)` pair
with a CrossEncoder, and return the most relevant contexts for generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_rag_kb.indexing.sparse import tokenize
from agentic_rag_kb.retrieval.models import RetrievedParentContext


@dataclass(slots=True)
class RerankConfig:
    """Configuration for second-stage reranking."""

    enable_rerank: bool = True
    rerank_top_n: int = 20
    final_context_k: int = 5


class PairScoringModel(Protocol):
    """Protocol for models that score `(query, context)` pairs."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one relevance score per pair."""


class SentenceTransformersCrossEncoderModel:
    """Thin wrapper around sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install sentence-transformers to use CrossEncoderReranker.") from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query/context pairs."""

        scores = self.model.predict(pairs)
        return [float(score) for score in scores]


class CrossEncoderReranker:
    """Rerank candidate parent contexts with a CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: PairScoringModel | None = None,
        config: RerankConfig | None = None,
    ) -> None:
        self.config = config or RerankConfig()
        self.model = model or SentenceTransformersCrossEncoderModel(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedParentContext],
        top_n: int | None = None,
        final_context_k: int | None = None,
    ) -> list[RetrievedParentContext]:
        """Return reranked contexts.

        The method first takes the highest `score_fused` candidates, then scores
        them with the cross-encoder, and finally returns `final_context_k` contexts.
        `top_n` is kept for backwards compatibility with earlier call sites.
        """

        config = RerankConfig(
            enable_rerank=self.config.enable_rerank,
            rerank_top_n=top_n or self.config.rerank_top_n,
            final_context_k=final_context_k or top_n or self.config.final_context_k,
        )
        return rerank_parent_contexts(query, candidates, self.model, config)


class LexicalReranker:
    """Offline reranker with the same shape as CrossEncoderReranker."""

    def __init__(self, config: RerankConfig | None = None) -> None:
        self.config = config or RerankConfig()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedParentContext],
        top_n: int | None = None,
        final_context_k: int | None = None,
    ) -> list[RetrievedParentContext]:
        """Rerank by lexical overlap with parent context."""

        config = RerankConfig(
            enable_rerank=self.config.enable_rerank,
            rerank_top_n=top_n or self.config.rerank_top_n,
            final_context_k=final_context_k or top_n or self.config.final_context_k,
        )
        if not config.enable_rerank:
            return _top_by_hybrid_score(candidates, config.final_context_k)

        top_candidates = _top_by_hybrid_score(candidates, config.rerank_top_n)
        query_terms = set(tokenize(query))
        for candidate in top_candidates:
            context_terms = set(tokenize(candidate.text))
            candidate.rerank_score = len(query_terms & context_terms) / max(len(query_terms), 1)
        return sorted(top_candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)[
            : config.final_context_k
        ]


def rerank_parent_contexts(
    query: str,
    candidates: list[RetrievedParentContext],
    model: PairScoringModel,
    config: RerankConfig,
) -> list[RetrievedParentContext]:
    """Rerank parent contexts with a pair-scoring model."""

    if not candidates:
        return []
    if not config.enable_rerank:
        return _top_by_hybrid_score(candidates, config.final_context_k)

    top_candidates = _top_by_hybrid_score(candidates, config.rerank_top_n)
    pairs = [(query, candidate.text) for candidate in top_candidates]
    scores = model.predict(pairs)
    for candidate, score in zip(top_candidates, scores):
        candidate.rerank_score = float(score)
    return sorted(
        top_candidates,
        key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
        reverse=True,
    )[: config.final_context_k]


def _top_by_hybrid_score(
    candidates: list[RetrievedParentContext],
    limit: int,
) -> list[RetrievedParentContext]:
    return sorted(candidates, key=lambda item: item.score_fused, reverse=True)[:limit]
