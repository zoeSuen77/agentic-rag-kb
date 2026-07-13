"""Tests for Cross-Encoder reranking stage."""

from __future__ import annotations

from agentic_rag_kb.rerank import CrossEncoderReranker, RerankConfig, rerank_parent_contexts
from agentic_rag_kb.retrieval.models import RetrievedParentContext


class FakeCrossEncoderModel:
    """Deterministic pair scorer for tests."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, context in pairs:
            query_terms = set(query.lower().split())
            context_terms = set(context.lower().split())
            scores.append(float(len(query_terms & context_terms)))
        return scores


def test_cross_encoder_reranker_preserves_data_structure() -> None:
    """Reranking should return the same RetrievedParentContext structure."""

    candidates = _contexts()
    reranker = CrossEncoderReranker(model=FakeCrossEncoderModel(), config=RerankConfig())

    reranked = reranker.rerank("database pool timeout", candidates, final_context_k=2)

    assert all(isinstance(context, RetrievedParentContext) for context in reranked)
    assert reranked[0].child_id
    assert reranked[0].parent_id
    assert isinstance(reranked[0].metadata, dict)


def test_rerank_score_exists() -> None:
    """Each returned context should receive a rerank_score."""

    reranked = rerank_parent_contexts(
        query="database pool timeout",
        candidates=_contexts(),
        model=FakeCrossEncoderModel(),
        config=RerankConfig(enable_rerank=True, rerank_top_n=3, final_context_k=2),
    )

    assert all(context.rerank_score is not None for context in reranked)
    assert reranked[0].parent_id == "parent_db"


def test_final_context_k_is_respected() -> None:
    """Reranker should return exactly final_context_k when enough candidates exist."""

    reranked = rerank_parent_contexts(
        query="database pool timeout",
        candidates=_contexts(),
        model=FakeCrossEncoderModel(),
        config=RerankConfig(enable_rerank=True, rerank_top_n=3, final_context_k=1),
    )

    assert len(reranked) == 1


def test_disable_rerank_returns_top_by_hybrid_score_without_scores() -> None:
    """When disabled, reranker should only truncate by hybrid score."""

    candidates = _contexts()
    reranked = rerank_parent_contexts(
        query="database pool timeout",
        candidates=candidates,
        model=FakeCrossEncoderModel(),
        config=RerankConfig(enable_rerank=False, rerank_top_n=3, final_context_k=2),
    )

    assert [context.parent_id for context in reranked] == ["parent_search", "parent_db"]
    assert all(context.rerank_score is None for context in reranked)


def test_rerank_top_n_limits_cross_encoder_candidates() -> None:
    """Cross-Encoder should score only the top rerank_top_n by hybrid score."""

    reranked = rerank_parent_contexts(
        query="database pool timeout",
        candidates=_contexts(),
        model=FakeCrossEncoderModel(),
        config=RerankConfig(enable_rerank=True, rerank_top_n=1, final_context_k=1),
    )

    assert reranked[0].parent_id == "parent_search"
    assert reranked[0].rerank_score == 0.0


def _contexts() -> list[RetrievedParentContext]:
    return [
        RetrievedParentContext(
            child_id="child_search",
            parent_id="parent_search",
            text="hybrid retrieval qdrant dense sparse",
            score_dense=0.92,
            score_sparse=3.0,
            score_fused=0.08,
            metadata={"source_path": "search.md"},
            parent={"parent_id": "parent_search", "text": "hybrid retrieval qdrant dense sparse"},
        ),
        RetrievedParentContext(
            child_id="child_db",
            parent_id="parent_db",
            text="database pool timeout max connections",
            score_dense=0.7,
            score_sparse=2.0,
            score_fused=0.07,
            metadata={"source_path": "db.md"},
            parent={"parent_id": "parent_db", "text": "database pool timeout max connections"},
        ),
        RetrievedParentContext(
            child_id="child_cache",
            parent_id="parent_cache",
            text="redis cache eviction memory policy",
            score_dense=0.6,
            score_sparse=1.0,
            score_fused=0.06,
            metadata={"source_path": "cache.md"},
            parent={"parent_id": "parent_cache", "text": "redis cache eviction memory policy"},
        ),
    ]
