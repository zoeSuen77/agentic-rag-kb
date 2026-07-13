"""Tests for the independent LangGraph retrieval subgraph."""

from __future__ import annotations

from agentic_rag_kb.graph.retrieval_subgraph import (
    RetrievalSubgraphConfig,
    RetrievalSubgraphDependencies,
    build_retrieval_subgraph,
)
from agentic_rag_kb.graph.schema import default_retrieval_subgraph_state
from agentic_rag_kb.rerank import RerankConfig
from agentic_rag_kb.retrieval.models import RetrievedChunk, RetrievalDebugInfo


class FakeRetriever:
    """Fake hybrid retriever returning predefined contexts."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def retrieve(self, query: str, top_k_dense: int, top_k_sparse: int, final_k: int) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "top_k_dense": top_k_dense,
                "top_k_sparse": top_k_sparse,
                "final_k": final_k,
            }
        )
        return self.results[:final_k]

    def get_debug_info(self) -> RetrievalDebugInfo:
        return RetrievalDebugInfo(
            dense_hits=[{"child_id": result.child_id, "parent_id": result.parent_id} for result in self.results],
            sparse_hits=[],
            fused_ranking=[{"child_id": result.child_id, "score_fused": result.score_fused} for result in self.results],
            parent_contexts=[{"parent_id": result.parent_id, "parent_found": True} for result in self.results],
        )


class FakeReranker:
    """Fake reranker that assigns deterministic rerank scores."""

    def __init__(self, score: float = 0.8) -> None:
        self.score = score
        self.calls = 0

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int | None = None,
        final_context_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        for candidate in candidates:
            candidate.rerank_score = self.score
        return candidates[: final_context_k or len(candidates)]


class FakeLLM:
    """Fake LLM that returns an answer with citations."""

    def generate(self, prompt: str) -> str:
        assert "引用来源" in prompt
        return "Qdrant hybrid search 需要同时配置 dense 和 sparse 检索。\n\n引用来源：docs/qdrant.md"


def _context(
    child_id: str = "child_1",
    parent_id: str = "parent_1",
    text: str = "Qdrant hybrid search combines dense vectors and sparse keyword matching.",
    score_fused: float = 0.05,
    source_path: str = "docs/qdrant.md",
) -> RetrievedChunk:
    return RetrievedChunk(
        child_id=child_id,
        parent_id=parent_id,
        text=text,
        score_dense=0.9,
        score_sparse=0.7,
        score_fused=score_fused,
        metadata={"source_path": source_path},
        parent={"source_path": source_path, "title_path": "Qdrant > Hybrid Search"},
    )


def test_subgraph_normal_retrieval_lifecycle() -> None:
    retriever = FakeRetriever([_context()])
    reranker = FakeReranker(score=0.9)
    graph = build_retrieval_subgraph(
        RetrievalSubgraphDependencies(
            retriever=retriever,
            reranker=reranker,
            llm_client=FakeLLM(),
            config=RetrievalSubgraphConfig(
                rerank_config=RerankConfig(enable_rerank=True, rerank_top_n=5, final_context_k=3)
            ),
        )
    )

    result = graph.invoke(default_retrieval_subgraph_state("task_1", "如何配置 Qdrant hybrid search?"))

    assert result["sub_task_id"] == "task_1"
    assert result["rewritten_sub_query"] == "如何配置 Qdrant hybrid search?"
    assert len(result["retrieved_chunks"]) == 1
    assert len(result["reranked_contexts"]) == 1
    assert "引用来源" in result["sub_answer"]
    assert result["confidence"] >= 0.35
    assert result["insufficient_context"] is False
    assert reranker.calls == 1
    assert result["debug"]["hybrid_retrieve"]["retrieved_count"] == 1


def test_subgraph_no_results_marks_insufficient_context() -> None:
    graph = build_retrieval_subgraph(
        RetrievalSubgraphDependencies(
            retriever=FakeRetriever([]),
            reranker=FakeReranker(),
            llm_client=FakeLLM(),
        )
    )

    result = graph.invoke(default_retrieval_subgraph_state("task_2", "不存在的配置项怎么配置？"))

    assert result["retrieved_chunks"] == []
    assert result["reranked_contexts"] == []
    assert "上下文不足" in result["sub_answer"]
    assert result["confidence"] == 0.0
    assert result["insufficient_context"] is True


def test_subgraph_low_confidence_when_scores_are_weak() -> None:
    graph = build_retrieval_subgraph(
        RetrievalSubgraphDependencies(
            retriever=FakeRetriever([_context(score_fused=0.001)]),
            reranker=FakeReranker(score=0.05),
            llm_client=FakeLLM(),
            config=RetrievalSubgraphConfig(confidence_threshold=0.5),
        )
    )

    result = graph.invoke(default_retrieval_subgraph_state("task_3", "如何配置 Qdrant?"))

    assert 0.0 < result["confidence"] < 0.5
    assert result["insufficient_context"] is True


def test_subgraph_runs_when_rerank_disabled() -> None:
    retriever = FakeRetriever(
        [
            _context("child_1", "parent_1", score_fused=0.02, source_path="docs/a.md"),
            _context("child_2", "parent_2", score_fused=0.09, source_path="docs/b.md"),
        ]
    )
    reranker = FakeReranker(score=0.9)
    graph = build_retrieval_subgraph(
        RetrievalSubgraphDependencies(
            retriever=retriever,
            reranker=reranker,
            llm_client=None,
            config=RetrievalSubgraphConfig(
                rerank_config=RerankConfig(enable_rerank=False, rerank_top_n=5, final_context_k=1)
            ),
        )
    )

    result = graph.invoke(default_retrieval_subgraph_state("task_4", "如何配置 Qdrant hybrid search?"))

    assert len(result["reranked_contexts"]) == 1
    assert result["reranked_contexts"][0]["parent_id"] == "parent_2"
    assert reranker.calls == 0
    assert "引用来源" in result["sub_answer"]
    assert result["debug"]["rerank"]["enable_rerank"] is False
