"""Tests for evidence-aware answer aggregation."""

from __future__ import annotations

from agentic_rag_kb.graph.main_graph import answer_aggregation_node
from agentic_rag_kb.graph.schema import default_main_graph_state


class FakeAggregationLLM:
    """Fake LLM returning a grounded final answer."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _base_state() -> dict:
    state = default_main_graph_state("父子分层索引、混合检索和重排如何提升 RAG？")
    state["rewritten_query"] = state["original_query"]
    return state


def _context(sub_task_id: str, parent_id: str, source: str, text: str) -> dict:
    return {
        "sub_task_id": sub_task_id,
        "sub_query": f"{sub_task_id} query",
        "child_id": f"child_{sub_task_id}",
        "parent_id": parent_id,
        "text": text,
        "score_fused": 0.05,
        "rerank_score": 0.8,
        "metadata": {"source_path": source},
        "parent": {"source_path": source, "title_path": "RAG > Design"},
    }


def test_multiple_sub_answers_are_aggregated_with_structure_and_debug() -> None:
    state = _base_state()
    state["sub_answers"] = [
        {
            "sub_task_id": "task_1",
            "sub_query": "父子分层索引解决什么问题",
            "sub_answer": "父子分层索引用父块保留上下文，用子块提升检索精度。\n\n引用来源：docs/hierarchy.md",
            "confidence": 0.86,
            "insufficient_context": False,
        },
        {
            "sub_task_id": "task_2",
            "sub_query": "混合检索解决什么问题",
            "sub_answer": "混合检索结合语义召回和关键词召回，减少漏召回。\n\n引用来源：docs/hybrid.md",
            "confidence": 0.84,
            "insufficient_context": False,
        },
    ]
    state["retrieved_contexts"] = [
        _context("task_1", "parent_1", "docs/hierarchy.md", "父子分层索引兼顾上下文完整性和检索精度。"),
        _context("task_2", "parent_2", "docs/hybrid.md", "Dense + Sparse hybrid retrieval improves recall."),
    ]

    result = answer_aggregation_node(state)

    assert "直接回答" in result["final_answer"]
    assert "分点解释" in result["final_answer"]
    assert "关键依据" in result["final_answer"]
    assert "引用来源" in result["final_answer"]
    assert result["aggregation_debug"]["used_sub_answers"][0]["sub_task_id"] == "task_1"
    assert result["aggregation_debug"]["discarded_duplicates"] == []
    assert result["aggregation_debug"]["citation_sources"] == ["docs/hierarchy.md", "docs/hybrid.md"]


def test_conflicting_sub_answers_are_reported_with_sources() -> None:
    state = _base_state()
    state["sub_answers"] = [
        {
            "sub_task_id": "task_1",
            "sub_query": "Qdrant hybrid search 是否需要 sparse vector",
            "sub_answer": "Qdrant hybrid search 必须开启 sparse vector 才能做关键词召回。\n\n引用来源：docs/qdrant-a.md",
            "confidence": 0.9,
            "insufficient_context": False,
        },
        {
            "sub_task_id": "task_2",
            "sub_query": "Qdrant hybrid search sparse vector 配置",
            "sub_answer": "Qdrant hybrid search 不需要 sparse vector，也可以完成关键词召回。\n\n引用来源：docs/qdrant-b.md",
            "confidence": 0.88,
            "insufficient_context": False,
        },
    ]
    state["retrieved_contexts"] = [
        _context("task_1", "parent_1", "docs/qdrant-a.md", "hybrid search requires sparse vector config."),
        _context("task_2", "parent_2", "docs/qdrant-b.md", "hybrid search does not require sparse vector config."),
    ]

    result = answer_aggregation_node(state)

    assert "冲突说明" in result["final_answer"]
    assert result["aggregation_debug"]["conflicts"]
    assert result["aggregation_debug"]["conflicts"][0]["sources"] == ["docs/qdrant-a.md", "docs/qdrant-b.md"]


def test_insufficient_context_is_explicitly_reported() -> None:
    state = _base_state()
    state["sub_answers"] = [
        {
            "sub_task_id": "task_1",
            "sub_query": "如何评测线上真实准确率",
            "sub_answer": "当前上下文不足，无法回答该子问题。\n\n引用来源：无",
            "confidence": 0.0,
            "insufficient_context": True,
        }
    ]

    result = answer_aggregation_node(state)

    assert "上下文不足" in result["final_answer"]
    assert "task_1" in result["final_answer"]
    assert result["aggregation_debug"]["insufficient_context"][0]["sub_task_id"] == "task_1"


def test_llm_aggregation_preserves_allowed_citations() -> None:
    state = _base_state()
    state["sub_answers"] = [
        {
            "sub_task_id": "task_1",
            "sub_query": "Cross-Encoder 重排作用",
            "sub_answer": "Cross-Encoder 对 query-context 成对打分，适合精排。\n\n引用来源：docs/rerank.md",
            "confidence": 0.91,
            "insufficient_context": False,
        }
    ]
    state["retrieved_contexts"] = [
        _context("task_1", "parent_1", "docs/rerank.md", "Cross-Encoder pair scoring improves precision.")
    ]
    llm = FakeAggregationLLM(
        "直接回答\nCross-Encoder 用于精排。 [docs/rerank.md]\n\n"
        "分点解释\n- 成对打分提升上下文精度。 [docs/rerank.md]\n\n"
        "关键依据\n- docs/rerank.md\n\n引用来源\n- docs/rerank.md\n\n上下文不足\n无"
    )

    result = answer_aggregation_node(state, llm)

    assert result["final_answer"] == llm.response
    assert "允许引用来源" in llm.prompts[0]
    assert result["aggregation_debug"]["citation_sources"] == ["docs/rerank.md"]


def test_llm_unknown_citation_falls_back_and_records_error() -> None:
    state = _base_state()
    state["sub_answers"] = [
        {
            "sub_task_id": "task_1",
            "sub_query": "父子分层索引",
            "sub_answer": "父子分层索引提升上下文完整性。\n\n引用来源：docs/hierarchy.md",
            "confidence": 0.8,
            "insufficient_context": False,
        }
    ]
    state["retrieved_contexts"] = [
        _context("task_1", "parent_1", "docs/hierarchy.md", "Parent context preserves complete sections.")
    ]
    llm = FakeAggregationLLM("直接回答\n结论来自不存在的来源。 [docs/fake.md]")

    result = answer_aggregation_node(state, llm)

    assert "docs/fake.md" not in result["final_answer"]
    assert any("unknown citations" in error for error in result["error_messages"])
