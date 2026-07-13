"""Tests for Agentic RAG fault tolerance and fallbacks."""

from __future__ import annotations

import json

from agentic_rag_kb.agents import ambiguity_detection_node, query_rewrite_node
from agentic_rag_kb.agents.clarification import MAX_CLARIFICATION_LOOPS, fallback_node
from agentic_rag_kb.graph.main_graph import MainGraphDependencies, build_main_graph
from agentic_rag_kb.graph.retrieval_subgraph import (
    RetrievalSubgraphDependencies,
    build_retrieval_subgraph,
)
from agentic_rag_kb.graph.schema import default_main_graph_state, default_retrieval_subgraph_state
from agentic_rag_kb.retrieval.models import RetrievalDebugInfo


class RawLLM:
    """Fake LLM returning raw strings in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.responses.pop(0)


class EmptyRetriever:
    """Retriever returning no contexts."""

    def retrieve(self, query: str, top_k_dense: int, top_k_sparse: int, final_k: int) -> list:
        return []

    def get_debug_info(self) -> RetrievalDebugInfo:
        return RetrievalDebugInfo()


class NoopReranker:
    """Reranker double."""

    def rerank(self, query: str, candidates: list, top_n: int | None = None, final_context_k: int | None = None) -> list:
        return candidates[: final_context_k or len(candidates)]


class FlakyMainSubgraph:
    """Subgraph that fails once for task_2 and succeeds otherwise."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def invoke(self, state: dict) -> dict:
        task_id = state["sub_task_id"]
        self.calls[task_id] = self.calls.get(task_id, 0) + 1
        if task_id == "task_2" and self.calls[task_id] == 1:
            raise RuntimeError("temporary subgraph failure")
        return {
            **state,
            "reranked_contexts": [
                {
                    "child_id": f"child_{task_id}",
                    "parent_id": f"parent_{task_id}",
                    "text": "context",
                    "score_fused": 0.05,
                    "metadata": {"source_path": f"docs/{task_id}.md"},
                    "parent": {"source_path": f"docs/{task_id}.md"},
                }
            ],
            "sub_answer": f"{state['sub_query']} answer\n\n引用来源：docs/{task_id}.md",
            "confidence": 0.8,
            "insufficient_context": False,
            "debug": {},
            "error_messages": [],
        }


def test_ambiguous_query_loop_limit_uses_fallback_type() -> None:
    state = fallback_node(
        {
            "original_query": "这个怎么弄？",
            "loop_count": MAX_CLARIFICATION_LOOPS,
            "error_messages": [],
        }
    )

    assert state["fallback_type"] == "ambiguous_query_fallback"
    assert "问题仍然不够明确" in state["final_answer"]
    assert "请问【系统/模块】" in state["final_answer"]
    assert any("ambiguous_query_fallback" in error for error in state["error_messages"])


def test_retrieval_empty_fallback_suggests_next_actions() -> None:
    graph = build_retrieval_subgraph(
        RetrievalSubgraphDependencies(
            retriever=EmptyRetriever(),
            reranker=NoopReranker(),
        )
    )

    result = graph.invoke(default_retrieval_subgraph_state("task_1", "如何配置不存在的模块？"))

    assert result["fallback_type"] == "retrieval_empty_fallback"
    assert "换一组" in result["sub_answer"]
    assert "上传相关技术文档" in result["sub_answer"]
    assert "指定文档名" in result["sub_answer"]
    assert any("retrieval_empty_fallback" in error for error in result["error_messages"])


def test_llm_json_parse_error_retries_once_then_succeeds() -> None:
    llm = RawLLM(
        [
            "not json",
            json.dumps(
                {
                    "rewritten_query": "如何配置 Qdrant hybrid search？",
                    "reason": "retry succeeded",
                    "missing_info": [],
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = query_rewrite_node(
        {"original_query": "如何配置 Qdrant hybrid search？", "chat_history": [], "compression_summary": ""},
        llm,
    )

    assert llm.calls == 2
    assert result["rewritten_query"] == "如何配置 Qdrant hybrid search？"
    assert any("json_parse_attempt_1_failed" in error for error in result["error_messages"])


def test_llm_json_parse_error_uses_safe_default_after_retry_exhausted() -> None:
    llm = RawLLM(["not json", "still not json"])

    result = query_rewrite_node(
        {"original_query": "这个怎么部署？", "chat_history": [], "compression_summary": ""},
        llm,
    )

    assert llm.calls == 2
    assert result["rewritten_query"] == "这个怎么部署？"
    assert result["fallback_type"] == "llm_parse_error_fallback"
    assert any("llm_parse_error_fallback" in error for error in result["error_messages"])


def test_ambiguity_json_parse_error_falls_back_safely() -> None:
    llm = RawLLM(["bad", "bad again"])

    result = ambiguity_detection_node({"rewritten_query": "这个怎么部署？"}, llm)

    assert result["fallback_type"] == "llm_parse_error_fallback"
    assert result["ambiguity_result"]["is_ambiguous"] is True
    assert any("llm_parse_error_fallback" in error for error in result["error_messages"])


def test_subgraph_exception_retries_and_main_graph_continues() -> None:
    subgraph = FlakyMainSubgraph()
    state = default_main_graph_state("复杂问题")
    state["rewritten_query"] = "复杂问题"
    state["decomposed_tasks"] = [
        {"sub_task_id": "task_1", "sub_query": "问题 1", "purpose": "definition", "priority": 1, "dependencies": []},
        {"sub_task_id": "task_2", "sub_query": "问题 2", "purpose": "definition", "priority": 2, "dependencies": []},
        {"sub_task_id": "task_3", "sub_query": "问题 3", "purpose": "definition", "priority": 3, "dependencies": []},
    ]
    graph = build_main_graph(MainGraphDependencies(retrieval_subgraph=subgraph, subgraph_retry_limit=1))

    result = graph.invoke(state)

    assert len(result["sub_answers"]) == 3
    assert subgraph.calls["task_2"] == 2
    assert any("sub_retrieval_graph_error_attempt_1" in error for error in result["error_messages"])
    assert "问题 3" in result["final_answer"]
