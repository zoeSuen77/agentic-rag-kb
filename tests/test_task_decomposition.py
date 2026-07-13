"""Tests for task decomposition before Send API fan-out."""

from __future__ import annotations

import json

from agentic_rag_kb.agents.query_decomposer import QueryDecomposerAgent, task_decomposition_node
from agentic_rag_kb.graph.schema import default_main_graph_state


class FakeLLMClient:
    """Tiny fake LLM client returning deterministic structured JSON."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def generate(self, prompt: str) -> str:
        assert "rewritten_query" in prompt
        return json.dumps(self.payload, ensure_ascii=False)


def test_single_intent_question_creates_one_task() -> None:
    state = default_main_graph_state("如何配置 Qdrant collection？")
    state["rewritten_query"] = "如何配置 Qdrant collection？"

    result = task_decomposition_node(state)

    assert len(result["decomposed_tasks"]) == 1
    assert result["decomposed_tasks"][0]["sub_task_id"] == "task_1"
    assert result["decomposed_tasks"][0]["purpose"] == "procedure"
    assert result["decomposed_tasks"][0]["dependencies"] == []
    assert result["decomposition_debug"]["is_complex"] is False


def test_multi_intent_question_decomposes_agentic_rag_example() -> None:
    state = default_main_graph_state()
    state["rewritten_query"] = "LangGraph 的主图子图怎么设计，和普通 RAG 相比有什么优势，如何评测？"

    result = task_decomposition_node(state)
    tasks = result["decomposed_tasks"]

    assert [task["sub_query"] for task in tasks] == [
        "LangGraph 主图子图架构是什么",
        "Agentic RAG 相比普通 RAG 的优势",
        "如何使用 RAGAS 评测",
    ]
    assert [task["purpose"] for task in tasks] == ["definition", "comparison", "procedure"]
    assert result["decomposition_debug"]["task_count"] == 3


def test_comparison_question_has_comparison_task() -> None:
    state = default_main_graph_state()
    state["rewritten_query"] = "Agentic RAG 和普通 RAG 有什么区别，分别适合什么场景？"

    result = task_decomposition_node(state)
    tasks = result["decomposed_tasks"]

    assert 1 <= len(tasks) <= 5
    assert any(task["purpose"] == "comparison" for task in tasks)
    assert any("Agentic RAG" in task["sub_query"] for task in tasks)
    assert result["decomposition_debug"]["task_count"] == len(tasks)


def test_procedure_question_splits_flow_steps_without_over_decomposition() -> None:
    state = default_main_graph_state()
    state["rewritten_query"] = "如何部署 Qdrant、Gradio，并配置数据库连接池？"

    result = task_decomposition_node(state)
    tasks = result["decomposed_tasks"]

    assert 2 <= len(tasks) <= 5
    assert all(task["purpose"] == "procedure" for task in tasks)
    assert all(task["sub_query"] for task in tasks)
    assert [task["priority"] for task in tasks] == list(range(1, len(tasks) + 1))


def test_llm_structured_output_is_normalized() -> None:
    llm = FakeLLMClient(
        {
            "tasks": [
                {
                    "sub_task_id": "custom_id",
                    "sub_query": "如何使用 RAGAS 评测 ContextRecall？",
                    "purpose": "unknown",
                    "priority": "3",
                    "dependencies": ["task_9", "task_1"],
                },
                {
                    "sub_query": "如何使用 RAGAS 评测 ContextRecall？",
                    "purpose": "procedure",
                },
            ],
            "debug": {"reason": "structured", "strategy": "llm_structured_decomposition"},
        }
    )
    state = default_main_graph_state()
    state["rewritten_query"] = "如何使用 RAGAS 评测 ContextRecall？"

    result = QueryDecomposerAgent(llm).run(state)
    tasks = result["decomposed_tasks"]

    assert len(tasks) == 1
    assert tasks[0]["sub_task_id"] == "task_1"
    assert tasks[0]["purpose"] == "procedure"
    assert tasks[0]["priority"] == 3
    assert tasks[0]["dependencies"] == ["task_1"]
    assert result["decomposition_debug"]["strategy"] == "llm_structured_decomposition"


def test_decomposition_debug_info_exists_for_fallback() -> None:
    state = default_main_graph_state()
    state["rewritten_query"] = "如何配置数据库连接池，并说明常见风险和示例？"

    result = task_decomposition_node(state)
    debug = result["decomposition_debug"]

    assert debug["strategy"] == "fallback_rules"
    assert debug["task_count"] == len(result["decomposed_tasks"])
    assert isinstance(debug["reason"], str)
    assert debug["max_tasks"] == 5
