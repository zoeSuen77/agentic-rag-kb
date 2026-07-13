"""Tests for main graph Send API fan-out and reduce behavior."""

from __future__ import annotations

import pytest

from agentic_rag_kb.graph.main_graph import (
    MainGraphDependencies,
    answer_aggregation_node,
    build_main_graph,
    dispatch_retrieval_subgraphs,
)
from agentic_rag_kb.graph.schema import default_main_graph_state


class FakeSubRetrievalGraph:
    """Fake subgraph returning one answer and one context per task."""

    def __init__(self, fail_task_ids: set[str] | None = None) -> None:
        self.fail_task_ids = fail_task_ids or set()
        self.invocations: list[dict] = []

    def invoke(self, state: dict) -> dict:
        self.invocations.append(state)
        sub_task_id = state["sub_task_id"]
        if sub_task_id in self.fail_task_ids:
            raise RuntimeError(f"boom {sub_task_id}")
        source = f"docs/{sub_task_id}.md"
        return {
            **state,
            "reranked_contexts": [
                {
                    "child_id": f"child_{sub_task_id}",
                    "parent_id": f"parent_{sub_task_id}",
                    "text": f"context for {state['sub_query']}",
                    "score_dense": 0.8,
                    "score_sparse": 0.6,
                    "score_fused": 0.05,
                    "rerank_score": 0.9,
                    "metadata": {"source_path": source},
                    "parent": {"source_path": source},
                }
            ],
            "sub_answer": f"{state['sub_query']} 的答案。\n\n引用来源：{source}",
            "confidence": 0.9,
            "insufficient_context": False,
            "debug": {"fake": {"task": sub_task_id}},
            "error_messages": [],
        }


def _state_with_tasks(task_count: int) -> dict:
    state = default_main_graph_state("complex question")
    state["rewritten_query"] = "complex question"
    state["decomposed_tasks"] = [
        {
            "sub_task_id": f"task_{index}",
            "sub_query": f"子问题 {index}",
            "purpose": "definition",
            "priority": index,
            "dependencies": [],
        }
        for index in range(1, task_count + 1)
    ]
    return state


def test_one_task_runs_and_aggregates() -> None:
    subgraph = FakeSubRetrievalGraph()
    graph = build_main_graph(MainGraphDependencies(retrieval_subgraph=subgraph))

    result = graph.invoke(_state_with_tasks(1))

    assert len(subgraph.invocations) == 1
    assert len(result["sub_answers"]) == 1
    assert len(result["retrieved_contexts"]) == 1
    assert "子问题 1" in result["final_answer"]


def test_three_tasks_create_three_sends_and_aggregate_results() -> None:
    state = _state_with_tasks(3)
    sends = dispatch_retrieval_subgraphs(state)
    graph = build_main_graph(MainGraphDependencies(retrieval_subgraph=FakeSubRetrievalGraph()))

    result = graph.invoke(state)

    assert len(sends) == 3
    assert [send.node for send in sends] == ["run_sub_retrieval", "run_sub_retrieval", "run_sub_retrieval"]
    assert [send.arg["sub_task_id"] for send in sends] == ["task_1", "task_2", "task_3"]
    assert len(result["sub_answers"]) == 3
    assert len(result["retrieval_debug"]["subgraphs"]) == 3


def test_one_subgraph_failure_does_not_block_other_subgraphs() -> None:
    subgraph = FakeSubRetrievalGraph(fail_task_ids={"task_2"})
    graph = build_main_graph(MainGraphDependencies(retrieval_subgraph=subgraph))

    result = graph.invoke(_state_with_tasks(3))

    assert len(subgraph.invocations) == 3
    assert len(result["sub_answers"]) == 3
    assert len(result["retrieved_contexts"]) == 2
    assert any("sub_retrieval_graph_error[task_2]" in error for error in result["error_messages"])
    failed = [item for item in result["sub_answers"] if item["sub_task_id"] == "task_2"][0]
    assert failed["insufficient_context"] is True


def test_context_reducer_dedupes_parent_ids() -> None:
    state = default_main_graph_state("aggregate")
    state["sub_answers"] = [{"sub_task_id": "task_1", "sub_query": "a", "sub_answer": "A", "confidence": 0.8}]
    state["retrieved_contexts"] = [
        {"parent_id": "parent_1", "metadata": {"source_path": "docs/a.md"}},
        {"parent_id": "parent_1", "metadata": {"source_path": "docs/a-duplicate.md"}},
    ]

    result = answer_aggregation_node(state)

    assert result["aggregation_debug"]["citation_sources"] == ["docs/a.md", "docs/a-duplicate.md"]
    assert "引用来源" in result["final_answer"]


@pytest.mark.parametrize("task_count", [1, 3])
def test_dispatch_skips_empty_sub_queries(task_count: int) -> None:
    state = _state_with_tasks(task_count)
    state["decomposed_tasks"].append(
        {"sub_task_id": "empty", "sub_query": "", "purpose": "definition", "priority": 99, "dependencies": []}
    )

    sends = dispatch_retrieval_subgraphs(state)

    assert len(sends) == task_count
