"""Tests for LangGraph state defaults and reducers."""

from agentic_rag_kb.graph.schema import (
    MAIN_GRAPH_NODE_IO,
    RETRIEVAL_SUBGRAPH_NODE_IO,
    default_main_graph_state,
    default_retrieval_subgraph_state,
)
from agentic_rag_kb.graph.state import (
    append_error_messages,
    append_retrieved_contexts,
    append_sub_answers,
    merge_retrieval_debug,
)


def test_default_main_graph_state_values() -> None:
    """Main graph defaults should provide stable empty values."""

    state = default_main_graph_state("how to configure db pool?")

    assert state["original_query"] == "how to configure db pool?"
    assert state["rewritten_query"] == ""
    assert state["chat_history"] == []
    assert state["ambiguity_result"] == {}
    assert state["decomposed_tasks"] == []
    assert state["decomposition_debug"] == {}
    assert state["sub_answers"] == []
    assert state["retrieved_contexts"] == []
    assert state["loop_count"] == 0
    assert state["error_messages"] == []
    assert state["retrieval_debug"] == {}


def test_default_retrieval_subgraph_state_values() -> None:
    """Retrieval subgraph defaults should provide stable empty values."""

    state = default_retrieval_subgraph_state("task_1", "database pool")

    assert state["sub_task_id"] == "task_1"
    assert state["sub_query"] == "database pool"
    assert state["rewritten_sub_query"] == ""
    assert state["retrieved_chunks"] == []
    assert state["reranked_contexts"] == []
    assert state["sub_answer"] == ""
    assert state["confidence"] == 0.0
    assert state["insufficient_context"] is False
    assert state["debug"] == {}
    assert state["error_messages"] == []


def test_append_sub_answers_reducer() -> None:
    """Subanswers should append in fan-in order."""

    merged = append_sub_answers(
        [{"sub_task_id": "a", "answer": "A"}],
        [{"sub_task_id": "b", "answer": "B"}],
    )

    assert [item["sub_task_id"] for item in merged] == ["a", "b"]


def test_append_retrieved_contexts_reducer_dedupes_parent_id() -> None:
    """Retrieved contexts should dedupe by parent_id."""

    merged = append_retrieved_contexts(
        [{"parent_id": "p1", "text": "one"}],
        [{"parent_id": "p1", "text": "duplicate"}, {"parent_id": "p2", "text": "two"}],
    )

    assert [item["parent_id"] for item in merged] == ["p1", "p2"]
    assert merged[0]["text"] == "one"


def test_append_error_messages_reducer() -> None:
    """Error messages should accumulate."""

    assert append_error_messages(["main error"], ["subgraph error"]) == [
        "main error",
        "subgraph error",
    ]


def test_merge_retrieval_debug_reducer() -> None:
    """Retrieval debug should merge nested dicts and append lists."""

    merged = merge_retrieval_debug(
        {"dense_hits": [{"child_id": "c1"}], "stats": {"dense": 1}},
        {"dense_hits": [{"child_id": "c2"}], "stats": {"sparse": 1}},
    )

    assert merged["dense_hits"] == [{"child_id": "c1"}, {"child_id": "c2"}]
    assert merged["stats"] == {"dense": 1, "sparse": 1}


def test_node_io_specs_are_declared() -> None:
    """State schema should document main graph and subgraph node inputs and outputs."""

    main_node_names = {spec.node_name for spec in MAIN_GRAPH_NODE_IO}
    sub_node_names = {spec.node_name for spec in RETRIEVAL_SUBGRAPH_NODE_IO}

    assert "detect_ambiguity" in main_node_names
    assert "dispatch_retrieval_subgraphs" in main_node_names
    assert "hybrid_retrieve" in sub_node_names
    assert "confidence_check" in sub_node_names
    assert all(spec.inputs or spec.outputs for spec in MAIN_GRAPH_NODE_IO)
