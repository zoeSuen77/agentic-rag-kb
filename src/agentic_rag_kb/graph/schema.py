"""Default state factories and node I/O contracts for Agentic RAG graphs.

This file intentionally documents the graph shape without building the graph. The
next implementation step can wire these contracts into LangGraph nodes and edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_rag_kb.graph.state import MainGraphState, RetrievalSubGraphState


def default_main_graph_state(original_query: str = "") -> MainGraphState:
    """Create a default main graph state."""

    return MainGraphState(
        original_query=original_query,
        rewritten_query="",
        chat_history=[],
        ambiguity_result={},
        clarification_question="",
        user_clarification="",
        decomposed_tasks=[],
        decomposition_debug={},
        sub_answers=[],
        retrieved_contexts=[],
        final_answer="",
        loop_count=0,
        error_messages=[],
        compression_summary="",
        retrieval_debug={},
    )


def default_retrieval_subgraph_state(
    sub_task_id: str = "",
    sub_query: str = "",
) -> RetrievalSubGraphState:
    """Create a default retrieval subgraph state."""

    return RetrievalSubGraphState(
        sub_task_id=sub_task_id,
        sub_query=sub_query,
        rewritten_sub_query="",
        retrieved_chunks=[],
        reranked_contexts=[],
        sub_answer="",
        confidence=0.0,
        error_messages=[],
    )


@dataclass(slots=True)
class NodeIOSpec:
    """Human-readable input/output contract for a graph node."""

    node_name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    description: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable node contract."""

        return asdict(self)


MAIN_GRAPH_NODE_IO: list[NodeIOSpec] = [
    NodeIOSpec(
        node_name="load_memory",
        inputs=["original_query", "chat_history"],
        outputs=["chat_history", "compression_summary"],
        description="Load recent conversation turns and any compressed summary.",
    ),
    NodeIOSpec(
        node_name="detect_ambiguity",
        inputs=["original_query", "chat_history", "compression_summary"],
        outputs=["ambiguity_result", "clarification_question"],
        description="Detect missing entities, versions, environments, or unsafe ambiguity.",
    ),
    NodeIOSpec(
        node_name="apply_clarification",
        inputs=["original_query", "user_clarification", "ambiguity_result"],
        outputs=["rewritten_query"],
        description="Merge human clarification into the query when required.",
    ),
    NodeIOSpec(
        node_name="decompose_query",
        inputs=["rewritten_query", "compression_summary"],
        outputs=["decomposed_tasks", "decomposition_debug"],
        description="Break complex questions into independent retrieval tasks.",
    ),
    NodeIOSpec(
        node_name="dispatch_retrieval_subgraphs",
        inputs=["decomposed_tasks"],
        outputs=["sub_answers", "retrieved_contexts", "retrieval_debug", "error_messages"],
        description="Use LangGraph Send API to run retrieval subgraphs in parallel.",
    ),
    NodeIOSpec(
        node_name="aggregate_answers",
        inputs=["sub_answers", "retrieved_contexts"],
        outputs=["final_answer"],
        description="Aggregate subanswers and parent contexts into a final answer draft.",
    ),
    NodeIOSpec(
        node_name="fallback_or_finish",
        inputs=["final_answer", "loop_count", "error_messages"],
        outputs=["final_answer", "loop_count"],
        description="Apply loop limits and fallback behavior when evidence is insufficient.",
    ),
]


RETRIEVAL_SUBGRAPH_NODE_IO: list[NodeIOSpec] = [
    NodeIOSpec(
        node_name="rewrite_sub_query",
        inputs=["sub_task_id", "sub_query"],
        outputs=["rewritten_sub_query"],
        description="Rewrite the subquestion for retrieval.",
    ),
    NodeIOSpec(
        node_name="hybrid_retrieve",
        inputs=["rewritten_sub_query"],
        outputs=["retrieved_chunks"],
        description="Run Dense + Sparse hybrid retrieval and RRF fusion.",
    ),
    NodeIOSpec(
        node_name="rerank_contexts",
        inputs=["rewritten_sub_query", "retrieved_chunks"],
        outputs=["reranked_contexts"],
        description="Apply Cross-Encoder reranking over parent-expanded contexts.",
    ),
    NodeIOSpec(
        node_name="generate_sub_answer",
        inputs=["sub_query", "reranked_contexts"],
        outputs=["sub_answer", "confidence"],
        description="Generate a local answer grounded in reranked contexts.",
    ),
]
