"""Minimal Human-in-the-loop LangGraph demo.

The production path builds a LangGraph graph that uses `interrupt` in
`clarification_node`. A lightweight local simulator is provided so the demo can
run in environments where LangGraph is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_rag_kb.agents.ambiguity import ambiguity_detection_node
from agentic_rag_kb.agents.clarification import (
    MAX_CLARIFICATION_LOOPS,
    clarification_node,
    fallback_node,
    should_fallback,
)
from agentic_rag_kb.agents.query_rewrite import query_rewrite_node
from agentic_rag_kb.graph.state import MainGraphState


def build_hitl_demo_graph():
    """Build a minimal LangGraph HITL demo graph.

    The graph flow is:
    query_rewrite -> ambiguity_detection -> clarification interrupt or finish.
    After resume, clarification returns `user_clarification` and loops back to
    query_rewrite.
    """

    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install langgraph to build the real HITL demo graph.") from exc

    graph = StateGraph(MainGraphState)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("ambiguity_detection", ambiguity_detection_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("finish", _finish_node)

    graph.add_edge(START, "query_rewrite")
    graph.add_edge("query_rewrite", "ambiguity_detection")
    graph.add_conditional_edges(
        "ambiguity_detection",
        _route_after_ambiguity,
        {
            "clarification": "clarification",
            "fallback": "fallback",
            "finish": "finish",
        },
    )
    graph.add_edge("clarification", "query_rewrite")
    graph.add_edge("fallback", END)
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=MemorySaver())


@dataclass(slots=True)
class LocalInterrupt:
    """Local stand-in for a LangGraph interrupt event."""

    clarification_question: str
    state: MainGraphState


class LocalHITLDemo:
    """Small local simulator for HITL behavior when LangGraph is unavailable."""

    def start(self, original_query: str) -> LocalInterrupt | MainGraphState:
        """Run until clarification is needed or the query is clear."""

        state: MainGraphState = {
            "original_query": original_query,
            "chat_history": [],
            "compression_summary": "",
            "loop_count": 0,
            "error_messages": [],
        }
        state = query_rewrite_node(state)
        state = ambiguity_detection_node(state)
        if should_fallback(state):
            return fallback_node(state)
        if state.get("ambiguity_result", {}).get("is_ambiguous", False):
            return LocalInterrupt(
                clarification_question=state["ambiguity_result"]["clarification_question"],
                state=state,
            )
        return _finish_node(state)

    def resume(self, interrupt_event: LocalInterrupt, user_input: str) -> MainGraphState:
        """Resume after a human clarification."""

        state = {
            **interrupt_event.state,
            "user_clarification": user_input,
            "loop_count": int(interrupt_event.state.get("loop_count", 0)) + 1,
        }
        state = query_rewrite_node(state)
        state = ambiguity_detection_node(state)
        if state.get("ambiguity_result", {}).get("is_ambiguous", False):
            if int(state.get("loop_count", 0)) >= MAX_CLARIFICATION_LOOPS:
                return fallback_node(state)
            return {
                **state,
                "final_answer": state["ambiguity_result"]["clarification_question"],
            }
        return _finish_node(state)


def _route_after_ambiguity(state: MainGraphState) -> str:
    if should_fallback(state):
        return "fallback"
    if state.get("ambiguity_result", {}).get("is_ambiguous", False):
        return "clarification"
    return "finish"


def _finish_node(state: MainGraphState) -> MainGraphState:
    return {
        **state,
        "final_answer": (
            "问题已澄清，可以继续进入后续任务拆解与检索。"
            f"\n重写后的问题：{state.get('rewritten_query', '')}"
        ),
    }

