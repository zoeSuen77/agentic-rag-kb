"""Human-in-the-loop clarification and fallback nodes.

These nodes are intentionally small and graph-friendly. `clarification_node`
uses LangGraph's `interrupt` when available, so execution can pause and resume
with a human clarification. The fallback node is used after the clarification
loop limit is reached.
"""

from __future__ import annotations

from typing import Any

from agentic_rag_kb.agents.fallback import build_fallback_response
from agentic_rag_kb.graph.state import MainGraphState


MAX_CLARIFICATION_LOOPS = 2


def clarification_node(state: MainGraphState) -> MainGraphState:
    """Pause for human clarification when the query is ambiguous.

    In a real LangGraph run, `interrupt(...)` pauses execution and returns the
    human-provided resume value when the graph resumes.
    """

    ambiguity_result = state.get("ambiguity_result", {})
    if not ambiguity_result.get("is_ambiguous", False):
        return state

    loop_count = int(state.get("loop_count", 0))
    if loop_count >= MAX_CLARIFICATION_LOOPS:
        return fallback_node(state)

    question = (
        ambiguity_result.get("clarification_question")
        or state.get("clarification_question")
        or "请补充你要查询的具体系统、模块或配置项。"
    )

    human_value = _langgraph_interrupt(
        {
            "type": "clarification_required",
            "clarification_question": question,
            "loop_count": loop_count,
            "max_loops": MAX_CLARIFICATION_LOOPS,
        }
    )
    return {
        **state,
        "clarification_question": question,
        "user_clarification": str(human_value).strip(),
        "loop_count": loop_count + 1,
    }


def fallback_node(state: MainGraphState) -> MainGraphState:
    """Return a fallback answer after too many clarification loops."""

    fallback = build_fallback_response(
        "ambiguous_query_fallback",
        query=state.get("original_query", ""),
        details="clarification_loop_limit_exceeded",
    )
    return {
        **state,
        "final_answer": fallback["answer"],
        "fallback_type": fallback["fallback_type"],
        "error_messages": [
            *state.get("error_messages", []),
            "clarification_loop_limit_exceeded",
            fallback["error_message"],
        ],
    }


def should_clarify(state: MainGraphState) -> bool:
    """Return whether the graph should enter the clarification path."""

    return bool(state.get("ambiguity_result", {}).get("is_ambiguous", False))


def should_fallback(state: MainGraphState) -> bool:
    """Return whether clarification loop limit has been reached."""

    return should_clarify(state) and int(state.get("loop_count", 0)) >= MAX_CLARIFICATION_LOOPS


def _langgraph_interrupt(payload: dict[str, Any]) -> Any:
    try:
        from langgraph.types import interrupt
    except ImportError as exc:  # pragma: no cover - exercised by demo fallback path
        raise RuntimeError("LangGraph is required for real interrupt execution.") from exc
    return interrupt(payload)
