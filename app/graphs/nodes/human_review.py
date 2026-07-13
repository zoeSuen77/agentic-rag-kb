from __future__ import annotations

from app.graphs.states import MainGraphState


def apply_human_clarification(state: MainGraphState) -> MainGraphState:
    raw_query = state.get("raw_query", "")
    clarification = state.get("human_clarification", "")
    if clarification:
        state["normalized_query"] = f"{raw_query}\nClarification: {clarification}"
    else:
        state["normalized_query"] = raw_query
    return state

