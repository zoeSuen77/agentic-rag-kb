from __future__ import annotations

from app.graphs.states import MainGraphState


def compress_context(state: MainGraphState) -> MainGraphState:
    history = state.get("conversation_history", [])
    if len(history) <= 6:
        state["compressed_history"] = ""
        return state
    recent = history[-6:]
    state["compressed_history"] = "Recent confirmed context: " + " | ".join(
        str(item.get("content", ""))[:160] for item in recent
    )
    return state

