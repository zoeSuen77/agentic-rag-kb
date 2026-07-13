from __future__ import annotations

from app.memory.summarizer import summarize_messages


def build_memory_context(history: list[dict], recent_window: int = 6) -> dict:
    return {
        "summary": summarize_messages(history[:-recent_window]) if len(history) > recent_window else "",
        "recent_messages": history[-recent_window:],
    }

