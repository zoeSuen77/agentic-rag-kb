from __future__ import annotations


def rewrite_sub_query(parent_query: str, sub_query: str, intent: str | None = None) -> str:
    parts = [sub_query.strip()]
    if parent_query and parent_query.strip() != sub_query.strip():
        parts.append(f"Original question context: {parent_query.strip()}")
    if intent:
        parts.append(f"Intent: {intent}")
    return "\n".join(parts)

