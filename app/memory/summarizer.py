from __future__ import annotations


def summarize_messages(messages: list[dict], max_chars: int = 1200) -> str:
    joined = " | ".join(f"{item.get('role')}: {item.get('content')}" for item in messages)
    return joined[:max_chars]

