from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SendTask:
    target: str
    payload: dict[str, Any]


def build_send_tasks(decomposed_questions: list[dict[str, Any]], parent_query: str) -> list[SendTask]:
    return [
        SendTask(
            target="sub_retrieval_graph",
            payload={
                "parent_query": parent_query,
                "sub_query": item["question"],
                "sub_query_id": item["sub_query_id"],
                "intent": item.get("intent", "fact_lookup"),
            },
        )
        for item in decomposed_questions
    ]
