from __future__ import annotations

from app.graphs.states import MainGraphState


AMBIGUOUS_TERMS = {"it", "this", "that", "这个", "那个", "它", "回滚", "报错", "失败"}


def detect_ambiguity(state: MainGraphState) -> MainGraphState:
    query = state.get("raw_query", "").strip()
    tokens = set(query.lower().split())
    has_short_reference = len(query) < 12 and any(term in query.lower() for term in AMBIGUOUS_TERMS)
    ambiguous = has_short_reference or bool(tokens & {"rollback", "error", "failed"} and len(tokens) < 4)
    result = {
        "is_ambiguous": ambiguous,
        "reason": "Missing target system or environment" if ambiguous else "",
        "missing_slots": ["system", "environment"] if ambiguous else [],
        "confidence": 0.78 if ambiguous else 0.12,
    }
    state["ambiguity_result"] = result
    if ambiguous:
        state["clarification_question"] = "请补充目标系统、环境和具体对象，例如 Kubernetes Deployment、数据库迁移或应用发布。"
    return state

