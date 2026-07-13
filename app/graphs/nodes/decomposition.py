from __future__ import annotations

import re

from app.graphs.states import MainGraphState


def classify_query(state: MainGraphState) -> MainGraphState:
    query = state.get("normalized_query") or state.get("raw_query", "")
    lowered = query.lower()
    if any(term in lowered for term in ["why", "root cause", "原因", "故障", "失败", "502", "timeout"]):
        state["query_type"] = "troubleshooting"
    elif any(term in lowered for term in ["compare", "区别", "对比"]):
        state["query_type"] = "comparison"
    elif any(term in lowered for term in ["how", "如何", "怎么"]):
        state["query_type"] = "how_to"
    else:
        state["query_type"] = "fact_lookup"
    return state


def detect_complexity(state: MainGraphState) -> MainGraphState:
    query = state.get("normalized_query") or state.get("raw_query", "")
    separators = len(re.findall(r"[，,;；]| and | or |以及|并且|同时", query, flags=re.IGNORECASE))
    state["complexity_level"] = "complex" if len(query) > 80 or separators >= 2 else "medium" if separators else "simple"
    return state


def decompose_question(state: MainGraphState) -> MainGraphState:
    query = state.get("normalized_query") or state.get("raw_query", "")
    if state.get("complexity_level") == "simple":
        state["decomposed_questions"] = [
            {"sub_query_id": "sq_1", "question": query, "intent": state.get("query_type", "fact_lookup"), "priority": 1}
        ]
        return state

    candidates = re.split(r"[，,;；]|以及|并且|同时| and | or ", query)
    questions = [part.strip() for part in candidates if len(part.strip()) >= 4]
    if len(questions) <= 1:
        questions = [
            f"背景和核心概念：{query}",
            f"关键步骤或机制：{query}",
            f"风险、限制和排查建议：{query}",
        ]
    state["decomposed_questions"] = [
        {"sub_query_id": f"sq_{index}", "question": question, "intent": state.get("query_type", "fact_lookup"), "priority": index}
        for index, question in enumerate(questions[:5], start=1)
    ]
    return state

