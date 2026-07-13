from __future__ import annotations

from app.graphs.states import MainGraphState


def generate_final_answer(state: MainGraphState) -> MainGraphState:
    if state.get("fallback_reason"):
        return state
    citations = state.get("citations", [])
    citation_text = "\n".join(
        f"[{index}] {item.get('source') or item.get('doc_id')} parent={item.get('parent_id')}"
        for index, item in enumerate(citations, start=1)
    )
    state["final_answer"] = (
        "结论：\n"
        f"{state.get('aggregated_answer', '').strip() or '未生成答案。'}\n\n"
        "引用：\n"
        f"{citation_text or '无'}"
    )
    return state

