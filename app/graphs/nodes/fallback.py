from __future__ import annotations

from app.graphs.states import MainGraphState


def apply_fallback(state: MainGraphState) -> MainGraphState:
    reason = state.get("fallback_reason") or "Evidence is insufficient."
    state["final_answer"] = (
        "当前知识库中没有找到足够可靠的证据来完整回答这个问题。\n\n"
        f"原因：{reason}\n\n"
        "建议：请补充更具体的系统、版本、错误日志或上传相关技术文档后重试。"
    )
    return state

