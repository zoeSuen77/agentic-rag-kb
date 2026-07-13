"""Fallback responses for resilient Agentic RAG execution.

Fallbacks keep the system honest when ambiguity, retrieval, parsing, or
confidence problems prevent a reliable answer. They never pretend the answer is
known; each response explains the failure mode and gives a concrete next step.
"""

from __future__ import annotations

from typing import Any, Literal

FallbackType = Literal[
    "ambiguous_query_fallback",
    "retrieval_empty_fallback",
    "llm_parse_error_fallback",
    "low_confidence_fallback",
]


def build_fallback_response(
    fallback_type: FallbackType,
    *,
    query: str = "",
    details: str = "",
) -> dict[str, Any]:
    """Return a structured fallback payload."""

    builders = {
        "ambiguous_query_fallback": _ambiguous_query_fallback,
        "retrieval_empty_fallback": _retrieval_empty_fallback,
        "llm_parse_error_fallback": _llm_parse_error_fallback,
        "low_confidence_fallback": _low_confidence_fallback,
    }
    answer = builders[fallback_type](query, details)
    return {
        "fallback_type": fallback_type,
        "answer": answer,
        "error_message": fallback_type if not details else f"{fallback_type}: {details}",
    }


def safe_default_json(fallback_type: FallbackType, *, query: str = "") -> dict[str, Any]:
    """Return safe defaults for structured-output parse failures."""

    if fallback_type == "llm_parse_error_fallback":
        return {
            "rewritten_query": query,
            "reason": "LLM structured output parsing failed; used original query as safe default.",
            "missing_info": [],
            "fallback_type": fallback_type,
        }
    return {"fallback_type": fallback_type}


def _ambiguous_query_fallback(query: str, details: str) -> str:
    templates = [
        "请问【系统/模块】的【具体配置项】应该如何配置？",
        "【组件】出现【错误码/日志】时应该如何排查？",
        "在【时间范围】内，【服务/接口】的【指标】为什么异常？",
    ]
    return (
        "问题仍然不够明确，我不能可靠检索知识库并生成答案。\n\n"
        "你可以改成下面这样的提问模板：\n"
        + "\n".join(f"- {template}" for template in templates)
    )


def _retrieval_empty_fallback(query: str, details: str) -> str:
    return (
        f"当前上下文不足，知识库没有检索到能回答该子问题的上下文：{query or '未提供子问题'}。\n\n"
        "我不能在没有证据的情况下编造答案。你可以尝试：\n"
        "- 换一组更贴近文档原文的关键词；\n"
        "- 上传相关技术文档后重新索引；\n"
        "- 指定文档名、模块名或配置文件名后再问。"
    )


def _llm_parse_error_fallback(query: str, details: str) -> str:
    return (
        "模型返回的结构化结果无法解析，系统已使用安全默认值继续执行。\n\n"
        "这一步不会使用未验证的模型输出，也不会编造缺失字段。"
    )


def _low_confidence_fallback(query: str, details: str) -> str:
    return (
        f"当前检索证据对该子问题的置信度较低：{query or '未提供子问题'}。\n\n"
        "我会保留已有证据，但不会把它当作确定结论。建议补充更明确的模块、版本、错误日志或文档名。"
    )
