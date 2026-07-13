"""Ambiguity detection node for Agentic RAG.

The ambiguity detector decides whether the rewritten query is ready for task
decomposition or whether the system should ask a human clarification question.
"""

from __future__ import annotations

from typing import Any, Literal

from agentic_rag_kb.agents.json_utils import parse_json_object
from agentic_rag_kb.graph.state import MainGraphState
from agentic_rag_kb.llm import LLMClient

AmbiguityType = Literal["missing_entity", "missing_time", "broad_scope", "unclear_metric", "none"]


AMBIGUITY_PROMPT = """你是企业知识库 RAG 的 ambiguity detection 节点。

请只输出 JSON，不要输出 Markdown。

判断 rewritten_query 是否足够清晰，可以进入后续任务拆解。

歧义类型只能选择：
- missing_entity: 缺少系统、组件、对象、配置项等实体
- missing_time: 问题需要时间范围但未提供
- broad_scope: 范围过大，无法检索到聚焦上下文
- unclear_metric: 指标、口径、成功标准不清楚
- none: 问题清晰

输出 JSON schema:
{
  "is_ambiguous": true,
  "ambiguity_type": "missing_entity | missing_time | broad_scope | unclear_metric | none",
  "clarification_question": "...",
  "confidence": 0.0
}

rewritten_query:
{rewritten_query}
"""


class AmbiguityDetectionAgent:
    """Agent that decides whether the user query needs clarification."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def run(self, state: MainGraphState) -> MainGraphState:
        """Return ambiguity result and optional clarification question."""

        return ambiguity_detection_node(state, self.llm_client)


def ambiguity_detection_node(
    state: MainGraphState,
    llm_client: LLMClient | None = None,
) -> MainGraphState:
    """Detect ambiguity from `rewritten_query`."""

    rewritten_query = state.get("rewritten_query") or state.get("original_query", "")

    if llm_client is not None:
        try:
            parsed = parse_json_object(llm_client.generate(AMBIGUITY_PROMPT.format(rewritten_query=rewritten_query)))
            result = _normalize_ambiguity_result(parsed)
            return {
                **state,
                "ambiguity_result": result,
                "clarification_question": result["clarification_question"],
            }
        except Exception as exc:
            errors = [*state.get("error_messages", []), f"ambiguity_detection_llm_error: {exc}"]
            state = {**state, "error_messages": errors}

    result = _fallback_ambiguity_detection(rewritten_query)
    return {**state, "ambiguity_result": result, "clarification_question": result["clarification_question"]}


def _normalize_ambiguity_result(parsed: dict[str, Any]) -> dict[str, Any]:
    ambiguity_type = str(parsed.get("ambiguity_type") or "none")
    if ambiguity_type not in {"missing_entity", "missing_time", "broad_scope", "unclear_metric", "none"}:
        ambiguity_type = "none"
    is_ambiguous = bool(parsed.get("is_ambiguous", ambiguity_type != "none"))
    if ambiguity_type == "none":
        is_ambiguous = False
    confidence = float(parsed.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    clarification_question = str(parsed.get("clarification_question") or "").strip()
    if is_ambiguous and not clarification_question:
        clarification_question = _clarification_for_type(ambiguity_type)
    return {
        "is_ambiguous": is_ambiguous,
        "ambiguity_type": ambiguity_type,
        "clarification_question": clarification_question,
        "confidence": confidence,
    }


def _fallback_ambiguity_detection(query: str) -> dict[str, Any]:
    stripped = query.strip()
    lowered = stripped.lower()

    if not stripped:
        return _result("missing_entity", "请补充你要查询的系统、组件或问题。", 0.95)

    if "用户澄清：" in stripped and stripped.split("用户澄清：", 1)[1].strip():
        return {
            "is_ambiguous": False,
            "ambiguity_type": "none",
            "clarification_question": "",
            "confidence": 0.86,
        }

    if any(token in lowered for token in ["这个", "那个", "它", "this", "that", "it"]) and "上下文：" not in stripped:
        if "部署" in stripped:
            return _result("missing_entity", "你指的是哪个模块的部署？", 0.9)
        return _result("missing_entity", "你说的“这个/它”具体指哪个系统、组件或配置项？", 0.9)

    if any(token in lowered for token in ["最近", "近期", "趋势", "变化", "增长", "下降", "latency", "qps"]):
        if not any(token in lowered for token in ["今天", "昨天", "过去", "近7天", "近 7 天", "202", "一周", "30天"]):
            return _result("missing_time", "请补充需要分析的时间范围，例如今天、过去 7 天或具体日期。", 0.82)

    if any(token in lowered for token in ["性能", "效果", "质量", "好不好", "是否正常", "metric", "指标"]):
        if not any(token in lowered for token in ["延迟", "吞吐", "准确率", "召回率", "错误率", "latency", "throughput"]):
            return _result("unclear_metric", "请说明你关心的具体指标或判断口径。", 0.78)

    broad_patterns = ["介绍一下", "讲一下", "说说", "全部", "所有", "整体", "架构是什么", "怎么优化"]
    if any(pattern in lowered for pattern in broad_patterns) and len(stripped) < 30:
        return _result("broad_scope", "请缩小问题范围，例如指定模块、场景、错误现象或配置项。", 0.76)

    return {
        "is_ambiguous": False,
        "ambiguity_type": "none",
        "clarification_question": "",
        "confidence": 0.88,
    }


def _result(ambiguity_type: AmbiguityType, question: str, confidence: float) -> dict[str, Any]:
    return {
        "is_ambiguous": True,
        "ambiguity_type": ambiguity_type,
        "clarification_question": question,
        "confidence": confidence,
    }


def _clarification_for_type(ambiguity_type: str) -> str:
    questions = {
        "missing_entity": "请补充具体系统、组件、配置项或错误对象。",
        "missing_time": "请补充需要分析的时间范围。",
        "broad_scope": "请缩小问题范围，说明具体模块或场景。",
        "unclear_metric": "请说明你关心的具体指标或判断口径。",
    }
    return questions.get(ambiguity_type, "")
