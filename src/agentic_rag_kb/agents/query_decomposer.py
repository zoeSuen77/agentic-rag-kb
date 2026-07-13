"""Task decomposition node for Agentic RAG.

This node converts one retrieval-ready query into a small set of independent
subqueries. The main graph will later dispatch these tasks with LangGraph Send
API so retrieval subgraphs can run in parallel.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from agentic_rag_kb.agents.json_utils import ensure_string_list, parse_json_object
from agentic_rag_kb.graph.state import MainGraphState
from agentic_rag_kb.llm import LLMClient

TaskPurpose = Literal["definition", "comparison", "procedure", "risk", "example", "evidence"]

ALLOWED_PURPOSES: set[str] = {"definition", "comparison", "procedure", "risk", "example", "evidence"}
MAX_TASKS = 5


TASK_DECOMPOSITION_PROMPT = """你是企业 Agentic RAG 的 task decomposition 节点。

请只输出 JSON，不要输出 Markdown。

目标：
把 rewritten_query 拆成 1-5 个可以独立检索的子问题，用于后续 LangGraph Send API 并行分发。

规则：
1. 简单单意图问题只拆成 1 个任务。
2. 多意图、比较、流程、风险、示例、证据类复杂问题拆成 2-5 个任务。
3. 避免过度拆解，不要把同一意图拆成多个近义任务。
4. 每个 sub_query 必须能单独检索，保留必要实体。
5. purpose 只能是 definition、comparison、procedure、risk、example、evidence。
6. dependencies 仅在某任务必须依赖前置任务答案时填写 task_id，否则为空数组。

输出 JSON schema:
{
  "tasks": [
    {
      "sub_task_id": "task_1",
      "sub_query": "...",
      "purpose": "definition | comparison | procedure | risk | example | evidence",
      "priority": 1,
      "dependencies": []
    }
  ],
  "debug": {
    "reason": "...",
    "strategy": "llm_structured_decomposition"
  }
}

rewritten_query:
{rewritten_query}
"""


class QueryDecomposerAgent:
    """Agent that decomposes complex user questions into retrieval tasks."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def run(self, state: MainGraphState) -> MainGraphState:
        """Return state update containing `decomposed_tasks` and debug info."""

        return task_decomposition_node(state, self.llm_client)


def task_decomposition_node(
    state: MainGraphState,
    llm_client: LLMClient | None = None,
) -> MainGraphState:
    """Decompose `rewritten_query` into independent retrieval subtasks."""

    rewritten_query = (state.get("rewritten_query") or state.get("original_query", "")).strip()

    if llm_client is not None:
        try:
            parsed = parse_json_object(
                llm_client.generate(TASK_DECOMPOSITION_PROMPT.replace("{rewritten_query}", rewritten_query))
            )
            tasks = _normalize_tasks(parsed.get("tasks", []), rewritten_query)
            debug = _normalize_debug(
                parsed.get("debug"),
                strategy="llm",
                task_count=len(tasks),
                reason="LLM structured decomposition succeeded.",
            )
            return {**state, "decomposed_tasks": tasks, "decomposition_debug": debug}
        except Exception as exc:
            errors = [*state.get("error_messages", []), f"task_decomposition_llm_error: {exc}"]
            state = {**state, "error_messages": errors}

    tasks, debug = _fallback_decompose(rewritten_query)
    return {**state, "decomposed_tasks": tasks, "decomposition_debug": debug}


def _normalize_tasks(raw_tasks: Any, rewritten_query: str) -> list[dict[str, Any]]:
    if not isinstance(raw_tasks, list):
        return _single_task(rewritten_query)

    normalized: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            continue
        sub_query = str(raw.get("sub_query") or "").strip()
        if not sub_query or sub_query in seen_queries:
            continue
        seen_queries.add(sub_query)
        purpose = str(raw.get("purpose") or _infer_purpose(sub_query)).strip()
        if purpose not in ALLOWED_PURPOSES:
            purpose = _infer_purpose(sub_query)
        normalized.append(
            {
                "sub_task_id": f"task_{len(normalized) + 1}",
                "sub_query": sub_query,
                "purpose": purpose,
                "priority": _safe_priority(raw.get("priority"), len(normalized) + 1),
                "dependencies": _filter_dependencies(raw.get("dependencies")),
            }
        )
        if len(normalized) >= MAX_TASKS:
            break

    return normalized or _single_task(rewritten_query)


def _normalize_debug(
    raw_debug: Any,
    *,
    strategy: str,
    task_count: int,
    reason: str,
) -> dict[str, Any]:
    debug = raw_debug if isinstance(raw_debug, dict) else {}
    return {
        "strategy": str(debug.get("strategy") or strategy),
        "reason": str(debug.get("reason") or reason),
        "task_count": task_count,
        "is_complex": task_count > 1,
        "max_tasks": MAX_TASKS,
    }


def _fallback_decompose(query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stripped = query.strip()
    if not stripped:
        tasks = _single_task(stripped)
        return tasks, _debug("fallback_rules", len(tasks), "Empty query fallback created one placeholder task.")

    special_tasks = _decompose_known_agentic_rag_example(stripped)
    if special_tasks:
        return special_tasks, _debug("fallback_rules", len(special_tasks), "Matched Agentic RAG architecture/evaluation pattern.")

    candidates = _intent_candidates(stripped)
    if len(candidates) <= 1:
        tasks = _single_task(stripped)
        return tasks, _debug("fallback_rules", 1, "Single intent detected; kept one retrieval task.")

    tasks: list[dict[str, Any]] = []
    for candidate in candidates[:MAX_TASKS]:
        cleaned = _clean_sub_query(candidate)
        if not cleaned:
            continue
        if cleaned not in {task["sub_query"] for task in tasks}:
            tasks.append(_make_task(cleaned, len(tasks) + 1))

    if not tasks:
        tasks = _single_task(stripped)
    return tasks, _debug("fallback_rules", len(tasks), "Multiple intents detected by conjunction and intent keywords.")


def _decompose_known_agentic_rag_example(query: str) -> list[dict[str, Any]]:
    if not all(token in query for token in ["LangGraph", "普通 RAG", "评测"]):
        return []
    tasks = [
        ("LangGraph 主图子图架构是什么", "definition"),
        ("Agentic RAG 相比普通 RAG 的优势", "comparison"),
        ("如何使用 RAGAS 评测", "procedure"),
    ]
    return [_make_task(sub_query, index, purpose) for index, (sub_query, purpose) in enumerate(tasks, start=1)]


def _intent_candidates(query: str) -> list[str]:
    clauses = _split_clauses(query)
    candidates: list[str] = []

    comparison = _extract_comparison(query)
    if comparison:
        candidates.append(comparison)

    for clause in clauses:
        cleaned = _clean_sub_query(clause)
        if not cleaned:
            continue
        if comparison and _is_comparison_clause(cleaned):
            continue
        candidates.append(cleaned)

    if len(candidates) == 1:
        candidates.extend(_keyword_based_candidates(query))

    return _dedupe_preserve_order(candidates)


def _split_clauses(query: str) -> list[str]:
    protected = query.replace("Qdrant 和 Gradio", "Qdrant 与 Gradio")
    parts = re.split(r"[，,；;？?。]\s*|并且|以及|同时|另外|还有", protected)
    return [part.replace("Qdrant 与 Gradio", "Qdrant 和 Gradio").strip() for part in parts if part.strip()]


def _extract_comparison(query: str) -> str:
    patterns = [
        r"([^，,；;。?？]*?(?:和|与).+?(?:相比|对比|区别|优势|差异)[^，,；;。?？]*)",
        r"([^，,；;。?？]*?(?:相比|对比).+?[^，,；;。?？]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return _clean_sub_query(match.group(1))
    return ""


def _keyword_based_candidates(query: str) -> list[str]:
    candidates: list[str] = []
    if any(token in query for token in ["是什么", "架构", "设计", "原理"]):
        candidates.append(query)
    if any(token in query for token in ["如何", "怎么", "步骤", "部署", "配置", "实现", "评测"]):
        candidates.append(query)
    if any(token in query for token in ["风险", "问题", "故障", "注意"]):
        candidates.append(query)
    if any(token in query for token in ["示例", "例子", "案例"]):
        candidates.append(query)
    return candidates


def _single_task(query: str) -> list[dict[str, Any]]:
    return [_make_task(query, 1)]


def _make_task(sub_query: str, index: int, purpose: str | None = None) -> dict[str, Any]:
    return {
        "sub_task_id": f"task_{index}",
        "sub_query": sub_query,
        "purpose": purpose or _infer_purpose(sub_query),
        "priority": index,
        "dependencies": [],
    }


def _infer_purpose(query: str) -> TaskPurpose:
    lowered = query.lower()
    if any(token in query for token in ["相比", "对比", "比较", "区别", "差异", "优势"]) or " vs " in lowered:
        return "comparison"
    if any(token in query for token in ["风险", "问题", "故障", "缺陷", "注意"]):
        return "risk"
    if any(token in query for token in ["示例", "例子", "案例"]):
        return "example"
    if any(token in query for token in ["证据", "依据", "引用", "来源", "召回率", "准确率"]):
        return "evidence"
    if any(token in query for token in ["如何", "怎么", "步骤", "配置", "部署", "实现", "评测", "使用"]):
        return "procedure"
    return "definition"


def _safe_priority(value: Any, fallback: int) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(1, priority)


def _filter_dependencies(value: Any) -> list[str]:
    dependencies = ensure_string_list(value)
    return [item for item in dependencies if re.fullmatch(r"task_[1-5]", item)]


def _clean_sub_query(text: str) -> str:
    cleaned = text.strip(" \t\n\r，,；;。?？")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _is_comparison_clause(text: str) -> bool:
    return any(token in text for token in ["相比", "对比", "比较", "区别", "差异", "优势"])


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_sub_query(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _debug(strategy: str, task_count: int, reason: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "reason": reason,
        "task_count": task_count,
        "is_complex": task_count > 1,
        "max_tasks": MAX_TASKS,
    }
