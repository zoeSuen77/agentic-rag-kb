"""Main LangGraph workflow with Send API fan-out.

The main graph performs the map-reduce portion of Agentic RAG:

1. decompose a rewritten query into independent retrieval tasks;
2. use LangGraph `Send` to dispatch each task to the same retrieval subgraph;
3. reduce subgraph outputs into `sub_answers`, `retrieved_contexts`, and debug;
4. aggregate subanswers into a final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol

from agentic_rag_kb.agents.query_decomposer import task_decomposition_node
from agentic_rag_kb.graph.schema import default_retrieval_subgraph_state
from agentic_rag_kb.graph.state import (
    MainGraphState,
    RetrievalSubGraphState,
    append_error_messages,
    append_retrieved_contexts,
    append_sub_answers,
    merge_retrieval_debug,
)
from agentic_rag_kb.memory import CompressionTrigger, ConversationCompressor

try:  # pragma: no cover - exercised when langgraph is installed.
    from langgraph.constants import Send
except ImportError:  # pragma: no cover - local test environment may not install langgraph.
    try:
        from langgraph.types import Send  # type: ignore
    except ImportError:

        @dataclass(slots=True)
        class Send:  # type: ignore[no-redef]
            """Small compatibility object matching LangGraph Send's public shape."""

            node: str
            arg: dict[str, Any]


class InvokableGraph(Protocol):
    """Protocol shared by compiled LangGraph objects and local test doubles."""

    def invoke(self, state: RetrievalSubGraphState) -> RetrievalSubGraphState:
        """Run a retrieval subgraph."""


class _NoopGraph:
    """No-op graph used only when testing memory nodes in isolation."""

    def invoke(self, state: RetrievalSubGraphState) -> RetrievalSubGraphState:
        return state


@dataclass(slots=True)
class MainGraphDependencies:
    """Dependencies injected into main graph nodes."""

    retrieval_subgraph: InvokableGraph
    llm_client: Any | None = None
    memory_trigger: CompressionTrigger | None = None
    memory_compressor: ConversationCompressor | None = None
    subgraph_retry_limit: int = 1


MAIN_GRAPH_MERMAID = """```mermaid
flowchart TD
    A["original_query / rewritten_query"] --> B["task_decomposition_node"]
    B --> C{"dispatch_retrieval_subgraphs<br/>LangGraph Send API"}
    C -->|Send task_1| S1["sub_retrieval_graph"]
    C -->|Send task_2| S2["sub_retrieval_graph"]
    C -->|Send task_n| S3["sub_retrieval_graph"]
    S1 --> R["reducers<br/>sub_answers + retrieved_contexts + retrieval_debug"]
    S2 --> R
    S3 --> R
    R --> D["answer_aggregation_node"]
    D --> E["final_answer"]

    subgraph cluster_sub["sub_retrieval_graph"]
        Q["sub_query"] --> W["rewrite_sub_query_node"]
        W --> H["hybrid_retrieve_node"]
        H --> X["rerank_node"]
        X --> G["generate_sub_answer_node"]
        G --> F["confidence_check_node"]
    end
```"""


def build_main_graph(dependencies: MainGraphDependencies | None = None):
    """Build and compile the main LangGraph Send API workflow."""

    if dependencies is None:
        raise ValueError("MainGraphDependencies with a retrieval_subgraph is required.")

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        return SequentialMainGraph(dependencies, import_error=exc)

    workflow = StateGraph(MainGraphState)
    workflow.add_node("memory_check", lambda state: memory_check_node(state, dependencies))
    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("run_sub_retrieval", lambda state: run_sub_retrieval_node(state, dependencies))
    workflow.add_node("aggregate_answers", lambda state: answer_aggregation_node(state, dependencies.llm_client))
    workflow.add_node("memory_update", memory_update_node)

    workflow.add_edge(START, "memory_check")
    workflow.add_edge("memory_check", "decompose_query")
    workflow.add_conditional_edges("decompose_query", dispatch_retrieval_subgraphs, ["run_sub_retrieval"])
    workflow.add_edge("run_sub_retrieval", "aggregate_answers")
    workflow.add_edge("aggregate_answers", "memory_update")
    workflow.add_edge("memory_update", END)
    return workflow.compile()


class SequentialMainGraph:
    """Fallback runner that mirrors the Send fan-out/fan-in flow without LangGraph."""

    def __init__(
        self,
        dependencies: MainGraphDependencies,
        import_error: ImportError | None = None,
    ) -> None:
        self.dependencies = dependencies
        self.import_error = import_error

    def invoke(self, state: MainGraphState) -> MainGraphState:
        """Run decompose -> Send map -> reducer fan-in -> aggregate."""

        current = memory_check_node(state, self.dependencies)
        current = decompose_query_node(current)
        sends = dispatch_retrieval_subgraphs(current)
        updates = [run_sub_retrieval_node(send.arg, self.dependencies) for send in sends]
        reduced = _reduce_main_updates(current, updates)
        if self.import_error is not None:
            reduced["retrieval_debug"] = merge_retrieval_debug(
                reduced.get("retrieval_debug"),
                {
                    "main_graph": {
                        "runner": "sequential_fallback",
                        "reason": str(self.import_error),
                        "send_count": len(sends),
                    }
                },
            )
        aggregated = answer_aggregation_node(reduced, self.dependencies.llm_client)
        return memory_update_node(aggregated)


def memory_check_node(
    state: MainGraphState,
    dependencies: MainGraphDependencies | None = None,
) -> MainGraphState:
    """Compress long chat history before retrieval planning."""

    dependencies = dependencies or MainGraphDependencies(retrieval_subgraph=_NoopGraph())
    trigger = dependencies.memory_trigger or CompressionTrigger()
    compressor = dependencies.memory_compressor or ConversationCompressor()
    should_compress, trigger_debug = trigger.should_compress(
        turns=state.get("chat_history", []),
        compression_summary=state.get("compression_summary", ""),
        recent_contexts=state.get("retrieved_contexts", []),
    )
    memory_debug = {
        **state.get("memory_debug", {}),
        "memory_check": {
            "should_compress": should_compress,
            **trigger_debug,
        },
    }
    if not should_compress:
        return {**state, "memory_debug": memory_debug}

    result = compressor.compress(
        turns=state.get("chat_history", []),
        existing_summary=state.get("compression_summary", ""),
    )
    compressed_history = _retain_recent_turns(state.get("chat_history", []), keep_last=4)
    memory_debug["compression"] = result.to_json_dict()
    return {
        **state,
        "chat_history": compressed_history,
        "compression_summary": result.summary_text,
        "compression_stats": result.stats.to_json_dict(),
        "memory_debug": memory_debug,
    }


def memory_update_node(state: MainGraphState) -> MainGraphState:
    """Append the current question and final answer after generation."""

    chat_history = [*state.get("chat_history", [])]
    original_query = state.get("original_query", "").strip()
    final_answer = state.get("final_answer", "").strip()
    if original_query:
        chat_history.append({"role": "user", "content": original_query})
    if final_answer:
        chat_history.append({"role": "assistant", "content": final_answer})
    memory_debug = {
        **state.get("memory_debug", {}),
        "memory_update": {
            "added_user_turn": bool(original_query),
            "added_assistant_turn": bool(final_answer),
            "turn_count": len(chat_history),
        },
    }
    return {**state, "chat_history": chat_history, "memory_debug": memory_debug}


def decompose_query_node(state: MainGraphState) -> MainGraphState:
    """Ensure `decomposed_tasks` exists before Send fan-out."""

    if state.get("decomposed_tasks"):
        return state
    return task_decomposition_node(state)


def dispatch_retrieval_subgraphs(state: MainGraphState) -> list[Send]:
    """Create one LangGraph Send per decomposed task."""

    sends: list[Send] = []
    for index, task in enumerate(state.get("decomposed_tasks", []), start=1):
        sub_task_id = str(task.get("sub_task_id") or f"task_{index}")
        sub_query = str(task.get("sub_query") or "").strip()
        if not sub_query:
            continue
        sub_state = default_retrieval_subgraph_state(sub_task_id, sub_query)
        sub_state["task_metadata"] = {
            "purpose": task.get("purpose"),
            "priority": task.get("priority", index),
            "dependencies": task.get("dependencies", []),
        }
        sends.append(Send("run_sub_retrieval", sub_state))
    return sends


def run_sub_retrieval_node(
    state: RetrievalSubGraphState,
    dependencies: MainGraphDependencies,
) -> MainGraphState:
    """Run one retrieval subgraph and convert its result to a main-state update."""

    result: RetrievalSubGraphState | None = None
    retry_errors: list[str] = []
    for attempt in range(dependencies.subgraph_retry_limit + 1):
        try:
            result = dependencies.retrieval_subgraph.invoke(state)
            break
        except Exception as exc:
            retry_errors.append(f"sub_retrieval_graph_error_attempt_{attempt + 1}: {exc}")
            if attempt >= dependencies.subgraph_retry_limit:
                sub_task_id = state.get("sub_task_id", "")
                sub_query = state.get("sub_query", "")
                return {
                    "sub_answers": [
                        {
                            "sub_task_id": sub_task_id,
                            "sub_query": sub_query,
                            "sub_answer": "当前子任务检索失败，已跳过该子问题。我不会在没有检索证据时编造答案。",
                            "confidence": 0.0,
                            "insufficient_context": True,
                            "fallback_type": "retrieval_empty_fallback",
                        }
                    ],
                    "retrieved_contexts": [],
                    "retrieval_debug": {
                        "subgraphs": [
                            {
                                "sub_task_id": sub_task_id,
                                "sub_query": sub_query,
                                "retry_attempts": attempt,
                                "error": str(exc),
                            }
                        ]
                    },
                    "error_messages": [*retry_errors, f"sub_retrieval_graph_error[{sub_task_id}]: {exc}"],
                }

    assert result is not None

    sub_answer = {
        "sub_task_id": result.get("sub_task_id", state.get("sub_task_id", "")),
        "sub_query": result.get("sub_query", state.get("sub_query", "")),
        "sub_answer": result.get("sub_answer", ""),
        "confidence": result.get("confidence", 0.0),
        "insufficient_context": result.get("insufficient_context", False),
        "fallback_type": result.get("fallback_type"),
    }
    return {
        "sub_answers": [sub_answer],
        "retrieved_contexts": _contexts_from_subgraph_result(result),
        "retrieval_debug": {
            "subgraphs": [
                {
                    "sub_task_id": sub_answer["sub_task_id"],
                    "sub_query": sub_answer["sub_query"],
                    "confidence": sub_answer["confidence"],
                    "insufficient_context": sub_answer["insufficient_context"],
                    "fallback_type": sub_answer.get("fallback_type"),
                    "retry_errors": retry_errors,
                    "debug": result.get("debug", {}),
                }
            ]
        },
        "error_messages": [*retry_errors, *result.get("error_messages", [])],
    }


def answer_aggregation_node(
    state: MainGraphState,
    llm_client: Any | None = None,
) -> MainGraphState:
    """Reduce subanswers into a structured final answer with citation audit."""

    sub_answers = state.get("sub_answers", [])
    aggregation_debug = _build_aggregation_debug(sub_answers, state.get("retrieved_contexts", []))
    if not sub_answers:
        return {
            **state,
            "final_answer": "当前没有可用的子任务答案，无法生成最终回答。",
            "aggregation_debug": aggregation_debug,
        }

    if llm_client is not None:
        try:
            prompt = _build_aggregation_prompt(state, aggregation_debug)
            final_answer = str(llm_client.generate(prompt)).strip()
            if final_answer:
                unknown_sources = _unknown_sources_in_answer(final_answer, aggregation_debug["citation_sources"])
                if not unknown_sources:
                    return {**state, "final_answer": final_answer, "aggregation_debug": aggregation_debug}
                raise ValueError(f"LLM output used unknown citations: {unknown_sources}")
        except Exception as exc:
            state = {
                **state,
                "error_messages": [*state.get("error_messages", []), f"answer_aggregation_llm_error: {exc}"],
            }

    return {
        **state,
        "final_answer": _build_deterministic_final_answer(state, aggregation_debug),
        "aggregation_debug": aggregation_debug,
    }


def _reduce_main_updates(
    base_state: MainGraphState,
    updates: list[MainGraphState],
) -> MainGraphState:
    """Apply main-state reducers to subgraph updates in fallback mode."""

    reduced = dict(base_state)
    for update in updates:
        reduced["sub_answers"] = append_sub_answers(reduced.get("sub_answers"), update.get("sub_answers"))
        reduced["retrieved_contexts"] = append_retrieved_contexts(
            reduced.get("retrieved_contexts"),
            update.get("retrieved_contexts"),
        )
        reduced["retrieval_debug"] = merge_retrieval_debug(
            reduced.get("retrieval_debug"),
            update.get("retrieval_debug"),
        )
        reduced["error_messages"] = append_error_messages(
            reduced.get("error_messages"),
            update.get("error_messages"),
        )
    return reduced


def _retain_recent_turns(turns: list[dict[str, str]], keep_last: int) -> list[dict[str, str]]:
    if keep_last <= 0:
        return []
    return [dict(turn) for turn in turns[-keep_last:]]


def _contexts_from_subgraph_result(result: RetrievalSubGraphState) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for context in result.get("reranked_contexts", []):
        item = dict(context)
        item["sub_task_id"] = result.get("sub_task_id")
        item["sub_query"] = result.get("sub_query")
        contexts.append(item)
    return contexts


def _build_aggregation_prompt(state: MainGraphState, aggregation_debug: dict[str, Any]) -> str:
    sub_answer_block = "\n\n".join(
        f"[{index}] 子问题：{item.get('sub_query')}\n"
        f"置信度：{item.get('confidence')}\n"
        f"证据不足：{item.get('insufficient_context', False)}\n"
        f"答案：{item.get('sub_answer')}"
        for index, item in enumerate(state.get("sub_answers", []), start=1)
    )
    context_block = "\n".join(_context_evidence_lines(state.get("retrieved_contexts", [])))
    allowed_sources = "\n".join(f"- {source}" for source in aggregation_debug["citation_sources"])
    conflict_block = "\n".join(
        f"- {item['description']} sources={item['sources']}" for item in aggregation_debug["conflicts"]
    )
    insufficient_block = "\n".join(
        f"- {item['sub_task_id']}: {item['sub_query']}" for item in aggregation_debug["insufficient_context"]
    )
    return f"""你是企业 Agentic RAG 主图的答案聚合节点。

请基于多个子任务答案生成一个结构完整、引用明确、无重复的最终回答。

硬性约束：
1. 不得使用子任务答案和可用上下文以外的知识。
2. 不得编造引用，只能使用“允许引用来源”列表中的 source。
3. 每个重要结论后面必须带 source。
4. 如果子答案冲突，必须指出冲突来源。
5. 如果某个子问题上下文不足，必须明确说明不足。
6. 不要简单拼接子答案，要合并重复结论。

最终答案结构必须包含：
- 直接回答
- 分点解释
- 关键依据
- 引用来源
- 上下文不足（如无不足，写“无”）

原始问题：
{state.get("original_query") or state.get("rewritten_query")}

子任务答案：
{sub_answer_block}

可用上下文：
{context_block}

允许引用来源：
{allowed_sources}

已识别冲突：
{conflict_block or "无"}

上下文不足的子问题：
{insufficient_block or "无"}
"""


def _build_deterministic_final_answer(
    state: MainGraphState,
    aggregation_debug: dict[str, Any],
) -> str:
    sub_answers = state.get("sub_answers", [])
    used_ids = {item["sub_task_id"] for item in aggregation_debug["used_sub_answers"]}
    duplicate_ids = {item["discarded_sub_task_id"] for item in aggregation_debug["discarded_duplicates"]}
    sufficient_answers = [
        item
        for item in sub_answers
        if item.get("sub_task_id") in used_ids
        and item.get("sub_task_id") not in duplicate_ids
        and not item.get("insufficient_context")
    ]
    insufficient = aggregation_debug["insufficient_context"]
    conflicts = aggregation_debug["conflicts"]

    lines = ["直接回答"]
    if sufficient_answers:
        source_hint = _sources_for_sub_answer(sufficient_answers[0], state.get("retrieved_contexts", []))
        lines.append(
            f"基于已检索到的子任务证据，可以回答原问题的主要部分；具体结论见下方分点解释。"
            f" [{', '.join(source_hint) or 'unknown'}]"
        )
    else:
        lines.append("当前上下文不足，无法形成完整可靠的最终答案。")

    lines.append("\n分点解释")
    for index, item in enumerate(sufficient_answers, start=1):
        sources = _sources_for_sub_answer(item, state.get("retrieved_contexts", []))
        answer_text = _strip_citation_section(str(item.get("sub_answer", ""))).strip()
        answer_text = answer_text or "该子问题没有可用答案。"
        lines.append(
            f"{index}. {item.get('sub_query', '')}：{answer_text} "
            f"[{', '.join(sources) or 'unknown'}]"
        )

    if conflicts:
        lines.append("\n冲突说明")
        for conflict in conflicts:
            lines.append(f"- {conflict['description']} 来源：{', '.join(conflict['sources']) or 'unknown'}")

    lines.append("\n关键依据")
    evidence_lines = _context_evidence_lines(state.get("retrieved_contexts", []))
    lines.extend(evidence_lines or ["- 无可用上下文依据。"])

    lines.append("\n引用来源")
    citation_lines = [f"- {source}" for source in aggregation_debug["citation_sources"]]
    lines.extend(citation_lines or ["- 无"])

    lines.append("\n上下文不足")
    if insufficient:
        for item in insufficient:
            lines.append(f"- {item['sub_task_id']}: {item['sub_query']}，原因：{item['reason']}")
    else:
        lines.append("无")

    return "\n".join(lines)


def _build_aggregation_debug(
    sub_answers: list[dict[str, Any]],
    retrieved_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    used = []
    insufficient = []
    for item in sub_answers:
        sources = _sources_for_sub_answer(item, retrieved_contexts)
        used.append(
            {
                "sub_task_id": item.get("sub_task_id", ""),
                "sub_query": item.get("sub_query", ""),
                "confidence": float(item.get("confidence", 0.0)),
                "sources": sources,
            }
        )
        if item.get("insufficient_context") or float(item.get("confidence", 0.0)) < 0.35:
            insufficient.append(
                {
                    "sub_task_id": item.get("sub_task_id", ""),
                    "sub_query": item.get("sub_query", ""),
                    "reason": "insufficient_context flag or low confidence",
                    "confidence": float(item.get("confidence", 0.0)),
                    "sources": sources,
                }
            )

    return {
        "used_sub_answers": used,
        "discarded_duplicates": _detect_duplicate_sub_answers(sub_answers),
        "conflicts": _detect_conflicts(sub_answers, retrieved_contexts),
        "insufficient_context": insufficient,
        "citation_sources": _unique_sources(sub_answers, retrieved_contexts),
    }


def _detect_duplicate_sub_answers(sub_answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for item in sub_answers:
        if item.get("insufficient_context"):
            continue
        normalized = _normalize_answer_for_duplicate(str(item.get("sub_answer", "")))
        if not normalized:
            continue
        if normalized in seen:
            kept = seen[normalized]
            duplicates.append(
                {
                    "kept_sub_task_id": kept.get("sub_task_id", ""),
                    "discarded_sub_task_id": item.get("sub_task_id", ""),
                    "reason": "same normalized answer content",
                }
            )
        else:
            seen[normalized] = item
    return duplicates


def _detect_conflicts(
    sub_answers: list[dict[str, Any]],
    retrieved_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(sub_answers):
        if left.get("insufficient_context"):
            continue
        for right in sub_answers[left_index + 1 :]:
            if right.get("insufficient_context"):
                continue
            if _answers_conflict(str(left.get("sub_answer", "")), str(right.get("sub_answer", ""))):
                sources = sorted(
                    set(
                        [
                            *_sources_for_sub_answer(left, retrieved_contexts),
                            *_sources_for_sub_answer(right, retrieved_contexts),
                        ]
                    )
                )
                conflicts.append(
                    {
                        "sub_task_ids": [left.get("sub_task_id", ""), right.get("sub_task_id", "")],
                        "description": (
                            f"{left.get('sub_task_id')} 与 {right.get('sub_task_id')} "
                            "对同一配置/结论存在肯定与否定表述。"
                        ),
                        "sources": sources,
                    }
                )
    return conflicts


def _answers_conflict(left: str, right: str) -> bool:
    left_text = _strip_citation_section(left)
    right_text = _strip_citation_section(right)
    positive_markers = ["需要", "必须", "应该", "开启", "启用", "支持", "可以"]
    negative_markers = ["不需要", "无需", "不能", "不应该", "关闭", "禁用", "不支持", "不可以"]
    left_positive = any(marker in left_text for marker in positive_markers)
    right_positive = any(marker in right_text for marker in positive_markers)
    left_negative = any(marker in left_text for marker in negative_markers)
    right_negative = any(marker in right_text for marker in negative_markers)
    if not ((left_positive and right_negative) or (left_negative and right_positive)):
        return False
    return bool(_content_terms(left_text) & _content_terms(right_text))


def _content_terms(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text))
    stopwords = {"引用来源", "当前上下文", "子问题", "答案", "需要", "不需要", "必须", "应该", "可以"}
    return tokens - stopwords


def _normalize_answer_for_duplicate(answer: str) -> str:
    text = _strip_citation_section(answer)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,。.;；:：\[\]（）()]", "", text)
    return text[:240]


def _strip_citation_section(answer: str) -> str:
    return re.split(r"引用来源[:：]", answer, maxsplit=1)[0].strip()


def _sources_for_sub_answer(
    sub_answer: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
) -> list[str]:
    sub_task_id = sub_answer.get("sub_task_id")
    sources = []
    for context in retrieved_contexts:
        if sub_task_id and context.get("sub_task_id") != sub_task_id:
            continue
        source = _source_from_context(context)
        if source not in sources:
            sources.append(source)
    for source in _sources_from_answer_text(str(sub_answer.get("sub_answer", ""))):
        if source not in sources:
            sources.append(source)
    return sources


def _unique_sources(
    sub_answers: list[dict[str, Any]],
    retrieved_contexts: list[dict[str, Any]],
) -> list[str]:
    sources: list[str] = []
    for item in sub_answers:
        for source in _sources_for_sub_answer(item, retrieved_contexts):
            if source not in sources:
                sources.append(source)
    for context in retrieved_contexts:
        source = _source_from_context(context)
        if source not in sources:
            sources.append(source)
    return sources


def _sources_from_answer_text(answer: str) -> list[str]:
    if "引用来源" not in answer:
        return []
    citation_part = re.split(r"引用来源[:：]", answer, maxsplit=1)[1]
    candidates = re.split(r"[,，\n、 ]+", citation_part)
    return [candidate.strip().strip("-") for candidate in candidates if candidate.strip().strip("-")]


def _unknown_sources_in_answer(answer: str, allowed_sources: list[str]) -> list[str]:
    allowed = set(allowed_sources)
    candidates = set(re.findall(r"[\w./-]+\.(?:md|txt|pdf|docx)|parent_[\w-]+", answer))
    return sorted(candidate for candidate in candidates if candidate not in allowed)


def _context_evidence_lines(contexts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    seen_sources: set[str] = set()
    for context in contexts:
        source = _source_from_context(context)
        if source in seen_sources:
            continue
        seen_sources.add(source)
        text = " ".join(str(context.get("text", "")).split())[:180]
        title_path = (context.get("parent") or {}).get("title_path") or context.get("metadata", {}).get("title_path", "")
        title = f" title_path={title_path}" if title_path else ""
        lines.append(f"- {source}{title}: {text}")
    return lines


def _citation_lines(contexts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        source = _source_from_context(context)
        if source in seen:
            continue
        seen.add(source)
        lines.append(f"- {source}")
    return lines


def _source_from_context(context: dict[str, Any]) -> str:
    return (
        context.get("metadata", {}).get("source_path")
        or (context.get("parent") or {}).get("source_path")
        or context.get("parent_id")
        or "unknown"
    )
