"""LangGraph retrieval subgraph for one decomposed task.

Each subgraph owns a complete retrieval lifecycle for a single subquestion:
rewrite, hybrid retrieve, rerank, grounded sub-answer generation, and confidence
checking. The main graph can later fan out many copies of this subgraph with
LangGraph Send API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag_kb.agents.fallback import build_fallback_response
from agentic_rag_kb.graph.state import RetrievalSubGraphState
from agentic_rag_kb.llm import LLMClient
from agentic_rag_kb.rerank import RerankConfig
from agentic_rag_kb.retrieval.models import RetrievedChunk, RetrievedParentContext, RetrievalDebugInfo


class RetrieverProtocol(Protocol):
    """Protocol implemented by HybridRetriever and test doubles."""

    def retrieve(
        self,
        query: str,
        top_k_dense: int,
        top_k_sparse: int,
        final_k: int,
    ) -> list[RetrievedParentContext]:
        """Return parent-expanded retrieval contexts."""

    def get_debug_info(self) -> RetrievalDebugInfo:
        """Return debug information for the most recent retrieval call."""


class RerankerProtocol(Protocol):
    """Protocol implemented by CrossEncoderReranker and test doubles."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedParentContext],
        top_n: int | None = None,
        final_context_k: int | None = None,
    ) -> list[RetrievedParentContext]:
        """Return reranked parent contexts."""


@dataclass(slots=True)
class RetrievalSubgraphConfig:
    """Runtime knobs for one retrieval subgraph."""

    top_k_dense: int = 30
    top_k_sparse: int = 30
    retrieval_final_k: int = 20
    rerank_config: RerankConfig | None = None
    confidence_threshold: float = 0.35
    subgraph_retry_limit: int = 1

    @property
    def effective_rerank_config(self) -> RerankConfig:
        """Return an explicit rerank config."""

        return self.rerank_config or RerankConfig()


@dataclass(slots=True)
class RetrievalSubgraphDependencies:
    """Dependencies injected into subgraph nodes."""

    retriever: RetrieverProtocol
    reranker: RerankerProtocol | None
    llm_client: LLMClient | None = None
    config: RetrievalSubgraphConfig | None = None

    @property
    def effective_config(self) -> RetrievalSubgraphConfig:
        """Return an explicit subgraph config."""

        return self.config or RetrievalSubgraphConfig()


SUB_ANSWER_PROMPT = """你是企业知识库 Agentic RAG 的子任务回答节点。

必须遵守：
1. 只能基于给定上下文回答子问题。
2. 如果上下文不足，直接说明“当前上下文不足”。
3. 不要编造配置、命令、结论或来源。
4. 必须在答案末尾写“引用来源”，列出 source_path 或 parent_id。

子问题：
{sub_query}

上下文：
{contexts}

请输出简洁、可合并到最终答案的子答案。
"""


def build_retrieval_subgraph(dependencies: RetrievalSubgraphDependencies):
    """Build and compile the LangGraph retrieval subgraph.

    The compiled graph can be invoked with `RetrievalSubGraphState`, making this
    lifecycle independently testable before it is called by the main graph.
    """

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        return SequentialRetrievalSubgraph(dependencies, import_error=exc)

    workflow = StateGraph(RetrievalSubGraphState)
    workflow.add_node("rewrite_sub_query", lambda state: rewrite_sub_query_node(state))
    workflow.add_node("hybrid_retrieve", lambda state: hybrid_retrieve_node(state, dependencies))
    workflow.add_node("rerank_contexts", lambda state: rerank_node(state, dependencies))
    workflow.add_node("generate_sub_answer", lambda state: generate_sub_answer_node(state, dependencies))
    workflow.add_node("confidence_check", lambda state: confidence_check_node(state, dependencies.effective_config))

    workflow.add_edge(START, "rewrite_sub_query")
    workflow.add_edge("rewrite_sub_query", "hybrid_retrieve")
    workflow.add_edge("hybrid_retrieve", "rerank_contexts")
    workflow.add_edge("rerank_contexts", "generate_sub_answer")
    workflow.add_edge("generate_sub_answer", "confidence_check")
    workflow.add_edge("confidence_check", END)
    return workflow.compile()


class SequentialRetrievalSubgraph:
    """Fallback runner with the same `invoke` shape as a compiled LangGraph.

    It is used only when `langgraph` is unavailable in a local test environment.
    Production code should install dependencies and will use `StateGraph`.
    """

    def __init__(
        self,
        dependencies: RetrievalSubgraphDependencies,
        import_error: ImportError | None = None,
    ) -> None:
        self.dependencies = dependencies
        self.import_error = import_error

    def invoke(self, state: RetrievalSubGraphState) -> RetrievalSubGraphState:
        """Run the same node sequence as the LangGraph subgraph."""

        next_state = rewrite_sub_query_node(state)
        next_state = hybrid_retrieve_node(next_state, self.dependencies)
        next_state = rerank_node(next_state, self.dependencies)
        next_state = generate_sub_answer_node(next_state, self.dependencies)
        next_state = confidence_check_node(next_state, self.dependencies.effective_config)
        if self.import_error is not None:
            debug = _merge_debug(
                next_state,
                "build_retrieval_subgraph",
                {"runner": "sequential_fallback", "reason": str(self.import_error)},
            )
            next_state = {**next_state, "debug": debug}
        return next_state


def rewrite_sub_query_node(state: RetrievalSubGraphState) -> RetrievalSubGraphState:
    """Rewrite a subquestion into a retrieval-friendly query."""

    sub_query = state.get("sub_query", "").strip()
    rewritten = _normalize_retrieval_query(sub_query)
    debug = _merge_debug(state, "rewrite_sub_query", {"rewritten_sub_query": rewritten})
    return {**state, "rewritten_sub_query": rewritten, "debug": debug}


def hybrid_retrieve_node(
    state: RetrievalSubGraphState,
    dependencies: RetrievalSubgraphDependencies,
) -> RetrievalSubGraphState:
    """Run Dense + Sparse hybrid retrieval for the rewritten subquery."""

    config = dependencies.effective_config
    query = state.get("rewritten_sub_query") or state.get("sub_query", "")
    errors = [*state.get("error_messages", [])]
    for attempt in range(config.subgraph_retry_limit + 1):
        try:
            contexts = dependencies.retriever.retrieve(
                query=query,
                top_k_dense=config.top_k_dense,
                top_k_sparse=config.top_k_sparse,
                final_k=config.retrieval_final_k,
            )
            debug_info = dependencies.retriever.get_debug_info()
            debug_payload = debug_info.to_json_dict() if hasattr(debug_info, "to_json_dict") else dict(debug_info)
            debug = _merge_debug(
                state,
                "hybrid_retrieve",
                {
                    "query": query,
                    "retrieved_count": len(contexts),
                    "retry_attempts": attempt,
                    "retrieval_debug": debug_payload,
                },
            )
            return {**state, "retrieved_chunks": _contexts_to_dicts(contexts), "error_messages": errors, "debug": debug}
        except Exception as exc:
            errors.append(f"hybrid_retrieve_error_attempt_{attempt + 1}: {exc}")
            if attempt >= config.subgraph_retry_limit:
                debug = _merge_debug(
                    state,
                    "hybrid_retrieve",
                    {"error": str(exc), "retrieved_count": 0, "retry_attempts": attempt},
                )
                return {**state, "retrieved_chunks": [], "error_messages": errors, "debug": debug}
    return {**state, "retrieved_chunks": [], "error_messages": errors}


def rerank_node(
    state: RetrievalSubGraphState,
    dependencies: RetrievalSubgraphDependencies,
) -> RetrievalSubGraphState:
    """Rerank retrieved parent contexts or pass through by hybrid score."""

    config = dependencies.effective_config
    rerank_config = config.effective_rerank_config
    query = state.get("rewritten_sub_query") or state.get("sub_query", "")
    candidates = _context_objects_from_state(state)

    try:
        if not candidates:
            contexts: list[RetrievedParentContext] = []
        elif not rerank_config.enable_rerank or dependencies.reranker is None:
            contexts = sorted(candidates, key=lambda item: item.score_fused, reverse=True)[: rerank_config.final_context_k]
        else:
            contexts = dependencies.reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=rerank_config.rerank_top_n,
                final_context_k=rerank_config.final_context_k,
            )
        debug = _merge_debug(
            state,
            "rerank",
            {
                "enable_rerank": rerank_config.enable_rerank,
                "candidate_count": len(candidates),
                "reranked_count": len(contexts),
            },
        )
        return {**state, "reranked_contexts": _contexts_to_dicts(contexts), "debug": debug}
    except Exception as exc:
        errors = [*state.get("error_messages", []), f"rerank_error: {exc}"]
        fallback = sorted(candidates, key=lambda item: item.score_fused, reverse=True)[: rerank_config.final_context_k]
        debug = _merge_debug(state, "rerank", {"error": str(exc), "reranked_count": len(fallback)})
        return {
            **state,
            "reranked_contexts": _contexts_to_dicts(fallback),
            "error_messages": errors,
            "debug": debug,
        }


def generate_sub_answer_node(
    state: RetrievalSubGraphState,
    dependencies: RetrievalSubgraphDependencies,
) -> RetrievalSubGraphState:
    """Generate a grounded sub-answer with citations."""

    contexts = _reranked_context_objects_from_state(state)
    sub_query = state.get("sub_query", "")
    if not contexts:
        fallback = build_fallback_response("retrieval_empty_fallback", query=sub_query)
        answer = f"{fallback['answer']}\n\n引用来源：无"
        debug = _merge_debug(
            state,
            "generate_sub_answer",
            {"used_context_count": 0, "generation": "retrieval_empty_fallback", "fallback_type": fallback["fallback_type"]},
        )
        return {
            **state,
            "sub_answer": answer,
            "fallback_type": fallback["fallback_type"],
            "error_messages": [*state.get("error_messages", []), fallback["error_message"]],
            "debug": debug,
        }

    if dependencies.llm_client is None:
        answer = _fallback_sub_answer(sub_query, contexts)
        debug = _merge_debug(state, "generate_sub_answer", {"used_context_count": len(contexts), "generation": "fallback"})
        return {**state, "sub_answer": answer, "debug": debug}

    try:
        prompt = SUB_ANSWER_PROMPT.format(sub_query=sub_query, contexts=_format_contexts_for_prompt(contexts))
        answer = dependencies.llm_client.generate(prompt).strip()
        if "引用来源" not in answer:
            answer = f"{answer}\n\n引用来源：{', '.join(_citation_labels(contexts))}"
        debug = _merge_debug(state, "generate_sub_answer", {"used_context_count": len(contexts), "generation": "llm"})
        return {**state, "sub_answer": answer, "debug": debug}
    except Exception as exc:
        errors = [*state.get("error_messages", []), f"generate_sub_answer_error: {exc}"]
        answer = _fallback_sub_answer(sub_query, contexts)
        debug = _merge_debug(state, "generate_sub_answer", {"error": str(exc), "generation": "fallback_after_error"})
        return {**state, "sub_answer": answer, "error_messages": errors, "debug": debug}


def confidence_check_node(
    state: RetrievalSubGraphState,
    config: RetrievalSubgraphConfig | None = None,
) -> RetrievalSubGraphState:
    """Estimate whether the subgraph has enough evidence for its answer."""

    effective_config = config or RetrievalSubgraphConfig()
    contexts = state.get("reranked_contexts", [])
    answer = state.get("sub_answer", "")
    confidence = _estimate_confidence(contexts, answer)
    insufficient_context = confidence < effective_config.confidence_threshold
    answer = state.get("sub_answer", "")
    errors = [*state.get("error_messages", [])]
    fallback_type = state.get("fallback_type")
    if insufficient_context and contexts and "low_confidence_fallback" not in errors:
        fallback = build_fallback_response(
            "low_confidence_fallback",
            query=state.get("sub_query", ""),
            details=f"confidence={confidence}",
        )
        answer = f"{answer}\n\n{fallback['answer']}"
        errors.append(fallback["error_message"])
        fallback_type = fallback["fallback_type"]
    debug = _merge_debug(
        state,
        "confidence_check",
        {
            "confidence": confidence,
            "threshold": effective_config.confidence_threshold,
            "insufficient_context": insufficient_context,
        },
    )
    return {
        **state,
        "sub_answer": answer,
        "confidence": confidence,
        "insufficient_context": insufficient_context,
        "fallback_type": fallback_type,
        "error_messages": errors,
        "debug": debug,
    }


def _normalize_retrieval_query(query: str) -> str:
    normalized = " ".join(query.strip().split())
    suffixes = ["请详细说明", "帮我解释", "麻烦说明一下"]
    for suffix in suffixes:
        normalized = normalized.replace(suffix, "").strip()
    return normalized


def _merge_debug(state: RetrievalSubGraphState, node_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    debug = dict(state.get("debug", {}))
    trace = [*debug.get("trace", []), {"node": node_name, **payload}]
    debug[node_name] = payload
    debug["trace"] = trace
    return debug


def _contexts_to_dicts(contexts: list[RetrievedParentContext]) -> list[dict[str, Any]]:
    return [context.to_json_dict() if hasattr(context, "to_json_dict") else dict(context) for context in contexts]


def _context_objects_from_state(state: RetrievalSubGraphState) -> list[RetrievedParentContext]:
    return [_dict_to_context(item) for item in state.get("retrieved_chunks", [])]


def _reranked_context_objects_from_state(state: RetrievalSubGraphState) -> list[RetrievedParentContext]:
    raw_contexts = state.get("reranked_contexts") or state.get("retrieved_chunks", [])
    return [_dict_to_context(item) for item in raw_contexts]


def _dict_to_context(item: dict[str, Any]) -> RetrievedParentContext:
    return RetrievedChunk(
        child_id=str(item.get("child_id", "")),
        parent_id=str(item.get("parent_id", "")),
        text=str(item.get("text", "")),
        score_dense=item.get("score_dense"),
        score_sparse=item.get("score_sparse"),
        score_fused=float(item.get("score_fused", 0.0)),
        metadata=dict(item.get("metadata") or {}),
        parent=item.get("parent"),
        rerank_score=item.get("rerank_score"),
    )


def _format_contexts_for_prompt(contexts: list[RetrievedParentContext]) -> str:
    lines: list[str] = []
    for index, context in enumerate(contexts, start=1):
        source = _context_source(context)
        title_path = (context.parent or {}).get("title_path") or context.metadata.get("title_path") or ""
        lines.append(f"[{index}] source={source} parent_id={context.parent_id} title_path={title_path}\n{context.text}")
    return "\n\n".join(lines)


def _fallback_sub_answer(sub_query: str, contexts: list[RetrievedParentContext]) -> str:
    snippets = []
    for context in contexts[:2]:
        text = " ".join(context.text.split())
        snippets.append(text[:260])
    evidence = "\n".join(f"- {snippet}" for snippet in snippets if snippet)
    citations = ", ".join(_citation_labels(contexts))
    return f"子问题：{sub_query}\n\n基于检索上下文，可参考以下证据：\n{evidence}\n\n引用来源：{citations}"


def _citation_labels(contexts: list[RetrievedParentContext]) -> list[str]:
    labels: list[str] = []
    for context in contexts:
        label = _context_source(context)
        if label not in labels:
            labels.append(label)
    return labels or ["unknown"]


def _context_source(context: RetrievedParentContext) -> str:
    return (
        context.metadata.get("source_path")
        or (context.parent or {}).get("source_path")
        or context.metadata.get("doc_id")
        or context.parent_id
        or "unknown"
    )


def _estimate_confidence(contexts: list[dict[str, Any]], answer: str) -> float:
    if not contexts or "上下文不足" in answer or "无法回答" in answer:
        return 0.0

    best_score = 0.0
    for context in contexts:
        rerank_score = context.get("rerank_score")
        fused_score = context.get("score_fused", 0.0)
        if rerank_score is not None:
            best_score = max(best_score, _normalize_rerank_score(float(rerank_score)))
        best_score = max(best_score, min(float(fused_score) * 20.0, 0.7))

    coverage_bonus = min(len(contexts) * 0.08, 0.24)
    citation_bonus = 0.1 if "引用来源" in answer else 0.0
    return round(max(0.0, min(1.0, best_score + coverage_bonus + citation_bonus)), 3)


def _normalize_rerank_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return max(0.0, min(1.0, (score + 10.0) / 20.0))
