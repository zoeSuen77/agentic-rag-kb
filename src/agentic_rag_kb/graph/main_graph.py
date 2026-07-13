"""Main LangGraph workflow with Send API fan-out.

The main graph performs the map-reduce portion of Agentic RAG:

1. decompose a rewritten query into independent retrieval tasks;
2. use LangGraph `Send` to dispatch each task to the same retrieval subgraph;
3. reduce subgraph outputs into `sub_answers`, `retrieved_contexts`, and debug;
4. aggregate subanswers into a final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class MainGraphDependencies:
    """Dependencies injected into main graph nodes."""

    retrieval_subgraph: InvokableGraph
    llm_client: Any | None = None


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
    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("run_sub_retrieval", lambda state: run_sub_retrieval_node(state, dependencies))
    workflow.add_node("aggregate_answers", lambda state: answer_aggregation_node(state, dependencies.llm_client))

    workflow.add_edge(START, "decompose_query")
    workflow.add_conditional_edges("decompose_query", dispatch_retrieval_subgraphs, ["run_sub_retrieval"])
    workflow.add_edge("run_sub_retrieval", "aggregate_answers")
    workflow.add_edge("aggregate_answers", END)
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

        current = decompose_query_node(state)
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
        return answer_aggregation_node(reduced, self.dependencies.llm_client)


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

    try:
        result = dependencies.retrieval_subgraph.invoke(state)
    except Exception as exc:
        sub_task_id = state.get("sub_task_id", "")
        sub_query = state.get("sub_query", "")
        return {
            "sub_answers": [
                {
                    "sub_task_id": sub_task_id,
                    "sub_query": sub_query,
                    "sub_answer": "当前子任务检索失败，已跳过该子问题。",
                    "confidence": 0.0,
                    "insufficient_context": True,
                }
            ],
            "retrieved_contexts": [],
            "retrieval_debug": {
                "subgraphs": [
                    {
                        "sub_task_id": sub_task_id,
                        "sub_query": sub_query,
                        "error": str(exc),
                    }
                ]
            },
            "error_messages": [f"sub_retrieval_graph_error[{sub_task_id}]: {exc}"],
        }

    sub_answer = {
        "sub_task_id": result.get("sub_task_id", state.get("sub_task_id", "")),
        "sub_query": result.get("sub_query", state.get("sub_query", "")),
        "sub_answer": result.get("sub_answer", ""),
        "confidence": result.get("confidence", 0.0),
        "insufficient_context": result.get("insufficient_context", False),
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
                    "debug": result.get("debug", {}),
                }
            ]
        },
        "error_messages": result.get("error_messages", []),
    }


def answer_aggregation_node(
    state: MainGraphState,
    llm_client: Any | None = None,
) -> MainGraphState:
    """Reduce subanswers into a final answer."""

    sub_answers = state.get("sub_answers", [])
    if not sub_answers:
        return {
            **state,
            "final_answer": "当前没有可用的子任务答案，无法生成最终回答。",
        }

    if llm_client is not None:
        try:
            prompt = _build_aggregation_prompt(state)
            final_answer = str(llm_client.generate(prompt)).strip()
            if final_answer:
                return {**state, "final_answer": final_answer}
        except Exception as exc:
            state = {
                **state,
                "error_messages": [*state.get("error_messages", []), f"answer_aggregation_llm_error: {exc}"],
            }

    lines = ["综合多个子问题的检索结果，答案如下："]
    for index, item in enumerate(sub_answers, start=1):
        sub_query = item.get("sub_query", "")
        confidence = float(item.get("confidence", 0.0))
        status = "证据不足" if item.get("insufficient_context") else "可回答"
        lines.append(f"\n{index}. {sub_query}（confidence={confidence:.2f}, {status}）")
        lines.append(str(item.get("sub_answer", "")).strip())

    citations = _citation_lines(state.get("retrieved_contexts", []))
    if citations:
        lines.append("\n汇总引用来源：")
        lines.extend(citations)
    return {**state, "final_answer": "\n".join(lines)}


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


def _contexts_from_subgraph_result(result: RetrievalSubGraphState) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for context in result.get("reranked_contexts", []):
        item = dict(context)
        item["sub_task_id"] = result.get("sub_task_id")
        item["sub_query"] = result.get("sub_query")
        contexts.append(item)
    return contexts


def _build_aggregation_prompt(state: MainGraphState) -> str:
    sub_answer_block = "\n\n".join(
        f"[{index}] 子问题：{item.get('sub_query')}\n"
        f"置信度：{item.get('confidence')}\n"
        f"答案：{item.get('sub_answer')}"
        for index, item in enumerate(state.get("sub_answers", []), start=1)
    )
    context_block = "\n".join(_citation_lines(state.get("retrieved_contexts", [])))
    return f"""你是企业 Agentic RAG 主图的答案聚合节点。

请基于多个子任务答案生成一个结构清晰的最终回答。
必须保留引用来源；如果某个子问题证据不足，要明确说明。

原始问题：
{state.get("original_query") or state.get("rewritten_query")}

子任务答案：
{sub_answer_block}

可用引用：
{context_block}
"""


def _citation_lines(contexts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        source = (
            context.get("metadata", {}).get("source_path")
            or (context.get("parent") or {}).get("source_path")
            or context.get("parent_id")
            or "unknown"
        )
        if source in seen:
            continue
        seen.add(source)
        lines.append(f"- {source}")
    return lines
