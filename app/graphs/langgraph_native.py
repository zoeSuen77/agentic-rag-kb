from __future__ import annotations

from typing import Any

from app.graphs.main_graph import MainGraph
from app.graphs.nodes.aggregation import aggregate_subanswers, check_faithfulness
from app.graphs.nodes.ambiguity import detect_ambiguity
from app.graphs.nodes.answer_generation import generate_final_answer
from app.graphs.nodes.compression import compress_context
from app.graphs.nodes.decomposition import classify_query, decompose_question, detect_complexity
from app.graphs.nodes.fallback import apply_fallback
from app.graphs.nodes.human_review import apply_human_clarification
from app.graphs.states import MainGraphState
from app.indexing.hybrid_indexer import HybridIndexer
from app.settings import AppSettings


def build_native_langgraph(indexer: HybridIndexer, settings: AppSettings) -> Any:
    """Build a native LangGraph graph when langgraph is installed.

    The default `MainGraph` class is intentionally dependency-light so tests and local
    development run without external services. This factory exposes the same graph
    semantics through LangGraph's `StateGraph` and `Send` API for production usage.
    """

    try:
        from langgraph.constants import END, Send
        from langgraph.graph import StateGraph
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install langgraph to build the native graph") from exc

    fallback_graph = MainGraph(indexer=indexer, settings=settings)

    def normalize(state: MainGraphState) -> MainGraphState:
        state.setdefault("normalized_query", state.get("raw_query", ""))
        state.setdefault("loop_count", 0)
        state.setdefault("max_loops", 2)
        state.setdefault("errors", [])
        return state

    def route_after_ambiguity(state: MainGraphState) -> str:
        ambiguous = state.get("ambiguity_result", {}).get("is_ambiguous")
        return "final" if ambiguous and not state.get("human_clarification") else "classify"

    def dispatch_with_send(state: MainGraphState) -> list[Send]:
        return [
            Send(
                "sub_retrieval",
                {
                    "parent_query": state.get("normalized_query", ""),
                    "sub_query": item["question"],
                    "sub_query_id": item["sub_query_id"],
                    "intent": item.get("intent", "fact_lookup"),
                },
            )
            for item in state.get("decomposed_questions", [])
        ]

    def sub_retrieval_node(state: dict) -> dict:
        return {"sub_results": [fallback_graph.sub_graph.invoke(state)]}

    def final_or_fallback(state: MainGraphState) -> MainGraphState:
        state = check_faithfulness(state)
        if state.get("fallback_reason"):
            return apply_fallback(state)
        return generate_final_answer(state)

    graph = StateGraph(MainGraphState)
    graph.add_node("normalize", normalize)
    graph.add_node("compress", compress_context)
    graph.add_node("ambiguity", detect_ambiguity)
    graph.add_node("clarify", apply_human_clarification)
    graph.add_node("classify", classify_query)
    graph.add_node("complexity", detect_complexity)
    graph.add_node("decompose", decompose_question)
    graph.add_node("sub_retrieval", sub_retrieval_node)
    graph.add_node("aggregate", aggregate_subanswers)
    graph.add_node("final", final_or_fallback)

    graph.set_entry_point("normalize")
    graph.add_edge("normalize", "compress")
    graph.add_edge("compress", "ambiguity")
    graph.add_edge("ambiguity", "clarify")
    graph.add_conditional_edges("clarify", route_after_ambiguity, {"final": "final", "classify": "classify"})
    graph.add_edge("classify", "complexity")
    graph.add_edge("complexity", "decompose")
    graph.add_conditional_edges("decompose", dispatch_with_send, ["sub_retrieval"])
    graph.add_edge("sub_retrieval", "aggregate")
    graph.add_edge("aggregate", "final")
    graph.add_edge("final", END)
    return graph.compile()

