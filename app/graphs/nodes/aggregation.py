from __future__ import annotations

from app.graphs.states import MainGraphState


def aggregate_subanswers(state: MainGraphState) -> MainGraphState:
    sub_results = state.get("sub_results", [])
    contexts_by_parent: dict[str, dict] = {}
    citations_by_parent: dict[str, dict] = {}
    answer_parts: list[str] = []

    for result in sub_results:
        if result.get("local_answer"):
            answer_parts.append(f"- {result.get('sub_query')}: {result.get('local_answer')}")
        for context in result.get("reranked_contexts", []):
            contexts_by_parent.setdefault(context["parent_id"], context)
        for citation in result.get("local_citations", []):
            citations_by_parent.setdefault(citation["parent_id"], citation)

    state["aggregated_contexts"] = list(contexts_by_parent.values())
    state["citations"] = list(citations_by_parent.values())
    state["aggregated_answer"] = "\n".join(answer_parts)
    return state


def check_faithfulness(state: MainGraphState) -> MainGraphState:
    if not state.get("aggregated_contexts"):
        state["fallback_reason"] = "No retrieval context was found."
    elif not state.get("aggregated_answer"):
        state["fallback_reason"] = "No subanswer was generated."
    else:
        state["fallback_reason"] = ""
    return state

