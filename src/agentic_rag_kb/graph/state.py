"""Typed state models and reducers for Agentic RAG LangGraph workflows.

The main graph owns planning, ambiguity handling, parallel subtask dispatch,
aggregation, and final answer generation. Retrieval subgraphs own one independent
subquestion lifecycle. Reducers are defined here so fan-out/fan-in behavior is
explicit before graph wiring is implemented.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def append_sub_answers(
    current: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Reducer for aggregating subgraph answers from parallel retrieval tasks."""

    return [*(current or []), *(update or [])]


def append_retrieved_contexts(
    current: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Reducer for aggregating retrieved parent contexts from subgraphs.

    Contexts are deduplicated by `parent_id` when present so one parent context
    does not appear repeatedly after parallel fan-in.
    """

    merged: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()
    for item in [*(current or []), *(update or [])]:
        parent_id = item.get("parent_id")
        if parent_id:
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
        merged.append(item)
    return merged


def append_error_messages(
    current: list[str] | None,
    update: list[str] | None,
) -> list[str]:
    """Reducer for accumulating non-fatal graph and subgraph errors."""

    return [*(current or []), *(update or [])]


def merge_retrieval_debug(
    current: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reducer for merging retrieval debug traces from multiple nodes."""

    merged: dict[str, Any] = dict(current or {})
    for key, value in (update or {}).items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = [*merged[key], *value]
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


class MainGraphState(TypedDict, total=False):
    """Global state shared across the main Agentic RAG graph."""

    # Original user query before rewriting, clarification, or decomposition.
    original_query: str

    # Query rewritten with conversation memory and human clarification.
    rewritten_query: str

    # Recent chat turns used by ambiguity detection and memory compression.
    chat_history: list[dict[str, str]]

    # Structured ambiguity detection result, e.g. missing slots and confidence.
    ambiguity_result: dict[str, Any]

    # Question shown to the user when human clarification is required.
    clarification_question: str

    # User response to the clarification question.
    user_clarification: str

    # Decomposed retrieval tasks dispatched to retrieval subgraphs.
    decomposed_tasks: list[dict[str, Any]]

    # Debug payload explaining task decomposition strategy, complexity, and task count.
    decomposition_debug: dict[str, Any]

    # Subanswers returned from parallel subgraphs; aggregated by append reducer.
    sub_answers: Annotated[list[dict[str, Any]], append_sub_answers]

    # Parent contexts collected from subgraphs; aggregated and deduped by parent_id.
    retrieved_contexts: Annotated[list[dict[str, Any]], append_retrieved_contexts]

    # Final synthesized answer returned to the user.
    final_answer: str

    # Number of graph retry/fallback loops already used.
    loop_count: int

    # Non-fatal graph errors accumulated across nodes and subgraphs.
    error_messages: Annotated[list[str], append_error_messages]

    # Long conversation summary used when chat_history becomes too large.
    compression_summary: str

    # Debug payload containing dense hits, sparse hits, fusion ranking, and parent recall.
    retrieval_debug: Annotated[dict[str, Any], merge_retrieval_debug]


class RetrievalSubGraphState(TypedDict, total=False):
    """State for one independent retrieval subgraph task."""

    # Stable ID for the decomposed retrieval task.
    sub_task_id: str

    # Subquestion assigned to this retrieval subgraph.
    sub_query: str

    # Subquestion rewritten for retrieval.
    rewritten_sub_query: str

    # Child-level hybrid retrieval candidates before final parent-context packaging.
    retrieved_chunks: list[dict[str, Any]]

    # Parent-expanded contexts after cross-encoder reranking.
    reranked_contexts: list[dict[str, Any]]

    # Local answer generated for this subquestion.
    sub_answer: str

    # Confidence score for this subgraph's local answer.
    confidence: float

    # Non-fatal errors produced by this subgraph.
    error_messages: Annotated[list[str], append_error_messages]


# Backward-compatible alias used by earlier skeleton modules.
RetrievalSubGraphStateAlias = RetrievalSubGraphState
