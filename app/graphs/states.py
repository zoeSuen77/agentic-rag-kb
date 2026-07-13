from __future__ import annotations

import operator
from typing import Any, TypedDict
from typing import Annotated


class MainGraphState(TypedDict, total=False):
    session_id: str
    user_id: str
    raw_query: str
    normalized_query: str
    conversation_history: list[dict[str, Any]]
    compressed_history: str
    ambiguity_result: dict[str, Any]
    clarification_question: str
    human_clarification: str
    query_type: str
    complexity_level: str
    decomposed_questions: list[dict[str, Any]]
    sub_tasks: list[dict[str, Any]]
    sub_results: Annotated[list[dict[str, Any]], operator.add]
    aggregated_contexts: list[dict[str, Any]]
    aggregated_answer: str
    final_answer: str
    citations: list[dict[str, Any]]
    loop_count: int
    max_loops: int
    fallback_reason: str
    errors: list[dict[str, Any]]
    evaluation_trace: dict[str, Any]


class SubRetrievalState(TypedDict, total=False):
    sub_query_id: str
    parent_query: str
    sub_query: str
    rewritten_query: str
    intent: str
    dense_results: list[dict[str, Any]]
    sparse_results: list[dict[str, Any]]
    fused_results: list[dict[str, Any]]
    parent_contexts: list[dict[str, Any]]
    reranked_contexts: list[dict[str, Any]]
    local_answer: str
    local_citations: list[dict[str, Any]]
    retrieval_quality: dict[str, Any]
    error: str
