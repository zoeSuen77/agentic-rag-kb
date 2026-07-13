"""Graph state definitions for main and sub retrieval workflows."""

from typing import Annotated, TypedDict
import operator


class MainGraphState(TypedDict, total=False):
    """State shared across the main Agentic RAG graph."""

    session_id: str
    user_query: str
    normalized_query: str
    ambiguity_result: dict
    clarification_question: str
    human_clarification: str
    decomposed_questions: list[dict]
    sub_results: Annotated[list[dict], operator.add]
    final_answer: str
    citations: list[dict]
    loop_count: int
    max_loops: int


class RetrievalSubGraphState(TypedDict, total=False):
    """State for one independent retrieval subagent lifecycle."""

    sub_query_id: str
    parent_query: str
    sub_query: str
    rewritten_query: str
    retrieved_contexts: list[dict]
    reranked_contexts: list[dict]
    local_answer: str
    citations: list[dict]

