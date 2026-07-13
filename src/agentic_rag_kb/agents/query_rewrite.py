"""Query rewriting node for Agentic RAG.

The rewrite step normalizes the user question before retrieval. It preserves the
user's intent, may include already-known chat context, and must not invent new
entities that were absent from the query or conversation state.
"""

from __future__ import annotations

from typing import Any

from agentic_rag_kb.agents.json_utils import ensure_string_list, parse_json_object
from agentic_rag_kb.graph.state import MainGraphState
from agentic_rag_kb.llm import LLMClient


QUERY_REWRITE_PROMPT = """你是企业知识库 RAG 的检索前 query rewrite 节点。

请只输出 JSON，不要输出 Markdown。

任务：
1. 保留用户真实意图。
2. 只能使用 original_query、chat_history、compression_summary 中已经出现的信息。
3. 可以补全省略指代，例如“这个”“它”“this”“it”，但不得引入未出现的新实体。
4. 如果仍缺信息，把缺失项写入 missing_info。

输出 JSON schema:
{
  "rewritten_query": "...",
  "reason": "...",
  "missing_info": ["..."]
}

original_query:
{original_query}

chat_history:
{chat_history}

compression_summary:
{compression_summary}
"""


class QueryRewriteAgent:
    """Agent that rewrites the original user query into a retrieval-ready query."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def run(self, state: MainGraphState) -> MainGraphState:
        """Return state update containing `rewritten_query` and rewrite metadata."""

        return query_rewrite_node(state, self.llm_client)


def query_rewrite_node(
    state: MainGraphState,
    llm_client: LLMClient | None = None,
) -> MainGraphState:
    """Rewrite `original_query` using chat history and compressed memory."""

    original_query = state.get("original_query", "").strip()
    user_clarification = state.get("user_clarification", "").strip()
    query_for_rewrite = (
        f"{original_query}\n用户澄清：{user_clarification}" if user_clarification else original_query
    )
    chat_history = state.get("chat_history", [])
    compression_summary = state.get("compression_summary", "")

    if llm_client is not None:
        try:
            parsed = parse_json_object(
                llm_client.generate(
                    QUERY_REWRITE_PROMPT.format(
                        original_query=original_query,
                        chat_history=chat_history,
                        compression_summary=compression_summary,
                    )
                )
            )
            result = _normalize_rewrite_result(parsed, query_for_rewrite)
            return {**state, "rewritten_query": result["rewritten_query"], "query_rewrite_result": result}
        except Exception as exc:
            errors = [*state.get("error_messages", []), f"query_rewrite_llm_error: {exc}"]
            state = {**state, "error_messages": errors}

    result = _fallback_rewrite(query_for_rewrite, chat_history, compression_summary)
    return {**state, "rewritten_query": result["rewritten_query"], "query_rewrite_result": result}


def _normalize_rewrite_result(parsed: dict[str, Any], original_query: str) -> dict[str, Any]:
    rewritten_query = str(parsed.get("rewritten_query") or original_query).strip()
    return {
        "rewritten_query": rewritten_query,
        "reason": str(parsed.get("reason") or "LLM structured rewrite").strip(),
        "missing_info": ensure_string_list(parsed.get("missing_info")),
    }


def _fallback_rewrite(
    original_query: str,
    chat_history: list[dict[str, str]],
    compression_summary: str,
) -> dict[str, Any]:
    context_hint = _latest_context_hint(chat_history) or compression_summary.strip()
    missing_info: list[str] = []
    rewritten = original_query
    reason = "Original query is already clear enough for retrieval."

    if _has_pronoun_reference(original_query):
        if context_hint:
            rewritten = f"{original_query}（上下文：{context_hint}）"
            reason = "Resolved omitted reference using existing chat context."
        else:
            missing_info.append("指代对象")
            reason = "Query contains an omitted reference but no prior context was available."

    return {"rewritten_query": rewritten, "reason": reason, "missing_info": missing_info}


def _latest_context_hint(chat_history: list[dict[str, str]]) -> str:
    for message in reversed(chat_history):
        content = str(message.get("content", "")).strip()
        if content:
            return content[:240]
    return ""


def _has_pronoun_reference(query: str) -> bool:
    lowered = query.lower()
    pronouns = ["这个", "那个", "它", "这", "该问题", "this", "that", "it", "them"]
    return any(token in lowered for token in pronouns)
