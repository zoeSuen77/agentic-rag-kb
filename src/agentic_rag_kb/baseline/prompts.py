"""Prompt templates for the non-Agentic RAG baseline."""

from __future__ import annotations

from agentic_rag_kb.retrieval.models import RetrievedParentContext


SYSTEM_INSTRUCTIONS = """你是企业内部技术知识库问答助手。

必须遵守：
1. 只能基于给定上下文回答。
2. 如果上下文不足以回答，必须明确说“不知道”或“当前上下文不足”。
3. 不要编造不存在的配置、步骤、命令或来源。
4. 回答末尾必须输出“引用来源”，列出使用到的 source。
"""


def build_baseline_prompt(query: str, contexts: list[RetrievedParentContext]) -> str:
    """Build the baseline RAG prompt from query and parent contexts."""

    context_block = _format_contexts(contexts)
    return f"""{SYSTEM_INSTRUCTIONS}

用户问题：
{query}

给定上下文：
{context_block if context_block else "无可用上下文。"}

请基于上下文回答。若上下文不足，请直接说明不知道。
"""


def _format_contexts(contexts: list[RetrievedParentContext]) -> str:
    lines: list[str] = []
    for index, context in enumerate(contexts, start=1):
        source = context.metadata.get("source_path") or (context.parent or {}).get("source_path") or "unknown"
        title_path = (context.parent or {}).get("title_path") or context.metadata.get("title_path") or ""
        lines.append(
            f"[{index}] source={source} parent_id={context.parent_id} title_path={title_path}\n"
            f"{context.text}"
        )
    return "\n\n".join(lines)

