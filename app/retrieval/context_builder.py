from __future__ import annotations

from app.indexing.schemas import ParentContext


def build_context_block(contexts: list[ParentContext], budget_chars: int = 8000) -> str:
    parts: list[str] = []
    used = 0
    for index, context in enumerate(contexts, start=1):
        source = context.metadata.get("source") or context.metadata.get("filename") or context.doc_id
        block = f"[{index}] source={source} parent_id={context.parent_id}\n{context.text.strip()}"
        if used + len(block) > budget_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)

