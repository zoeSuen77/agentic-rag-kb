from __future__ import annotations

from app.indexing.schemas import ParentContext


def build_citations(contexts: list[ParentContext]) -> list[dict]:
    citations: list[dict] = []
    for index, context in enumerate(contexts, start=1):
        citations.append(
            {
                "citation_id": index,
                "doc_id": context.doc_id,
                "parent_id": context.parent_id,
                "source": context.metadata.get("source") or context.metadata.get("filename"),
                "score": round(context.score, 4),
            }
        )
    return citations
