from __future__ import annotations

from app.indexing.schemas import ParentContext
from app.utils.text import tokenize


class LexicalCrossEncoderReranker:
    """Local reranker with the same interface expected from a cross-encoder."""

    def rerank(self, query: str, contexts: list[ParentContext], top_n: int) -> list[ParentContext]:
        query_terms = set(tokenize(query))
        reranked: list[ParentContext] = []
        for context in contexts:
            context_terms = set(tokenize(context.text))
            overlap = len(query_terms & context_terms)
            coverage = overlap / max(len(query_terms), 1)
            context.score = context.score + coverage
            reranked.append(context)
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_n]

