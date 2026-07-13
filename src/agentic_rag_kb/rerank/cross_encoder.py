"""Cross-encoder reranker.

The production reranker uses `sentence_transformers.CrossEncoder` to score
`(query, parent_context)` pairs after hybrid retrieval. A deterministic lexical
reranker is also provided for tests and offline development.
"""

from __future__ import annotations

from agentic_rag_kb.indexing.sparse import tokenize
from agentic_rag_kb.retrieval.models import RetrievedChunk


class CrossEncoderReranker:
    """Rerank hybrid retrieval candidates with a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install sentence-transformers to use CrossEncoderReranker.") from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Return reranked contexts."""

        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(
            candidates,
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )[:top_n]


class LexicalReranker:
    """Offline reranker with the same interface as the cross-encoder reranker."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Rerank by lexical overlap with parent context."""

        query_terms = set(tokenize(query))
        for candidate in candidates:
            context_terms = set(tokenize(candidate.text))
            candidate.rerank_score = len(query_terms & context_terms) / max(len(query_terms), 1)
        return sorted(candidates, key=lambda item: item.rerank_score or 0.0, reverse=True)[:top_n]
