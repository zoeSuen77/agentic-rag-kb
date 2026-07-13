"""Cross-encoder reranker interface.

TODO:
- Load sentence-transformers CrossEncoder model.
- Score `(query, context)` pairs.
- Return top-N contexts with rerank scores.
"""


class CrossEncoderReranker:
    """Rerank hybrid retrieval candidates with a cross-encoder model."""

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """Return reranked contexts."""

        raise NotImplementedError("Cross-encoder reranking will be implemented in the rerank phase.")

