"""Hybrid retrieval and rank fusion.

TODO:
- Run dense and sparse retrieval in parallel.
- Fuse rankings with Reciprocal Rank Fusion.
- Expand child hits to parent chunks before reranking.
"""


class HybridRetriever:
    """Combine dense and sparse retrieval for robust technical QA."""

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Return fused retrieval candidates."""

        raise NotImplementedError("Hybrid retrieval will be implemented after dense and sparse retrievers.")

