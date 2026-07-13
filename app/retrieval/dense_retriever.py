from __future__ import annotations

from app.indexing.hybrid_indexer import HybridIndexer
from app.indexing.schemas import SearchResult


class DenseRetriever:
    def __init__(self, indexer: HybridIndexer) -> None:
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        return self.indexer.dense_search(query, top_k)

