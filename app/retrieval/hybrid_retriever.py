from __future__ import annotations

from collections import defaultdict

from app.indexing.hybrid_indexer import HybridIndexer
from app.indexing.schemas import SearchResult
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]], rrf_k: int = 60, top_k: int = 50
) -> list[SearchResult]:
    fused_scores: dict[str, float] = defaultdict(float)
    best_result: dict[str, SearchResult] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            fused_scores[result.chunk_id] += 1.0 / (rrf_k + rank)
            current = best_result.get(result.chunk_id)
            if current is None or result.score > current.score:
                best_result[result.chunk_id] = result
    fused: list[SearchResult] = []
    for chunk_id, score in fused_scores.items():
        result = best_result[chunk_id]
        fused.append(
            SearchResult(
                chunk_id=result.chunk_id,
                parent_id=result.parent_id,
                doc_id=result.doc_id,
                text=result.text,
                score=score,
                metadata=result.metadata,
                source="hybrid",
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]


class HybridRetriever:
    def __init__(self, indexer: HybridIndexer, rrf_k: int = 60) -> None:
        self.dense = DenseRetriever(indexer)
        self.sparse = SparseRetriever(indexer)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, dense_top_k: int, sparse_top_k: int, fusion_top_k: int) -> list[SearchResult]:
        dense_results = self.dense.retrieve(query, dense_top_k)
        sparse_results = self.sparse.retrieve(query, sparse_top_k)
        return reciprocal_rank_fusion([dense_results, sparse_results], self.rrf_k, fusion_top_k)

