from __future__ import annotations

from collections import defaultdict

from app.indexing.hybrid_indexer import HybridIndexer
from app.indexing.schemas import ParentContext, SearchResult


class ParentChildRetriever:
    def __init__(self, indexer: HybridIndexer) -> None:
        self.indexer = indexer

    def expand(self, child_hits: list[SearchResult]) -> list[ParentContext]:
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for hit in child_hits:
            grouped[hit.parent_id].append(hit)

        contexts: list[ParentContext] = []
        for parent_id, hits in grouped.items():
            parent = self.indexer.parents_by_id.get(parent_id)
            if parent is None:
                continue
            contexts.append(
                ParentContext(
                    parent_id=parent.parent_id,
                    doc_id=parent.doc_id,
                    text=parent.text,
                    score=max(hit.score for hit in hits),
                    child_hits=sorted(hits, key=lambda item: item.score, reverse=True),
                    metadata=parent.metadata,
                )
            )
        return sorted(contexts, key=lambda item: item.score, reverse=True)

