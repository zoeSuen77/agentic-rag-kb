from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.indexing.schemas import ChildChunk, SearchResult
from app.utils.text import tokenize


class SparseIndexer:
    def __init__(self) -> None:
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_length = 1.0

    def index(self, chunks: list[ChildChunk]) -> None:
        self.term_freqs.clear()
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        for chunk in chunks:
            terms = tokenize(chunk.text)
            counter = Counter(terms)
            self.term_freqs[chunk.child_id] = counter
            self.doc_lengths[chunk.child_id] = len(terms)
            for term in counter:
                self.doc_freqs[term] += 1
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def search(self, query: str, chunks_by_id: dict[str, ChildChunk], top_k: int) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scores: dict[str, float] = defaultdict(float)
        total_docs = max(len(self.term_freqs), 1)
        k1 = 1.5
        b = 0.75
        for term in query_terms:
            doc_freq = self.doc_freqs.get(term, 0)
            if doc_freq == 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            for child_id, freqs in self.term_freqs.items():
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                length = self.doc_lengths.get(child_id, 0)
                denom = tf + k1 * (1 - b + b * length / self.avg_doc_length)
                scores[child_id] += idf * ((tf * (k1 + 1)) / denom)

        results: list[SearchResult] = []
        for child_id, score in scores.items():
            chunk = chunks_by_id.get(child_id)
            if chunk is None:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.child_id,
                    parent_id=chunk.parent_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                    source="sparse",
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

