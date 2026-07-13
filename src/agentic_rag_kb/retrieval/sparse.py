"""Sparse lexical retrieval.

The first implementation uses BM25 over sparse payload terms stored with each
Qdrant point. This preserves the Dense+Sparse architecture without requiring
Qdrant sparse vectors on day one.
"""

from __future__ import annotations

import math
from collections import Counter

from agentic_rag_kb.indexing.qdrant_store import VectorStore
from agentic_rag_kb.indexing.sparse import tokenize


class SparseRetriever:
    """BM25 retriever for keyword-heavy technical questions."""

    def __init__(self, vector_store: VectorStore, collection_name: str) -> None:
        self.vector_store = vector_store
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Return sparse retrieval candidates."""

        payloads = self.vector_store.scroll_payloads(self.collection_name)
        query_terms = tokenize(query)
        if not payloads or not query_terms:
            return []

        term_freqs = [_term_freq(payload) for payload in payloads]
        doc_lengths = [sum(freqs.values()) for freqs in term_freqs]
        avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
        doc_freq = Counter(term for freqs in term_freqs for term in freqs)

        scored: list[dict] = []
        for payload, freqs, doc_length in zip(payloads, term_freqs, doc_lengths):
            score = _bm25_score(
                query_terms=query_terms,
                term_freq=freqs,
                doc_freq=doc_freq,
                total_docs=len(payloads),
                doc_length=doc_length,
                avg_doc_length=avg_doc_length,
            )
            if score <= 0:
                continue
            scored.append(
                {
                    "child_id": payload.get("child_chunk_id"),
                    "parent_id": payload.get("parent_id"),
                    "text": payload.get("text", ""),
                    "score_sparse": score,
                    "metadata": payload.get("metadata", {}),
                    "payload": payload,
                }
            )
        return sorted(scored, key=lambda item: item["score_sparse"], reverse=True)[:top_k]


def _term_freq(payload: dict) -> Counter[str]:
    sparse_freq = payload.get("sparse_term_freq") or {}
    return Counter({str(term): int(count) for term, count in sparse_freq.items()})


def _bm25_score(
    query_terms: list[str],
    term_freq: Counter[str],
    doc_freq: Counter[str],
    total_docs: int,
    doc_length: int,
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    score = 0.0
    for term in query_terms:
        tf = term_freq.get(term, 0)
        if tf == 0:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * doc_length / max(avg_doc_length, 1e-9))
        score += idf * (tf * (k1 + 1)) / denom
    return score
