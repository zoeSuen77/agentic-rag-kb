"""Sparse retrieval payload support.

The first production path uses dense Qdrant search plus BM25-compatible sparse
payloads stored with each point. This keeps the schema ready for Dense+Sparse
hybrid retrieval while allowing Qdrant sparse vectors to be added later.
"""

from __future__ import annotations

from collections import Counter

from agentic_rag_kb.indexing.embeddings import TOKEN_PATTERN


def tokenize(text: str) -> list[str]:
    """Tokenize technical text for sparse payloads and BM25."""

    return TOKEN_PATTERN.findall(text.lower())


class SparsePayloadBuilder:
    """Build sparse payload fields for a child chunk."""

    def build(self, text: str, top_terms: int = 80) -> dict:
        """Return sparse terms and term frequencies for payload storage."""

        counts = Counter(tokenize(text))
        most_common = counts.most_common(top_terms)
        return {
            "sparse_terms": [term for term, _ in most_common],
            "sparse_term_freq": {term: count for term, count in most_common},
        }
