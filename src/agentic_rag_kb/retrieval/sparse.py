"""Sparse lexical retrieval.

TODO:
- Implement rank-bm25 local retrieval or fastembed sparse vector search.
- Prioritize exact technical terms such as error codes, config keys, and class names.
"""


class SparseRetriever:
    """Lexical retriever for keyword-heavy technical questions."""

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Return sparse retrieval candidates."""

        raise NotImplementedError("Sparse retrieval will be implemented with BM25 or fastembed sparse.")

