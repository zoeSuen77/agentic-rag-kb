"""Retrieval subgraph builder.

Each subgraph handles a single subquestion lifecycle: rewrite, hybrid retrieval,
parent expansion, rerank, local answer generation, and citation packaging.
"""


def build_retrieval_subgraph():
    """Build the retrieval subgraph for one decomposed question."""

    raise NotImplementedError("Retrieval subgraph wiring will be implemented with LangGraph.")

