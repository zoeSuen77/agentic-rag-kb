"""Complex question decomposition agent.

TODO:
- Detect whether a query is simple or complex.
- Decompose complex troubleshooting questions into independent retrieval tasks.
- Preserve task priority and intent.
"""


class QueryDecomposerAgent:
    """Agent that decomposes complex user questions."""

    def run(self, state: dict) -> dict:
        """Return decomposed subquestions."""

        raise NotImplementedError("Query decomposition will be implemented with LLM structured output.")

