"""Answer aggregation agent.

TODO:
- Merge local subanswers.
- Resolve conflicting evidence.
- Deduplicate citations.
- Produce final answer outline.
"""


class AnswerAggregatorAgent:
    """Agent that aggregates parallel retrieval subanswers."""

    def run(self, state: dict) -> dict:
        """Return aggregated answer state."""

        raise NotImplementedError("Answer aggregation will be implemented after retrieval subgraph.")

