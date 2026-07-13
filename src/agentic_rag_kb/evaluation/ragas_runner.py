"""RAGAS evaluation runner.

TODO:
- Convert graph traces to RAGAS dataset format.
- Run AnswerCorrectness, ContextRecall, Faithfulness, and ContextPrecision.
- Generate markdown and CSV reports.
"""


class RagasEvaluator:
    """Run RAGAS metrics for Agentic RAG outputs."""

    def evaluate(self, samples: list[dict]) -> dict:
        """Evaluate QA samples with RAGAS."""

        raise NotImplementedError("RAGAS evaluation will be implemented after answer traces exist.")

