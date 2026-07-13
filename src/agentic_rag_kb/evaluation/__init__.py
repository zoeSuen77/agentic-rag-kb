"""Evaluation layer.

The evaluation module runs RAGAS metrics over curated QA datasets and graph traces,
including AnswerCorrectness, ContextRecall, Faithfulness, and ContextPrecision.
"""

from agentic_rag_kb.evaluation.ragas_runner import RagasEvaluator

__all__ = ["RagasEvaluator"]

