"""RAGAS evaluation runner.

The evaluator accepts JSONL-style QA samples and reports the four metrics used by
the project: AnswerCorrectness, ContextRecall, Faithfulness, and ContextPrecision.
When RAGAS and model backends are available, this module can be extended to call
the official metrics directly. The deterministic fallback keeps the Gradio demo
usable in local environments without external model services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRIC_NAMES = ["AnswerCorrectness", "ContextRecall", "Faithfulness", "ContextPrecision"]


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation output for UI and CLI surfaces."""

    metrics: dict[str, float]
    sample_count: int
    mode: str
    rows: list[dict[str, Any]]
    warnings: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable output."""

        return {
            "metrics": self.metrics,
            "sample_count": self.sample_count,
            "mode": self.mode,
            "rows": self.rows,
            "warnings": self.warnings,
        }


class RagasEvaluator:
    """Run RAGAS-compatible metrics for Agentic RAG outputs."""

    def evaluate(self, samples: list[dict[str, Any]]) -> EvaluationResult:
        """Evaluate QA samples."""

        normalized = [_normalize_sample(sample) for sample in samples if sample]
        if not normalized:
            return EvaluationResult(
                metrics={metric: 0.0 for metric in METRIC_NAMES},
                sample_count=0,
                mode="fallback",
                rows=[],
                warnings=["empty_eval_dataset"],
            )

        rows = [_score_sample(sample) for sample in normalized]
        metrics = {
            metric: round(sum(row[metric] for row in rows) / len(rows), 3) for metric in METRIC_NAMES
        }
        return EvaluationResult(
            metrics=metrics,
            sample_count=len(rows),
            mode="fallback",
            rows=rows,
            warnings=["ragas_backend_not_configured_using_deterministic_fallback"],
        )

    def evaluate_jsonl(self, path: Path) -> EvaluationResult:
        """Read a JSONL dataset and evaluate it."""

        samples = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    samples.append(json.loads(line))
        return self.evaluate(samples)


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    question = str(sample.get("question") or sample.get("query") or "").strip()
    answer = str(sample.get("answer") or sample.get("response") or "").strip()
    ground_truth = str(sample.get("ground_truth") or sample.get("reference") or "").strip()
    contexts = sample.get("contexts") or sample.get("retrieved_contexts") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    contexts = [str(context.get("text", context)) if isinstance(context, dict) else str(context) for context in contexts]
    return {"question": question, "answer": answer, "ground_truth": ground_truth, "contexts": contexts}


def _score_sample(sample: dict[str, Any]) -> dict[str, Any]:
    answer_terms = _terms(sample["answer"])
    truth_terms = _terms(sample["ground_truth"])
    context_terms = _terms(" ".join(sample["contexts"]))
    answer_correctness = _overlap(answer_terms, truth_terms) if truth_terms else 0.0
    context_recall = _overlap(truth_terms, context_terms) if truth_terms else 0.0
    faithfulness = _overlap(answer_terms, context_terms) if answer_terms else 0.0
    context_precision = _overlap(context_terms, answer_terms | truth_terms) if context_terms else 0.0
    return {
        "question": sample["question"],
        "AnswerCorrectness": round(answer_correctness, 3),
        "ContextRecall": round(context_recall, 3),
        "Faithfulness": round(faithfulness, 3),
        "ContextPrecision": round(context_precision, 3),
    }


def _terms(text: str) -> set[str]:
    import re

    return set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text.lower()))


def _overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)
