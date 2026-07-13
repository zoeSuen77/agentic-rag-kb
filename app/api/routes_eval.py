from __future__ import annotations

from app.evaluation.ragas_runner import evaluate_samples, summarize_scores


def evaluate(samples: list[dict]) -> dict:
    results = evaluate_samples(samples)
    return {"summary": summarize_scores(results), "results": results}

