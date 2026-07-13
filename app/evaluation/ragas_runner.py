from __future__ import annotations

from app.evaluation.metrics import answer_correctness, context_precision, context_recall, faithfulness


def evaluate_samples(samples: list[dict]) -> list[dict]:
    results: list[dict] = []
    for sample in samples:
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth") or sample.get("reference", "")
        answer = sample.get("answer", "")
        contexts = sample.get("contexts", [])
        results.append(
            {
                "question": question,
                "answer_correctness": answer_correctness(answer, ground_truth),
                "context_recall": context_recall(ground_truth, contexts),
                "faithfulness": faithfulness(answer, contexts),
                "context_precision": context_precision(question, contexts),
            }
        )
    return results


def summarize_scores(results: list[dict]) -> dict:
    metric_names = ["answer_correctness", "context_recall", "faithfulness", "context_precision"]
    if not results:
        return {metric: 0.0 for metric in metric_names}
    return {
        metric: round(sum(float(item.get(metric, 0.0)) for item in results) / len(results), 4)
        for metric in metric_names
    }

