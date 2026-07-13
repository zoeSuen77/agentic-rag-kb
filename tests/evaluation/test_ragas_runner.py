from __future__ import annotations

from app.evaluation.ragas_runner import evaluate_samples, summarize_scores


def test_evaluation_runner_returns_required_metrics() -> None:
    results = evaluate_samples(
        [
            {
                "question": "How to check CoreDNS?",
                "ground_truth": "check CoreDNS pods and service",
                "answer": "check CoreDNS pods",
                "contexts": ["CoreDNS pods and kube-dns service should be checked"],
            }
        ]
    )
    summary = summarize_scores(results)
    assert set(summary) == {"answer_correctness", "context_recall", "faithfulness", "context_precision"}
