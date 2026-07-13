from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.dataset_builder import load_eval_dataset
from app.evaluation.ragas_runner import evaluate_samples, summarize_scores
from app.evaluation.reports import write_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, default=Path("data/eval/reports/latest.md"))
    args = parser.parse_args()

    samples = load_eval_dataset(args.dataset)
    results = evaluate_samples(samples)
    summary = summarize_scores(results)
    write_markdown_report(args.report, summary, results)
    print(summary)


if __name__ == "__main__":
    main()

