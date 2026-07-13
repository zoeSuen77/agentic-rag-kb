from __future__ import annotations

from pathlib import Path


def write_markdown_report(path: Path, summary: dict, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# RAGAS Evaluation Report", ""]
    for metric, value in summary.items():
        lines.append(f"- {metric}: {value}")
    lines.extend(["", "## Samples", ""])
    for item in results:
        lines.append(f"### {item.get('question')}")
        for metric, value in item.items():
            if metric != "question":
                lines.append(f"- {metric}: {value:.4f}" if isinstance(value, float) else f"- {metric}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

