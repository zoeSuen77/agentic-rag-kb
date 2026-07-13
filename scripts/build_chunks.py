"""CLI for parent-child hierarchical chunking.

Usage:
    python scripts/build_chunks.py --input data/processed/documents.jsonl --output data/chunks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic_rag_kb.chunking.io import (  # noqa: E402
    read_documents_jsonl,
    write_child_chunks_jsonl,
    write_parent_chunks_jsonl,
)
from agentic_rag_kb.chunking.parent_child import ParentChildChunker  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build parent-child chunks from parsed documents.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/documents.jsonl"),
        help="Input parsed documents JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks"),
        help="Output directory for parent_chunks.jsonl and child_chunks.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    """Run parent-child chunking."""

    args = parse_args()
    documents = read_documents_jsonl(args.input)
    chunker = ParentChildChunker()
    parents, children = chunker.split(documents)
    report = chunker.build_report(len(documents), parents, children)

    args.output.mkdir(parents=True, exist_ok=True)
    write_parent_chunks_jsonl(args.output / "parent_chunks.jsonl", parents)
    write_child_chunks_jsonl(args.output / "child_chunks.jsonl", children)
    (args.output / "chunking_report.json").write_text(
        json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "Built "
        f"{len(parents)} parent chunks and {len(children)} child chunks "
        f"from {len(documents)} documents into {args.output}"
    )


if __name__ == "__main__":
    main()
