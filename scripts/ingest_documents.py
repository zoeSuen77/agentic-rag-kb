"""CLI for document ingestion.

Usage:
    python scripts/ingest_documents.py --input data/raw --output data/processed/documents.jsonl
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

from agentic_rag_kb.document_loader import DocumentLoaderRouter


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Parse raw documents into standardized JSONL.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Input file or directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/documents.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def write_jsonl(output_path: Path, rows: list[dict]) -> None:
    """Write parsed documents as UTF-8 JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the ingestion CLI."""

    args = parse_args()
    router = DocumentLoaderRouter()
    documents = router.load(args.input)
    write_jsonl(args.output, [document.to_json_dict() for document in documents])
    print(f"Parsed {len(documents)} document records into {args.output}")


if __name__ == "__main__":
    main()
