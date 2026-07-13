"""JSONL IO helpers for chunking inputs and outputs."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk
from agentic_rag_kb.document_loader.base import Document


def read_documents_jsonl(path: Path) -> list[Document]:
    """Read standardized parsed documents from JSONL."""

    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append(
                Document(
                    doc_id=row["doc_id"],
                    source_path=Path(row["source_path"]),
                    title=row.get("title") or Path(row["source_path"]).stem,
                    page_number=row.get("page_number"),
                    section_title=row.get("section_title"),
                    text=row.get("text", ""),
                    metadata={**row.get("metadata", {}), "jsonl_line_number": line_number},
                )
            )
    return documents


def write_parent_chunks_jsonl(path: Path, chunks: list[ParentChunk]) -> None:
    """Write parent chunks to JSONL."""

    _write_jsonl(path, [chunk.to_json_dict() for chunk in chunks])


def write_child_chunks_jsonl(path: Path, chunks: list[ChildChunk]) -> None:
    """Write child chunks to JSONL."""

    _write_jsonl(path, [chunk.to_json_dict() for chunk in chunks])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
