"""JSONL IO helpers for indexing chunk files."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk


def read_parent_chunks_jsonl(path: Path) -> list[ParentChunk]:
    """Read parent chunks from JSONL."""

    return [ParentChunk(**row) for row in _read_jsonl(path)]


def read_child_chunks_jsonl(path: Path) -> list[ChildChunk]:
    """Read child chunks from JSONL."""

    return [ChildChunk(**row) for row in _read_jsonl(path)]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
