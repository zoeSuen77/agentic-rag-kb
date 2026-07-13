"""Parent chunk docstore.

Child chunks are indexed in Qdrant for retrieval. Parent chunks are stored locally
so answer generation can expand child hits back to full context.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rag_kb.chunking.models import ParentChunk


class ParentDocStore:
    """JSONL-backed parent chunk store."""

    def __init__(self, path: Path = Path("data/docstore/parent_chunks.jsonl")) -> None:
        self.path = path
        self._parents: dict[str, dict] | None = None

    def save(self, parents: list[ParentChunk]) -> None:
        """Persist parent chunks to JSONL and refresh in-memory cache."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for parent in parents:
                file.write(json.dumps(parent.to_json_dict(), ensure_ascii=False) + "\n")
        self._parents = {parent.parent_id: parent.to_json_dict() for parent in parents}

    def get_parent(self, parent_id: str) -> dict | None:
        """Return one parent chunk by ID."""

        return self._load().get(parent_id)

    def get_many(self, parent_ids: list[str]) -> list[dict]:
        """Return parent chunks in the requested order, skipping missing IDs."""

        parents = self._load()
        return [parents[parent_id] for parent_id in parent_ids if parent_id in parents]

    def _load(self) -> dict[str, dict]:
        if self._parents is not None:
            return self._parents
        parents: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    parents[row["parent_id"]] = row
        self._parents = parents
        return parents
