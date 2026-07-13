"""Conversation memory storage interface and local JSON implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_rag_kb.memory.compression import ConversationMemory


class ConversationMemoryStore:
    """Store and retrieve conversation memory by session."""

    def load(self, session_id: str) -> dict[str, Any]:
        """Load memory for one session."""

        raise NotImplementedError

    def save(self, session_id: str, memory: dict[str, Any]) -> None:
        """Save memory for one session."""

        raise NotImplementedError


class LocalJSONMemoryStore(ConversationMemoryStore):
    """Local JSON file memory store for development and demos."""

    def __init__(self, path: Path | str = "data/memory/conversations.json") -> None:
        self.path = Path(path)

    def load(self, session_id: str) -> dict[str, Any]:
        """Load one session memory or return an empty memory payload."""

        data = self._read_all()
        return data.get(session_id, ConversationMemory().to_json_dict())

    def save(self, session_id: str, memory: dict[str, Any]) -> None:
        """Persist one session memory."""

        data = self._read_all()
        data[session_id] = memory
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
