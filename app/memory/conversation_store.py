from __future__ import annotations

from pathlib import Path

from app.indexing.metadata_store import read_json, write_json


class ConversationStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self, session_id: str) -> list[dict]:
        return read_json(self.directory / f"{session_id}.json", [])

    def append(self, session_id: str, role: str, content: str) -> None:
        history = self.load(session_id)
        history.append({"role": role, "content": content})
        write_json(self.directory / f"{session_id}.json", history)

