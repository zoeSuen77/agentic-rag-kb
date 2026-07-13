from __future__ import annotations

from pathlib import Path

from app.indexing.metadata_store import read_json


def load_eval_dataset(path: Path) -> list[dict]:
    data = read_json(path, [])
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a list of samples")
    return data

