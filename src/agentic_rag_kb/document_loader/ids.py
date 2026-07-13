"""Stable document ID helpers for ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path


def build_doc_id(path: Path, text: str, suffix: str | int | None = None) -> str:
    """Build a stable ID from source path, optional page/section suffix, and text."""

    payload = f"{path.resolve()}::{suffix or ''}::{text[:5000]}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"

