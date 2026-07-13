from __future__ import annotations

from app.utils.text import chunk_by_chars


def split_parent_sections(text: str, parent_size: int = 2400, overlap: int = 180) -> list[str]:
    return chunk_by_chars(text, parent_size, overlap)


def split_child_chunks(text: str, child_size: int = 520, overlap: int = 80) -> list[str]:
    return chunk_by_chars(text, child_size, overlap)

