from __future__ import annotations

from pathlib import Path

from app.ingestion.cleaners import clean_text
from app.ingestion.loaders import discover_files
from app.ingestion.parent_child_builder import build_parent_child_chunks
from app.ingestion.parsers import parse_file
from app.indexing.schemas import ChildChunk, ParentChunk


def ingest_path(path: Path) -> tuple[list[ParentChunk], list[ChildChunk]]:
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for file in discover_files(path):
        document = parse_file(file)
        document.text = clean_text(document.text)
        doc_parents, doc_children = build_parent_child_chunks(document)
        parents.extend(doc_parents)
        children.extend(doc_children)

    return parents, children
