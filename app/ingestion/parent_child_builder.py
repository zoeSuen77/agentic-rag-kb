from __future__ import annotations

from app.ingestion.splitters import split_child_chunks, split_parent_sections
from app.indexing.schemas import ChildChunk, ParentChunk, RawDocument
from app.utils.hashing import stable_hash


def build_parent_child_chunks(document: RawDocument) -> tuple[list[ParentChunk], list[ChildChunk]]:
    title = document.metadata.get("filename") or document.source
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for parent_index, parent_text in enumerate(split_parent_sections(document.text)):
        parent_id = stable_hash(
            {"doc_id": document.doc_id, "parent_index": parent_index, "text": parent_text[:256]},
            prefix="parent",
        )
        section_path = [str(title), f"section-{parent_index + 1}"]
        parent = ParentChunk(
            parent_id=parent_id,
            doc_id=document.doc_id,
            text=parent_text,
            title=str(title),
            section_path=section_path,
            metadata={**document.metadata, "parent_index": parent_index},
        )
        parents.append(parent)

        for child_index, child_text in enumerate(split_child_chunks(parent_text)):
            child_id = stable_hash(
                {
                    "doc_id": document.doc_id,
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "text": child_text[:256],
                },
                prefix="child",
            )
            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    doc_id=document.doc_id,
                    text=child_text,
                    title=str(title),
                    section_path=section_path,
                    metadata={**document.metadata, "parent_index": parent_index, "child_index": child_index},
                )
            )

    return parents, children

