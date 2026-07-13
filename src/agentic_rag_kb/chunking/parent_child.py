"""Parent-child hierarchical chunk construction.

This module converts parsed `Document` records into two levels:

- parent chunks: larger semantic blocks used as answer context;
- child chunks: smaller retrieval units that point back to parent chunks.

The chunker does not embed, index, rerank, or call LLMs.
"""

from __future__ import annotations

import hashlib
import re

from agentic_rag_kb.chunking.models import ChildChunk, ChunkingReport, ParentChunk
from agentic_rag_kb.chunking.text_splitter import RecursiveCharacterSplitter, estimate_tokens
from agentic_rag_kb.document_loader.base import LoadedDocument


class ParentChildChunker:
    """Build parent chunks and child chunks from loaded documents."""

    def __init__(
        self,
        parent_min_tokens: int = 500,
        parent_target_tokens: int = 1100,
        parent_max_tokens: int = 1500,
        child_min_tokens: int = 80,
        child_target_tokens: int = 250,
        child_max_tokens: int = 350,
    ) -> None:
        self.parent_splitter = RecursiveCharacterSplitter(
            min_tokens=parent_min_tokens,
            target_tokens=parent_target_tokens,
            max_tokens=parent_max_tokens,
        )
        self.child_splitter = RecursiveCharacterSplitter(
            min_tokens=child_min_tokens,
            target_tokens=child_target_tokens,
            max_tokens=child_max_tokens,
        )

    def split(self, documents: list[LoadedDocument]) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Split documents into parent and child chunks."""

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for document in documents:
            parent_texts = self.parent_splitter.split(document.text)
            for parent_index, parent_text in enumerate(parent_texts):
                title_path = build_title_path(document.title, document.section_title, parent_text)
                parent_id = _stable_chunk_id(
                    "parent",
                    document.doc_id,
                    parent_index,
                    parent_text,
                )
                parent_metadata = {
                    **document.metadata,
                    "doc_id": document.doc_id,
                    "source_path": str(document.source_path),
                    "title": document.title,
                    "section_title": document.section_title,
                    "title_path": title_path,
                    "chunk_index": parent_index,
                    "token_count": estimate_tokens(parent_text),
                }
                parent_chunk = ParentChunk(
                    parent_id=parent_id,
                    doc_id=document.doc_id,
                    source_path=str(document.source_path),
                    title=document.title,
                    title_path=title_path,
                    chunk_index=parent_index,
                    text=parent_text,
                    metadata=parent_metadata,
                )
                parents.append(parent_chunk)

                child_texts = self.child_splitter.split(parent_text)
                for child_index, child_text in enumerate(child_texts):
                    child_id = _stable_chunk_id(
                        "child",
                        document.doc_id,
                        f"{parent_index}:{child_index}",
                        child_text,
                    )
                    child_metadata = {
                        **document.metadata,
                        "doc_id": document.doc_id,
                        "parent_id": parent_id,
                        "source_path": str(document.source_path),
                        "title": document.title,
                        "section_title": document.section_title,
                        "title_path": title_path,
                        "chunk_index": child_index,
                        "parent_chunk_index": parent_index,
                        "token_count": estimate_tokens(child_text),
                    }
                    children.append(
                        ChildChunk(
                            child_id=child_id,
                            parent_id=parent_id,
                            doc_id=document.doc_id,
                            source_path=str(document.source_path),
                            title=document.title,
                            title_path=title_path,
                            chunk_index=child_index,
                            text=child_text,
                            metadata=child_metadata,
                        )
                    )

        return parents, children

    def build_report(
        self,
        document_count: int,
        parents: list[ParentChunk],
        children: list[ChildChunk],
    ) -> ChunkingReport:
        """Build aggregate chunking statistics."""

        parent_lengths = [estimate_tokens(chunk.text) for chunk in parents]
        child_lengths = [estimate_tokens(chunk.text) for chunk in children]
        return ChunkingReport(
            document_count=document_count,
            parent_chunk_count=len(parents),
            child_chunk_count=len(children),
            average_parent_tokens=_average(parent_lengths),
            average_child_tokens=_average(child_lengths),
            min_parent_tokens=min(parent_lengths) if parent_lengths else 0,
            max_parent_tokens=max(parent_lengths) if parent_lengths else 0,
            min_child_tokens=min(child_lengths) if child_lengths else 0,
            max_child_tokens=max(child_lengths) if child_lengths else 0,
        )


def build_title_path(title: str, section_title: str | None, text: str) -> str:
    """Build a human-readable title path from document and chunk headings."""

    parts = [title.strip()] if title.strip() else []
    if section_title and section_title.strip() and section_title.strip() not in parts:
        parts.append(section_title.strip())
    headings = _extract_heading_path(text)
    for heading in headings:
        if heading not in parts:
            parts.append(heading)
    return " > ".join(parts)


def _extract_heading_path(text: str) -> list[str]:
    heading_stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if not match:
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading))
    return [heading for _, heading in heading_stack]


def _stable_chunk_id(prefix: str, doc_id: str, index: int | str, text: str) -> str:
    payload = f"{doc_id}::{index}::{text[:1000]}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)
