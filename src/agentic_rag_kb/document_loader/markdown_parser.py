"""Markdown parser for technical documentation.

Markdown structure is important for RAG because headings, tables, and fenced code
blocks often define the semantic boundaries later used by parent-child chunking.
This parser preserves Markdown text rather than rendering it away.
"""

from __future__ import annotations

import re
from pathlib import Path

from agentic_rag_kb.document_loader.base import Document
from agentic_rag_kb.document_loader.cleaners import clean_document_text, extract_markdown_title
from agentic_rag_kb.document_loader.ids import build_doc_id


class MarkdownDocumentParser:
    """Parse `.md` and `.markdown` files into structured section documents."""

    supported_extensions = {".md", ".markdown"}

    def load(self, path: Path) -> list[Document]:
        """Read Markdown and emit one document per top-level section."""

        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_document_text(raw_text)
        title = extract_markdown_title(text, fallback=path.stem)
        sections = _split_markdown_sections(text)
        if not sections:
            sections = [(None, text)]

        documents: list[Document] = []
        for index, (section_title, section_text) in enumerate(sections, start=1):
            cleaned_section = clean_document_text(section_text)
            documents.append(
                Document(
                    doc_id=build_doc_id(path, cleaned_section, suffix=index),
                    source_path=path,
                    title=title,
                    section_title=section_title,
                    text=cleaned_section,
                    metadata={
                        "file_type": "markdown",
                        "filename": path.name,
                        "section_index": index,
                    },
                )
            )
        return documents


def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Split Markdown on level-one and level-two headings, preserving heading text."""

    heading_matches = list(re.finditer(r"^(#{1,2})\s+(.+)$", text, flags=re.MULTILINE))
    if not heading_matches:
        return []

    sections: list[tuple[str | None, str]] = []
    for index, match in enumerate(heading_matches):
        start = match.start()
        end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(text)
        section_title = match.group(2).strip()
        sections.append((section_title, text[start:end].strip()))
    return sections

