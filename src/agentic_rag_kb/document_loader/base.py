"""Base document ingestion contracts.

The document loader layer converts raw enterprise technical files into standardized
`Document` objects. It does not split chunks, write indexes, or call LLMs. Keeping
this boundary clean makes the downstream chunking and retrieval layers easier to
test and replace.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class Document:
    """Standard parsed document emitted by the ingestion pipeline.

    Attributes:
        doc_id: Stable identifier for this parsed document unit.
        source_path: Original file path.
        title: File-level or document-level title.
        text: Cleaned text content. Markdown headings, code blocks, and table text
            should be preserved where possible.
        page_number: Optional page number for paginated formats such as PDF.
        section_title: Optional section heading for structured formats.
        metadata: Extra source-specific metadata for citations and filters.
    """

    doc_id: str
    source_path: Path
    title: str
    text: str
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        """Backward-compatible alias used by early chunking skeletons."""

        return self.doc_id

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable representation for JSONL output."""

        return {
            "doc_id": self.doc_id,
            "source_path": str(self.source_path),
            "title": self.title,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "text": self.text,
            "metadata": self.metadata,
        }


LoadedDocument = Document


class DocumentLoader(Protocol):
    """Protocol for document loaders."""

    def load(self, path: Path) -> list[Document]:
        """Load one supported file into standardized documents."""

