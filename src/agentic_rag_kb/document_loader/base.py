"""Base document loader contracts.

TODO:
- Implement PDF, Markdown, HTML, TXT, and DOCX loaders.
- Add document metadata extraction such as source, page, section, owner, and version.
- Add cleaning hooks for noisy enterprise exports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class LoadedDocument:
    """Raw parsed document before parent-child chunking."""

    document_id: str
    text: str
    source_path: Path | None = None
    metadata: dict = field(default_factory=dict)


class DocumentLoader(Protocol):
    """Protocol for document loaders."""

    def load(self, path: Path) -> list[LoadedDocument]:
        """Load one file or a directory of files into raw documents."""

