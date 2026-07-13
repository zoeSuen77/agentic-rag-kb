"""Document loader router.

`DocumentLoaderRouter` is the public ingestion entry point. It discovers supported
files under an input path and delegates each file to the parser registered for its
extension.
"""

from __future__ import annotations

from pathlib import Path

from agentic_rag_kb.document_loader.base import Document, DocumentLoader
from agentic_rag_kb.document_loader.docx_parser import DocxDocumentParser
from agentic_rag_kb.document_loader.markdown_parser import MarkdownDocumentParser
from agentic_rag_kb.document_loader.pdf_parser import PdfDocumentParser
from agentic_rag_kb.document_loader.txt_parser import TxtDocumentParser


class DocumentLoaderRouter:
    """Route files to parser implementations by suffix."""

    def __init__(self, parsers: list[DocumentLoader] | None = None) -> None:
        self.parsers = parsers or [
            PdfDocumentParser(),
            MarkdownDocumentParser(),
            TxtDocumentParser(),
            DocxDocumentParser(),
        ]
        self._parser_by_extension = self._build_parser_map(self.parsers)

    @property
    def supported_extensions(self) -> set[str]:
        """Return all supported lower-case file extensions."""

        return set(self._parser_by_extension)

    def load(self, path: Path) -> list[Document]:
        """Load all supported documents from a file or directory."""

        documents: list[Document] = []
        for file_path in self.discover_files(path):
            parser = self.get_parser(file_path)
            documents.extend(parser.load(file_path))
        return documents

    def discover_files(self, path: Path) -> list[Path]:
        """Discover supported files under `path`."""

        if path.is_file():
            return [path] if path.suffix.lower() in self.supported_extensions else []
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in self.supported_extensions
        )

    def get_parser(self, path: Path) -> DocumentLoader:
        """Return the parser registered for `path`."""

        extension = path.suffix.lower()
        try:
            return self._parser_by_extension[extension]
        except KeyError as exc:
            raise ValueError(f"Unsupported document extension: {extension}") from exc

    def _build_parser_map(self, parsers: list[DocumentLoader]) -> dict[str, DocumentLoader]:
        parser_by_extension: dict[str, DocumentLoader] = {}
        for parser in parsers:
            extensions = getattr(parser, "supported_extensions", set())
            for extension in extensions:
                parser_by_extension[extension.lower()] = parser
        return parser_by_extension

