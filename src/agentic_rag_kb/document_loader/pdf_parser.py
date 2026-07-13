"""PDF parser for paginated technical documents.

The parser emits one `Document` per page so page numbers remain available for
citations. Chunking is intentionally not performed here.
"""

from __future__ import annotations

from pathlib import Path

from agentic_rag_kb.document_loader.base import Document
from agentic_rag_kb.document_loader.cleaners import clean_document_text
from agentic_rag_kb.document_loader.ids import build_doc_id


class PdfDocumentParser:
    """Parse `.pdf` files using pypdf."""

    supported_extensions = {".pdf"}

    def load(self, path: Path) -> list[Document]:
        """Extract text from a PDF and return page-level documents."""

        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install pypdf or `pip install -e '.[documents]'` to parse PDFs.") from exc

        reader = PdfReader(str(path))
        title = _read_pdf_title(path, reader)
        documents: list[Document] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = clean_document_text(page.extract_text() or "")
            if not text:
                continue
            documents.append(
                Document(
                    doc_id=build_doc_id(path, text, suffix=page_index),
                    source_path=path,
                    title=title,
                    page_number=page_index,
                    text=text,
                    metadata={
                        "file_type": "pdf",
                        "filename": path.name,
                        "page_number": page_index,
                        "page_count": len(reader.pages),
                    },
                )
            )
        return documents


def _read_pdf_title(path: Path, reader) -> str:
    metadata_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    return str(metadata_title).strip() if metadata_title else path.stem

