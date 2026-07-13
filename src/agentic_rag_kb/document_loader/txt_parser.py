"""Plain text parser for enterprise notes and exported technical documents."""

from __future__ import annotations

from pathlib import Path

from agentic_rag_kb.document_loader.base import Document
from agentic_rag_kb.document_loader.cleaners import clean_document_text
from agentic_rag_kb.document_loader.ids import build_doc_id


class TxtDocumentParser:
    """Parse `.txt` files into one standardized `Document`."""

    supported_extensions = {".txt"}

    def load(self, path: Path) -> list[Document]:
        """Read and clean a plain text file."""

        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_document_text(raw_text)
        title = path.stem
        return [
            Document(
                doc_id=build_doc_id(path, text),
                source_path=path,
                title=title,
                text=text,
                metadata={"file_type": "txt", "filename": path.name},
            )
        ]

