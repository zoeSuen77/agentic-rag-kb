"""Local filesystem document loader.

This compatibility wrapper delegates to `DocumentLoaderRouter`. It is useful for
call sites that want a generic local loader without depending on parser details.
"""

from pathlib import Path

from agentic_rag_kb.document_loader.base import Document
from agentic_rag_kb.document_loader.router import DocumentLoaderRouter


class LocalDocumentLoader:
    """Load supported documents from the local filesystem."""

    def __init__(self, router: DocumentLoaderRouter | None = None) -> None:
        self.router = router or DocumentLoaderRouter()

    def load(self, path: Path) -> list[Document]:
        """Load documents from a path."""

        return self.router.load(path)

