"""Local filesystem document loader.

This loader will be the first implementation used by upload scripts and the Gradio UI.
It should parse files from disk and emit `LoadedDocument` instances.
"""

from pathlib import Path

from agentic_rag_kb.document_loader.base import LoadedDocument


class LocalDocumentLoader:
    """Load documents from the local filesystem.

    TODO:
    - Discover supported file extensions recursively.
    - Delegate parsing by file type.
    - Preserve metadata needed for citations and ACL filtering.
    """

    def load(self, path: Path) -> list[LoadedDocument]:
        """Load documents from a path."""

        raise NotImplementedError("Local document loading will be implemented in the ingestion phase.")

