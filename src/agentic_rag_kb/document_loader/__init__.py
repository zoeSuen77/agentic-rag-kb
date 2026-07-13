"""Document loading and parsing layer.

This module is responsible for loading enterprise technical documents from PDF,
Markdown, TXT, and DOCX files, then converting them into normalized `Document`
objects before chunking.
"""

from agentic_rag_kb.document_loader.base import Document, DocumentLoader, LoadedDocument
from agentic_rag_kb.document_loader.router import DocumentLoaderRouter

__all__ = ["Document", "DocumentLoader", "DocumentLoaderRouter", "LoadedDocument"]
