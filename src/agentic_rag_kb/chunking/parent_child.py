"""Parent-child chunk construction.

TODO:
- Implement structure-aware splitting for headings, sections, pages, and code blocks.
- Add overlap controls for parent and child chunks.
- Preserve offsets for citation highlighting.
"""

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk
from agentic_rag_kb.document_loader.base import LoadedDocument


class ParentChildChunker:
    """Build parent chunks and child chunks from loaded documents."""

    def split(self, documents: list[LoadedDocument]) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Split documents into parent and child chunks."""

        raise NotImplementedError("Parent-child chunking will be implemented after loader contracts settle.")

