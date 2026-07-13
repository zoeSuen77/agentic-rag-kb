from __future__ import annotations

from app.ingestion.parent_child_builder import build_parent_child_chunks
from app.indexing.schemas import RawDocument


def test_parent_child_builder_preserves_relationships() -> None:
    doc = RawDocument(
        doc_id="doc_1",
        source="manual.md",
        text="CoreDNS failure troubleshooting. Check pods, service, network policy. " * 80,
        metadata={"filename": "manual.md"},
    )
    parents, children = build_parent_child_chunks(doc)
    assert parents
    assert children
    parent_ids = {parent.parent_id for parent in parents}
    assert all(child.parent_id in parent_ids for child in children)
    assert all(child.doc_id == "doc_1" for child in children)

