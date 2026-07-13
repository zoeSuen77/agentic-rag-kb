"""Hybrid retrieval and reciprocal rank fusion.

This module implements the first retrieval stage:

1. dense search over Qdrant vectors;
2. sparse BM25 over stored sparse payloads;
3. Reciprocal Rank Fusion;
4. child-level and parent-level duplicate control;
5. parent context expansion for generation.
"""

from __future__ import annotations

from agentic_rag_kb.indexing.docstore import ParentDocStore
from agentic_rag_kb.indexing.embeddings import EmbeddingModel
from agentic_rag_kb.indexing.qdrant_store import DEFAULT_COLLECTION_NAME, VectorStore
from agentic_rag_kb.retrieval.dense import DenseRetriever
from agentic_rag_kb.retrieval.models import RetrievedChunk, RetrievalDebugInfo
from agentic_rag_kb.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """Combine dense and sparse retrieval for robust technical QA."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        parent_docstore: ParentDocStore,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        rrf_k: int = 60,
        max_children_per_parent: int = 2,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.parent_docstore = parent_docstore
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.max_children_per_parent = max_children_per_parent
        self.dense_retriever = DenseRetriever(vector_store, embedding_model, collection_name)
        self.sparse_retriever = SparseRetriever(vector_store, collection_name)
        self.last_debug_info = RetrievalDebugInfo()

    def retrieve(
        self,
        query: str,
        top_k_dense: int,
        top_k_sparse: int,
        final_k: int,
    ) -> list[RetrievedChunk]:
        """Return fused retrieval candidates with parent context in `text`."""

        dense_hits = self.dense_retriever.retrieve(query, top_k_dense)
        sparse_hits = self.sparse_retriever.retrieve(query, top_k_sparse)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, rrf_k=self.rrf_k)
        deduped = self._dedupe_and_limit_parents(fused, final_k)
        retrieved = [self._to_retrieved_chunk(item) for item in deduped]
        self.last_debug_info = RetrievalDebugInfo(
            dense_hits=_debug_hits(dense_hits, "score_dense"),
            sparse_hits=_debug_hits(sparse_hits, "score_sparse"),
            fused_ranking=[
                {
                    "rank": index,
                    "child_id": item["child_id"],
                    "parent_id": item["parent_id"],
                    "score_fused": item["score_fused"],
                    "score_dense": item.get("score_dense"),
                    "score_sparse": item.get("score_sparse"),
                }
                for index, item in enumerate(deduped, start=1)
            ],
            parent_contexts=[
                {
                    "child_id": chunk.child_id,
                    "parent_id": chunk.parent_id,
                    "parent_found": chunk.parent is not None,
                    "title_path": (chunk.parent or {}).get("title_path"),
                }
                for chunk in retrieved
            ],
        )
        return retrieved

    def get_debug_info(self) -> RetrievalDebugInfo:
        """Return debug details from the most recent retrieval call."""

        return self.last_debug_info

    def _dedupe_and_limit_parents(self, fused: list[dict], final_k: int) -> list[dict]:
        seen_children: set[str] = set()
        parent_counts: dict[str, int] = {}
        output: list[dict] = []
        for item in fused:
            child_id = item["child_id"]
            parent_id = item["parent_id"]
            if not child_id or child_id in seen_children:
                continue
            if parent_counts.get(parent_id, 0) >= self.max_children_per_parent:
                continue
            seen_children.add(child_id)
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
            output.append(item)
            if len(output) >= final_k:
                break
        return output

    def _to_retrieved_chunk(self, item: dict) -> RetrievedChunk:
        parent = self.parent_docstore.get_parent(item["parent_id"])
        parent_text = parent["text"] if parent else item.get("text", "")
        return RetrievedChunk(
            child_id=item["child_id"],
            parent_id=item["parent_id"],
            text=parent_text,
            score_dense=item.get("score_dense"),
            score_sparse=item.get("score_sparse"),
            score_fused=item["score_fused"],
            metadata=item.get("metadata", {}),
            parent=parent,
        )


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse dense and sparse rankings with Reciprocal Rank Fusion."""

    fused: dict[str, dict] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        child_id = hit.get("child_id")
        if not child_id:
            continue
        entry = fused.setdefault(child_id, _base_entry(hit))
        entry["score_dense"] = hit.get("score_dense")
        entry["score_fused"] += 1.0 / (rrf_k + rank)

    for rank, hit in enumerate(sparse_hits, start=1):
        child_id = hit.get("child_id")
        if not child_id:
            continue
        entry = fused.setdefault(child_id, _base_entry(hit))
        entry["score_sparse"] = hit.get("score_sparse")
        entry["score_fused"] += 1.0 / (rrf_k + rank)

    return sorted(fused.values(), key=lambda item: item["score_fused"], reverse=True)


def _base_entry(hit: dict) -> dict:
    return {
        "child_id": hit.get("child_id"),
        "parent_id": hit.get("parent_id"),
        "text": hit.get("text", ""),
        "metadata": hit.get("metadata", {}),
        "score_dense": None,
        "score_sparse": None,
        "score_fused": 0.0,
    }


def _debug_hits(hits: list[dict], score_key: str) -> list[dict]:
    return [
        {
            "rank": index,
            "child_id": hit.get("child_id"),
            "parent_id": hit.get("parent_id"),
            score_key: hit.get(score_key),
        }
        for index, hit in enumerate(hits, start=1)
    ]
