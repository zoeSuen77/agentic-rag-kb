from __future__ import annotations

from app.graphs.states import SubRetrievalState
from app.indexing.hybrid_indexer import HybridIndexer
from app.retrieval.citation import build_citations
from app.retrieval.context_builder import build_context_block
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.parent_child_retriever import ParentChildRetriever
from app.retrieval.query_rewriter import rewrite_sub_query
from app.retrieval.reranker import LexicalCrossEncoderReranker
from app.settings import AppSettings


class SubRetrievalGraph:
    def __init__(self, indexer: HybridIndexer, settings: AppSettings) -> None:
        self.indexer = indexer
        self.settings = settings
        self.hybrid = HybridRetriever(indexer, settings.rrf_k)
        self.parent_expander = ParentChildRetriever(indexer)
        self.reranker = LexicalCrossEncoderReranker()

    def invoke(self, state: SubRetrievalState) -> SubRetrievalState:
        try:
            rewritten = rewrite_sub_query(
                parent_query=state.get("parent_query", ""),
                sub_query=state.get("sub_query", ""),
                intent=state.get("intent", ""),
            )
            dense = self.indexer.dense_search(rewritten, self.settings.dense_top_k)
            sparse = self.indexer.sparse_search(rewritten, self.settings.sparse_top_k)
            fused = self.hybrid.retrieve(
                rewritten,
                dense_top_k=self.settings.dense_top_k,
                sparse_top_k=self.settings.sparse_top_k,
                fusion_top_k=self.settings.fusion_top_k,
            )
            parents = self.parent_expander.expand(fused)
            reranked = self.reranker.rerank(rewritten, parents, self.settings.rerank_top_n)
            context_block = build_context_block(reranked, self.settings.parent_context_budget_chars)
            citations = build_citations(reranked)
            state.update(
                {
                    "rewritten_query": rewritten,
                    "dense_results": [item.to_dict() for item in dense],
                    "sparse_results": [item.to_dict() for item in sparse],
                    "fused_results": [item.to_dict() for item in fused],
                    "parent_contexts": [item.to_dict() for item in parents],
                    "reranked_contexts": [item.to_dict() for item in reranked],
                    "local_answer": self._generate_local_answer(state.get("sub_query", ""), context_block),
                    "local_citations": citations,
                    "retrieval_quality": {
                        "dense_hits": len(dense),
                        "sparse_hits": len(sparse),
                        "fused_hits": len(fused),
                        "parent_contexts": len(parents),
                        "reranked_contexts": len(reranked),
                    },
                }
            )
        except Exception as exc:  # defensive graph boundary
            state["error"] = str(exc)
            state["retrieval_quality"] = {"failed": True}
        return state

    def _generate_local_answer(self, sub_query: str, context_block: str) -> str:
        if not context_block:
            return "未在知识库中找到足够上下文。"
        first_context = context_block.split("\n\n", 1)[0]
        snippet = " ".join(first_context.splitlines()[1:])[:360]
        return f"针对子问题“{sub_query}”，相关文档表明：{snippet}"


def build_sub_retrieval_graph(indexer: HybridIndexer, settings: AppSettings) -> SubRetrievalGraph:
    return SubRetrievalGraph(indexer=indexer, settings=settings)

