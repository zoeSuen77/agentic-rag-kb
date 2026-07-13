"""Gradio demo UI for the Agentic RAG knowledge base."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from agentic_rag_kb.agents import ambiguity_detection_node, query_rewrite_node, task_decomposition_node
from agentic_rag_kb.chunking import ParentChildChunker
from agentic_rag_kb.config import get_settings
from agentic_rag_kb.document_loader import DocumentLoaderRouter
from agentic_rag_kb.evaluation.ragas_runner import RagasEvaluator
from agentic_rag_kb.graph import MainGraphDependencies, build_main_graph, default_main_graph_state
from agentic_rag_kb.graph.retrieval_subgraph import (
    RetrievalSubgraphDependencies,
    build_retrieval_subgraph,
)
from agentic_rag_kb.indexing import KnowledgeBaseIndexer, ParentDocStore
from agentic_rag_kb.indexing.embeddings import DeterministicEmbeddingModel
from agentic_rag_kb.indexing.qdrant_store import InMemoryVectorStore
from agentic_rag_kb.rerank import LexicalReranker, RerankConfig
from agentic_rag_kb.retrieval import HybridRetriever


UPLOAD_DIR = Path(os.getenv("AGENTIC_RAG_UI_UPLOAD_DIR", "/tmp/agentic_rag_kb_ui_uploads"))
DOCSTORE_PATH = Path(os.getenv("AGENTIC_RAG_UI_DOCSTORE", "/tmp/agentic_rag_kb_ui_docstore/parent_chunks.jsonl"))


@dataclass
class UIAppState:
    """Mutable backend state stored inside Gradio session state."""

    vector_store: InMemoryVectorStore | None = None
    embedding_model: DeterministicEmbeddingModel | None = None
    parent_docstore: ParentDocStore | None = None
    collection_name: str = "agentic_rag_ui_chunks"
    chat_history: list[dict[str, str]] = field(default_factory=list)
    pending_query: str = ""
    pending_state: dict[str, Any] = field(default_factory=dict)
    last_debug: dict[str, Any] = field(default_factory=dict)
    index_status: dict[str, Any] = field(default_factory=dict)


class GradioRAGService:
    """Backend service used by the Gradio callbacks."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def empty_state(self) -> UIAppState:
        """Create an empty UI state."""

        return UIAppState()

    def build_index(self, files: list[Any] | None, state: UIAppState | None) -> tuple[UIAppState, str, dict[str, Any]]:
        """Parse uploaded documents, build parent-child chunks, and index them."""

        state = state or self.empty_state()
        if not files:
            status = {"status": "no_files", "message": "请先上传 pdf/md/txt/docx 文件。"}
            return state, "未上传文件。", status

        try:
            upload_dir = _prepare_upload_dir(files)
            documents = DocumentLoaderRouter().load(upload_dir)
            chunker = ParentChildChunker()
            parents, children = chunker.split(documents)
            report = chunker.build_report(len(documents), parents, children).to_json_dict()

            vector_store = InMemoryVectorStore()
            embedding_model = DeterministicEmbeddingModel()
            parent_docstore = ParentDocStore(DOCSTORE_PATH)
            indexer = KnowledgeBaseIndexer(
                vector_store=vector_store,
                embedding_model=embedding_model,
                parent_docstore=parent_docstore,
                collection_name=state.collection_name,
            )
            indexer.build_index(parents, children)
            state.vector_store = vector_store
            state.embedding_model = embedding_model
            state.parent_docstore = parent_docstore
            state.index_status = {"status": "indexed", **report}
            message = (
                "索引构建完成。\n"
                f"- 文档数：{report['document_count']}\n"
                f"- parent chunks：{report['parent_chunk_count']}\n"
                f"- child chunks：{report['child_chunk_count']}"
            )
            return state, message, state.index_status
        except Exception as exc:
            status = {"status": "failed", "error": str(exc)}
            state.index_status = status
            return state, f"索引构建失败：{exc}", status

    def start_chat(
        self,
        query: str,
        history: list[dict[str, str]] | None,
        state: UIAppState | None,
    ) -> tuple[list[dict[str, str]], UIAppState, str, dict[str, Any], str]:
        """Start a chat turn and return either a clarification request or ready state."""

        state = state or self.empty_state()
        history = history or []
        query = query.strip()
        if not query:
            return history, state, "", state.last_debug, "请输入问题。"

        graph_state = default_main_graph_state(query)
        graph_state["chat_history"] = state.chat_history
        rewritten = query_rewrite_node(graph_state)
        ambiguous = ambiguity_detection_node(rewritten)
        state.pending_query = query
        state.pending_state = ambiguous
        if ambiguous.get("ambiguity_result", {}).get("is_ambiguous"):
            question = ambiguous.get("clarification_question") or "请补充更具体的系统、模块或文档名。"
            history = [*history, {"role": "user", "content": query}, {"role": "assistant", "content": question}]
            state.last_debug = _debug_from_state(ambiguous)
            return history, state, question, state.last_debug, "需要澄清后继续。"
        return history, state, "", _debug_from_state(ambiguous), "问题清晰，可以生成答案。"

    def stream_answer(
        self,
        query: str,
        clarification: str,
        history: list[dict[str, str]] | None,
        state: UIAppState | None,
    ) -> Iterator[tuple[list[dict[str, str]], UIAppState, str, dict[str, Any], str]]:
        """Generate an answer and stream it to the chat panel."""

        state = state or self.empty_state()
        history = history or []
        if not state.vector_store or not state.embedding_model or not state.parent_docstore:
            answer = "当前还没有可用索引。请先上传文档并点击“构建索引”。"
            yield [*history, {"role": "assistant", "content": answer}], state, "", state.last_debug, "未构建索引。"
            return

        graph_state = dict(state.pending_state) if state.pending_state else default_main_graph_state(query)
        if clarification.strip():
            graph_state["user_clarification"] = clarification.strip()
            graph_state = query_rewrite_node(graph_state)
            graph_state = ambiguity_detection_node(graph_state)
        graph_state = task_decomposition_node(graph_state)
        graph_state["chat_history"] = state.chat_history

        try:
            graph = self._build_agentic_graph(state)
            result = graph.invoke(graph_state)
            state.chat_history = result.get("chat_history", state.chat_history)
            state.last_debug = _debug_from_state(result)
            answer = result.get("final_answer", "")
            citations = _format_citations(result.get("retrieved_contexts", []))
            if citations:
                answer = f"{answer}\n\n引用来源：\n{citations}"
            streamed = ""
            if not history or history[-1].get("role") != "user" or history[-1].get("content") != graph_state.get("original_query"):
                history = [*history, {"role": "user", "content": graph_state.get("original_query", query)}]
            history = [*history, {"role": "assistant", "content": ""}]
            for chunk in _stream_chunks(answer):
                streamed += chunk
                history[-1] = {"role": "assistant", "content": streamed}
                yield history, state, "", state.last_debug, "生成中..."
            yield history, state, "", state.last_debug, "完成。"
        except Exception as exc:
            answer = f"执行 Agentic RAG 流程失败：{exc}"
            history = [*history, {"role": "assistant", "content": answer}]
            state.last_debug = {"error": str(exc), **state.last_debug}
            yield history, state, "", state.last_debug, "失败。"

    def run_evaluation(self, dataset_file: Any | None) -> tuple[list[list[Any]], dict[str, Any], str]:
        """Run RAGAS-compatible evaluation from an uploaded JSONL file."""

        if dataset_file is None:
            return [], {}, "请上传 eval_dataset.jsonl。"
        path = Path(getattr(dataset_file, "name", dataset_file))
        result = RagasEvaluator().evaluate_jsonl(path)
        rows = [
            [
                row["question"],
                row["AnswerCorrectness"],
                row["ContextRecall"],
                row["Faithfulness"],
                row["ContextPrecision"],
            ]
            for row in result.rows
        ]
        summary = (
            f"评测完成：{result.sample_count} samples，mode={result.mode}\n"
            + "\n".join(f"- {metric}: {score}" for metric, score in result.metrics.items())
        )
        return rows, result.to_json_dict(), summary

    def _build_agentic_graph(self, state: UIAppState):
        retriever = HybridRetriever(
            vector_store=state.vector_store,
            embedding_model=state.embedding_model,
            parent_docstore=state.parent_docstore,
            collection_name=state.collection_name,
        )
        reranker = LexicalReranker(
            RerankConfig(
                enable_rerank=self.settings.enable_rerank,
                rerank_top_n=self.settings.rerank_top_n,
                final_context_k=self.settings.final_context_k,
            )
        )
        subgraph = build_retrieval_subgraph(
            RetrievalSubgraphDependencies(retriever=retriever, reranker=reranker, llm_client=None)
        )
        return build_main_graph(MainGraphDependencies(retrieval_subgraph=subgraph, llm_client=None))


def create_gradio_app():
    """Create the Gradio Blocks app."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install gradio to run the UI: pip install -e '.[dev]' or pip install gradio") from exc

    service = GradioRAGService()

    with gr.Blocks(title="Agentic RAG KB", theme=gr.themes.Soft()) as demo:
        app_state = gr.State(service.empty_state())
        gr.Markdown("# Agentic RAG 知识库问答")

        with gr.Tab("知识库"):
            files = gr.File(label="上传文档", file_count="multiple", file_types=[".pdf", ".md", ".txt", ".docx"])
            build_btn = gr.Button("构建索引", variant="primary")
            index_status = gr.Textbox(label="索引状态", lines=5)
            index_json = gr.JSON(label="Chunk / Index Report")
            build_btn.click(service.build_index, inputs=[files, app_state], outputs=[app_state, index_status, index_json])

        with gr.Tab("对话"):
            chatbot = gr.Chatbot(label="Agentic RAG", type="messages", height=420)
            query = gr.Textbox(label="用户问题", placeholder="例如：父子分层索引和混合检索分别解决什么问题？")
            clarification_box = gr.Textbox(label="澄清回答", placeholder="如果系统要求澄清，在这里补充。")
            clarify_question = gr.Textbox(label="澄清问题", interactive=False)
            status = gr.Textbox(label="状态", interactive=False)
            with gr.Row():
                check_btn = gr.Button("检查问题 / 触发澄清")
                answer_btn = gr.Button("生成答案", variant="primary")

        with gr.Accordion("Debug 面板", open=False):
            debug_json = gr.JSON(label="Graph Debug")

        with gr.Tab("Evaluation"):
            eval_file = gr.File(label="上传 eval_dataset.jsonl", file_types=[".jsonl"])
            eval_btn = gr.Button("运行 RAGAS")
            eval_table = gr.Dataframe(
                headers=["question", "AnswerCorrectness", "ContextRecall", "Faithfulness", "ContextPrecision"],
                label="RAGAS Metrics",
            )
            eval_json = gr.JSON(label="Evaluation Debug")
            eval_status = gr.Textbox(label="评测状态")

        check_btn.click(
            service.start_chat,
            inputs=[query, chatbot, app_state],
            outputs=[chatbot, app_state, clarify_question, debug_json, status],
        )
        answer_btn.click(
            service.stream_answer,
            inputs=[query, clarification_box, chatbot, app_state],
            outputs=[chatbot, app_state, clarify_question, debug_json, status],
        )
        eval_btn.click(service.run_evaluation, inputs=[eval_file], outputs=[eval_table, eval_json, eval_status])

    return demo


def _prepare_upload_dir(files: list[Any]) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for existing in UPLOAD_DIR.iterdir():
        if existing.is_file():
            existing.unlink()
    for file_obj in files:
        source = Path(getattr(file_obj, "name", file_obj))
        if source.exists():
            shutil.copy2(source, UPLOAD_DIR / source.name)
    return UPLOAD_DIR


def _debug_from_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "rewritten_query": state.get("rewritten_query"),
        "ambiguity_result": state.get("ambiguity_result"),
        "decomposed_tasks": state.get("decomposed_tasks"),
        "retrieval_debug": state.get("retrieval_debug"),
        "sub_answers": state.get("sub_answers"),
        "final_contexts": state.get("retrieved_contexts"),
        "aggregation_debug": state.get("aggregation_debug"),
        "memory_debug": state.get("memory_debug"),
        "error_messages": state.get("error_messages"),
    }


def _format_citations(contexts: list[dict[str, Any]]) -> str:
    sources = []
    for context in contexts:
        source = (
            context.get("metadata", {}).get("source_path")
            or (context.get("parent") or {}).get("source_path")
            or context.get("parent_id")
        )
        rerank_score = context.get("rerank_score")
        label = f"- {source}"
        if rerank_score is not None:
            label += f" (rerank_score={rerank_score})"
        if source and label not in sources:
            sources.append(label)
    return "\n".join(sources)


def _stream_chunks(text: str, chunk_size: int = 24) -> Iterator[str]:
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]
