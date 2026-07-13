from __future__ import annotations

from app.graphs.main_graph import MainGraph


def build_gradio_app(graph: MainGraph):
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install gradio to run the UI") from exc

    def respond(message: str, history: list):
        state = graph.invoke({"session_id": "gradio", "user_id": "ui", "raw_query": message})
        return state.get("final_answer", "")

    with gr.Blocks(title="Agentic RAG KB") as demo:
        gr.Markdown("# Agentic RAG Knowledge Base")
        gr.ChatInterface(respond)
    return demo
