"""User interface layer.

The UI module will expose a Gradio streaming chat application with document upload,
retrieval traces, human clarification flow, citations, and evaluation controls.
"""

from agentic_rag_kb.ui.gradio_app import create_gradio_app

__all__ = ["create_gradio_app"]

