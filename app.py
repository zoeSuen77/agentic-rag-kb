"""Application entry point for the Gradio Agentic RAG demo.

Run:
    python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic_rag_kb.ui.gradio_app import create_gradio_app  # noqa: E402


def main() -> None:
    """Launch the Gradio app."""

    app = create_gradio_app()
    app.launch()


if __name__ == "__main__":
    main()
