from __future__ import annotations

from app.main import create_app_graph
from app.ui.gradio_app import build_gradio_app


def main() -> None:
    graph = create_app_graph()
    demo = build_gradio_app(graph)
    demo.launch()


if __name__ == "__main__":
    main()

