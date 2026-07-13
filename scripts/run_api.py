from __future__ import annotations

from app.main import create_app_graph


def main() -> None:
    graph = create_app_graph()
    print("Graph initialized. FastAPI wiring can import app.api route helpers.")
    print(graph.invoke({"raw_query": "health check", "session_id": "api", "user_id": "api"})["final_answer"])


if __name__ == "__main__":
    main()

