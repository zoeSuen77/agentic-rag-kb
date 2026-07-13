from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from app.graphs.nodes.aggregation import aggregate_subanswers, check_faithfulness
from app.graphs.nodes.ambiguity import detect_ambiguity
from app.graphs.nodes.answer_generation import generate_final_answer
from app.graphs.nodes.compression import compress_context
from app.graphs.nodes.decomposition import classify_query, decompose_question, detect_complexity
from app.graphs.nodes.fallback import apply_fallback
from app.graphs.nodes.human_review import apply_human_clarification
from app.graphs.nodes.routing import build_send_tasks
from app.graphs.states import MainGraphState
from app.graphs.sub_retrieval_graph import build_sub_retrieval_graph
from app.indexing.hybrid_indexer import HybridIndexer
from app.settings import AppSettings


class MainGraph:
    def __init__(self, indexer: HybridIndexer, settings: AppSettings) -> None:
        self.indexer = indexer
        self.settings = settings
        self.sub_graph = build_sub_retrieval_graph(indexer, settings)

    def invoke(self, initial_state: MainGraphState) -> MainGraphState:
        state: MainGraphState = {
            "session_id": initial_state.get("session_id", "default"),
            "user_id": initial_state.get("user_id", "anonymous"),
            "raw_query": initial_state.get("raw_query", ""),
            "human_clarification": initial_state.get("human_clarification", ""),
            "conversation_history": initial_state.get("conversation_history", []),
            "loop_count": initial_state.get("loop_count", 0),
            "max_loops": initial_state.get("max_loops", 2),
            "errors": [],
        }

        state = compress_context(state)
        state = detect_ambiguity(state)
        state = apply_human_clarification(state)
        if state.get("ambiguity_result", {}).get("is_ambiguous") and not state.get("human_clarification"):
            state["final_answer"] = state.get("clarification_question", "请补充问题细节。")
            return state

        state = classify_query(state)
        state = detect_complexity(state)
        state = decompose_question(state)
        state["sub_tasks"] = [task.payload for task in build_send_tasks(state["decomposed_questions"], state["normalized_query"])]
        state["sub_results"] = self._dispatch_subtasks(state["sub_tasks"])
        state = aggregate_subanswers(state)
        state = check_faithfulness(state)
        if state.get("fallback_reason"):
            state = apply_fallback(state)
        state = generate_final_answer(state)
        return state

    def _dispatch_subtasks(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as executor:
            futures = [executor.submit(self.sub_graph.invoke, task.copy()) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda item: item.get("sub_query_id", ""))


def build_main_graph(indexer: HybridIndexer, settings: AppSettings) -> MainGraph:
    return MainGraph(indexer=indexer, settings=settings)

