from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


ROOT_DIR = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to load configuration files")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


@dataclass(frozen=True)
class AppSettings:
    root_dir: Path
    data_dir: Path
    memory_dir: Path
    index_dir: Path
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    qdrant_url: str
    qdrant_api_key: str | None
    child_collection: str
    parent_collection: str
    embedding_dimensions: int
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rerank_top_n: int
    rrf_k: int
    parent_context_budget_chars: int


def load_settings() -> AppSettings:
    app_cfg = _read_yaml(ROOT_DIR / "configs" / "app.yaml")
    model_cfg = _read_yaml(ROOT_DIR / "configs" / "model.yaml")
    qdrant_cfg = _read_yaml(ROOT_DIR / "configs" / "qdrant.yaml")
    retriever_cfg = _read_yaml(ROOT_DIR / "configs" / "retriever.yaml")

    data_dir = ROOT_DIR / app_cfg.get("data_dir", "data")
    memory_dir = ROOT_DIR / app_cfg.get("memory_dir", "data/memory")
    index_dir = ROOT_DIR / app_cfg.get("index_dir", "data/index")
    ollama = model_cfg.get("ollama", {})
    fallback = model_cfg.get("local_fallback", {})

    return AppSettings(
        root_dir=ROOT_DIR,
        data_dir=data_dir,
        memory_dir=memory_dir,
        index_dir=index_dir,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", ollama.get("base_url", "http://localhost:11434")),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", ollama.get("chat_model", "qwen2.5:7b")),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", ollama.get("embed_model", "nomic-embed-text")),
        qdrant_url=os.getenv("QDRANT_URL", qdrant_cfg.get("url", "http://localhost:6333")),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or qdrant_cfg.get("api_key"),
        child_collection=qdrant_cfg.get("child_collection", "kb_child_chunks"),
        parent_collection=qdrant_cfg.get("parent_collection", "kb_parent_chunks"),
        embedding_dimensions=int(qdrant_cfg.get("vector_size") or fallback.get("embedding_dimensions", 256)),
        dense_top_k=int(retriever_cfg.get("dense_top_k", 30)),
        sparse_top_k=int(retriever_cfg.get("sparse_top_k", 30)),
        fusion_top_k=int(retriever_cfg.get("fusion_top_k", 50)),
        rerank_top_n=int(retriever_cfg.get("rerank_top_n", 8)),
        rrf_k=int(retriever_cfg.get("rrf_k", 60)),
        parent_context_budget_chars=int(retriever_cfg.get("parent_context_budget_chars", 8000)),
    )

