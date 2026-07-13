"""Ollama LLM client.

The baseline RAG chain depends on this small interface instead of importing
Ollama-specific code directly. That keeps tests deterministic and makes it easy
to swap local models such as qwen2.5, llama3.1, or mistral.
"""

from __future__ import annotations

import json
from typing import Iterator, Protocol
from urllib import request


class LLMClient(Protocol):
    """Minimal LLM interface used by RAG chains."""

    def generate(self, prompt: str) -> str:
        """Generate a complete answer."""

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """Stream answer tokens or chunks."""


class OllamaLLMClient:
    """HTTP client for Ollama's local generation API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        """Generate a complete answer with Ollama."""

        payload = {"model": self.model, "prompt": prompt, "stream": False}
        response = self._post_json("/api/generate", payload)
        return str(response.get("response", ""))

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """Stream answer chunks with Ollama."""

        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": True}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
            for raw_line in response:
                if not raw_line.strip():
                    continue
                data = json.loads(raw_line.decode("utf-8"))
                chunk = data.get("response", "")
                if chunk:
                    yield str(chunk)
                if data.get("done"):
                    break

    def _post_json(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

