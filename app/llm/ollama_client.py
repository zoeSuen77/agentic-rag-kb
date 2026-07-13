from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request


@dataclass(slots=True)
class OllamaClient:
    base_url: str
    chat_model: str
    embed_model: str
    timeout_seconds: int = 60

    def generate(self, prompt: str) -> str:
        payload = json.dumps({"model": self.chat_model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return data.get("response", "")

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.embed_model, "prompt": text}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return data.get("embedding", [])

