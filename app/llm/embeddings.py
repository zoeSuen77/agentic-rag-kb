from __future__ import annotations

import math

from app.utils.hashing import stable_hash
from app.utils.text import tokenize


class HashEmbeddingModel:
    """Deterministic local embedding fallback for development and tests."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            bucket = int(stable_hash(token), 16) % self.dimensions
            sign = 1.0 if int(stable_hash(f"sign:{token}"), 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(length))

