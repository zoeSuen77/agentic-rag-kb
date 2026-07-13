"""Recursive text splitting utilities for hierarchical chunking.

The splitter estimates token length locally and recursively splits over semantic
separators: headings, blank lines, line breaks, sentence punctuation, and spaces.
Short chunks are adaptively merged with neighbors, while long chunks are split
again until they fit the configured maximum.
"""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:#-]+|[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """Estimate token count without requiring a tokenizer dependency."""

    return len(TOKEN_PATTERN.findall(text))


class RecursiveCharacterSplitter:
    """Recursive character splitter with adaptive merge and split behavior."""

    def __init__(
        self,
        min_tokens: int,
        target_tokens: int,
        max_tokens: int,
        separators: tuple[str, ...] | None = None,
    ) -> None:
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.separators = separators or ("\n# ", "\n## ", "\n\n", "\n", "。", ". ", " ", "")

    def split(self, text: str) -> list[str]:
        """Split text into reasonably sized chunks."""

        normalized = text.strip()
        if not normalized:
            return []
        raw_chunks = self._recursive_split(normalized, self.separators)
        merged = self._merge_short_chunks(raw_chunks)
        return self._ensure_max_size(merged)

    def _recursive_split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if estimate_tokens(text) <= self.max_tokens:
            return [text.strip()]
        if not separators:
            return [text.strip()]

        separator = separators[0]
        rest = separators[1:]
        if separator == "":
            return self._hard_split(text)

        parts = _split_keep_separator(text, separator)
        if len(parts) <= 1:
            return self._recursive_split(text, rest)

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{part}" if current else part
            if estimate_tokens(candidate) <= self.target_tokens:
                current = candidate
                continue
            if current:
                chunks.extend(self._recursive_split(current.strip(), rest))
            current = part
        if current:
            chunks.extend(self._recursive_split(current.strip(), rest))
        return [chunk for chunk in chunks if chunk.strip()]

    def _merge_short_chunks(self, chunks: list[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""
        for chunk in chunks:
            if not buffer:
                buffer = chunk
                continue
            candidate = f"{buffer}\n\n{chunk}"
            if estimate_tokens(buffer) < self.min_tokens and estimate_tokens(candidate) <= self.max_tokens:
                buffer = candidate
            else:
                merged.append(buffer.strip())
                buffer = chunk
        if buffer:
            if merged and estimate_tokens(buffer) < self.min_tokens:
                candidate = f"{merged[-1]}\n\n{buffer}"
                if estimate_tokens(candidate) <= self.max_tokens:
                    merged[-1] = candidate.strip()
                else:
                    merged.append(buffer.strip())
            else:
                merged.append(buffer.strip())
        return [chunk for chunk in merged if chunk.strip()]

    def _ensure_max_size(self, chunks: list[str]) -> list[str]:
        result: list[str] = []
        for chunk in chunks:
            if estimate_tokens(chunk) <= self.max_tokens:
                result.append(chunk)
            else:
                result.extend(self._recursive_split(chunk, self.separators[1:]))
        return [chunk for chunk in result if chunk.strip()]

    def _hard_split(self, text: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall(text)
        if not tokens:
            return [text[: self.max_tokens * 4]]
        chunks: list[str] = []
        for start in range(0, len(tokens), self.target_tokens):
            chunks.append(" ".join(tokens[start : start + self.target_tokens]))
        return chunks


def _split_keep_separator(text: str, separator: str) -> list[str]:
    if separator not in text:
        return [text]
    raw_parts = text.split(separator)
    parts: list[str] = []
    for index, part in enumerate(raw_parts):
        if not part:
            continue
        if index == 0:
            parts.append(part)
        else:
            parts.append(f"{separator}{part}")
    return parts
