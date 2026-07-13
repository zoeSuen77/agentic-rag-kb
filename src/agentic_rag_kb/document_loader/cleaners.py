"""Text cleaning utilities for document ingestion.

The cleaner normalizes noisy enterprise document text without destroying structure
that matters for RAG: headings, code fences, and table-like rows should remain
available to downstream chunking and citation logic.
"""

from __future__ import annotations

import re
from collections import Counter


HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+.+$", re.MULTILINE)
CODE_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)


def normalize_newlines(text: str) -> str:
    """Normalize newline characters and trim trailing spaces."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def collapse_excess_blank_lines(text: str) -> str:
    """Collapse three or more blank lines into a maximum of two."""

    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text)


def remove_obvious_page_noise(text: str) -> str:
    """Remove common page headers, footers, and page number noise.

    This function intentionally uses conservative rules. It removes obvious page
    markers and repeated short lines, while preserving headings, code blocks, and
    table rows.
    """

    text = _protect_code_blocks(text)
    lines = text.split("\n")
    repeated_short_lines = _find_repeated_short_lines(lines)
    cleaned: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _is_page_marker(stripped):
            continue
        if stripped in repeated_short_lines:
            continue
        cleaned.append(line)

    return _restore_code_blocks("\n".join(cleaned))


def clean_document_text(text: str) -> str:
    """Apply the standard ingestion cleaning pipeline."""

    text = normalize_newlines(text)
    text = remove_obvious_page_noise(text)
    text = collapse_excess_blank_lines(text)
    return text.strip()


def extract_markdown_title(text: str, fallback: str) -> str:
    """Extract the first Markdown heading as a document title."""

    match = HEADING_PATTERN.search(text)
    if not match:
        return fallback
    return match.group(0).lstrip("#").strip()


def _is_page_marker(line: str) -> bool:
    if not line:
        return False
    patterns = [
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^第\s*\d+\s*页$",
        r"^\d+\s*/\s*\d+$",
        r"^-\s*\d+\s*-$",
        r"^\d+$",
    ]
    return any(re.match(pattern, line, flags=re.IGNORECASE) for pattern in patterns)


def _find_repeated_short_lines(lines: list[str]) -> set[str]:
    candidates = [line.strip() for line in lines if 0 < len(line.strip()) <= 80]
    counts = Counter(candidates)
    return {
        line
        for line, count in counts.items()
        if count >= 3 and not line.startswith("#") and "|" not in line and not line.startswith("```")
    }


_CODE_BLOCKS: list[str] = []


def _protect_code_blocks(text: str) -> str:
    global _CODE_BLOCKS
    _CODE_BLOCKS = []

    def replace(match: re.Match[str]) -> str:
        _CODE_BLOCKS.append(match.group(0))
        return f"__CODE_BLOCK_{len(_CODE_BLOCKS) - 1}__"

    return CODE_FENCE_PATTERN.sub(replace, text)


def _restore_code_blocks(text: str) -> str:
    for index, block in enumerate(_CODE_BLOCKS):
        text = text.replace(f"__CODE_BLOCK_{index}__", block)
    return text

