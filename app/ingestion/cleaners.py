from __future__ import annotations

import re

from app.utils.text import normalize_whitespace


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s*/\s*\d+\s*\n", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return normalize_whitespace(text)

