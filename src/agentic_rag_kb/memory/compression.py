"""Long conversation memory and context compression.

The memory layer keeps raw conversation turns while maintaining a structured
summary for the main graph. Compression is triggered before retrieval when chat
history, token estimate, or recently retrieved context becomes large enough to
pollute prompts or exceed model context limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


TARGET_RATIO_MIN = 0.55
TARGET_RATIO_MAX = 0.70


@dataclass(slots=True)
class StructuredSummary:
    """Structured long-term memory preserved after compression."""

    user_long_term_goal: str = ""
    current_task: str = ""
    confirmed_facts: list[str] = field(default_factory=list)
    pending_clarifications: list[str] = field(default_factory=list)
    important_entities: list[str] = field(default_factory=list)
    must_not_forget_constraints: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Render the summary into stable prompt-ready text."""

        return "\n".join(
            [
                "用户长期目标：" + (self.user_long_term_goal or "未明确"),
                "当前任务：" + (self.current_task or "未明确"),
                "不能遗忘的约束：",
                *_bullet_lines(self.must_not_forget_constraints),
                "已确认事实：",
                *_bullet_lines(self.confirmed_facts),
                "待澄清问题：",
                *_bullet_lines(self.pending_clarifications),
                "重要实体：",
                *_bullet_lines(self.important_entities),
            ]
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable summary."""

        return asdict(self)


@dataclass(slots=True)
class CompressionStats:
    """Compression ratio statistics."""

    original_token_estimate: int
    compressed_token_estimate: int
    compression_ratio: float
    target_min: float = TARGET_RATIO_MIN
    target_max: float = TARGET_RATIO_MAX
    status: str = "ok"

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable stats."""

        return asdict(self)


@dataclass(slots=True)
class CompressionResult:
    """Result returned by the compressor."""

    summary: StructuredSummary
    summary_text: str
    stats: CompressionStats
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable compression output."""

        return {
            "summary": self.summary.to_json_dict(),
            "summary_text": self.summary_text,
            "stats": self.stats.to_json_dict(),
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class ConversationMemory:
    """Conversation memory containing raw turns and a compressed summary."""

    turns: list[dict[str, str]] = field(default_factory=list)
    compression_summary: str = ""
    structured_summary: StructuredSummary = field(default_factory=StructuredSummary)
    compression_stats: CompressionStats | None = None

    def add_turn(self, role: str, content: str) -> None:
        """Append one raw conversation turn."""

        self.turns.append({"role": role, "content": content})

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable memory."""

        return {
            "turns": self.turns,
            "compression_summary": self.compression_summary,
            "structured_summary": self.structured_summary.to_json_dict(),
            "compression_stats": self.compression_stats.to_json_dict() if self.compression_stats else None,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ConversationMemory":
        """Build memory from graph state fields."""

        return cls(
            turns=[dict(turn) for turn in state.get("chat_history", [])],
            compression_summary=str(state.get("compression_summary", "")),
        )


@dataclass(slots=True)
class CompressionTrigger:
    """Dynamic threshold policy for deciding when to compress."""

    max_token_estimate: int = 1400
    max_turns: int = 8
    max_recent_context_tokens: int = 900
    token_margin_ratio: float = 0.85

    def should_compress(
        self,
        turns: list[dict[str, str]],
        compression_summary: str = "",
        recent_contexts: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Return whether compression is needed and why."""

        recent_contexts = recent_contexts or []
        history_tokens = estimate_turn_tokens(turns) + estimate_tokens(compression_summary)
        context_tokens = estimate_tokens(" ".join(str(context.get("text", "")) for context in recent_contexts))
        dynamic_token_threshold = max(200, int(self.max_token_estimate - context_tokens * self.token_margin_ratio))
        reasons = []
        if history_tokens >= dynamic_token_threshold:
            reasons.append("token_threshold")
        if len(turns) >= self.max_turns:
            reasons.append("turn_threshold")
        if context_tokens >= self.max_recent_context_tokens:
            reasons.append("recent_context_threshold")
        return bool(reasons), {
            "history_token_estimate": history_tokens,
            "recent_context_token_estimate": context_tokens,
            "dynamic_token_threshold": dynamic_token_threshold,
            "turn_count": len(turns),
            "reasons": reasons,
        }


class ConversationCompressor:
    """Compress raw conversation history into a structured durable summary."""

    def __init__(
        self,
        target_ratio_min: float = TARGET_RATIO_MIN,
        target_ratio_max: float = TARGET_RATIO_MAX,
    ) -> None:
        self.target_ratio_min = target_ratio_min
        self.target_ratio_max = target_ratio_max

    def compress(
        self,
        turns: list[dict[str, str]],
        existing_summary: str = "",
    ) -> CompressionResult:
        """Return structured compression result."""

        original_text = _conversation_text(turns, existing_summary)
        original_tokens = max(estimate_tokens(original_text), 1)
        summary = self._build_structured_summary(turns, existing_summary)
        summary_text = self._fit_summary_ratio(summary, original_tokens)
        compressed_tokens = max(estimate_tokens(summary_text), 1)
        ratio = round(compressed_tokens / original_tokens, 3)
        warnings: list[str] = []
        status = "ok"
        if ratio < self.target_ratio_min:
            status = "too_aggressive"
            warnings.append("compression_ratio_below_target")
        elif ratio > self.target_ratio_max:
            status = "too_verbose"
            warnings.append("compression_ratio_above_target")
        stats = CompressionStats(original_tokens, compressed_tokens, ratio, self.target_ratio_min, self.target_ratio_max, status)
        return CompressionResult(summary=summary, summary_text=summary_text, stats=stats, warnings=warnings)

    def _build_structured_summary(
        self,
        turns: list[dict[str, str]],
        existing_summary: str,
    ) -> StructuredSummary:
        user_messages = [turn.get("content", "") for turn in turns if turn.get("role") == "user"]
        assistant_messages = [turn.get("content", "") for turn in turns if turn.get("role") == "assistant"]
        all_text = "\n".join([existing_summary, *user_messages, *assistant_messages])
        return StructuredSummary(
            user_long_term_goal=_infer_long_term_goal(user_messages, existing_summary),
            current_task=_latest_nonempty(user_messages) or _latest_nonempty(assistant_messages),
            confirmed_facts=_extract_confirmed_facts(all_text),
            pending_clarifications=_extract_pending_clarifications(all_text),
            important_entities=_extract_entities(all_text),
            must_not_forget_constraints=_extract_constraints(all_text),
        )

    def _fit_summary_ratio(self, summary: StructuredSummary, original_tokens: int) -> str:
        target_tokens = max(1, int(original_tokens * ((self.target_ratio_min + self.target_ratio_max) / 2)))
        text = summary.to_text()
        if estimate_tokens(text) < int(original_tokens * self.target_ratio_min):
            text = _pad_summary_with_recent_details(text, summary, target_tokens)
        if estimate_tokens(text) > int(original_tokens * self.target_ratio_max):
            text = trim_to_token_estimate(text, int(original_tokens * self.target_ratio_max))
        return text


def estimate_turn_tokens(turns: list[dict[str, str]]) -> int:
    """Estimate tokens for chat turns."""

    return estimate_tokens(_conversation_text(turns))


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a deterministic mixed Chinese/English heuristic."""

    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_chars = len(re.sub(r"[\u4e00-\u9fff]", "", text))
    return max(1, int(cjk_chars * 0.8 + ascii_chars / 4))


def trim_to_token_estimate(text: str, max_tokens: int) -> str:
    """Trim text until its estimated token count is under `max_tokens`."""

    if estimate_tokens(text) <= max_tokens:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line])
        if estimate_tokens(candidate) > max_tokens:
            continue
        kept.append(line)
    return "\n".join(kept) or text[: max_tokens * 2]


def _conversation_text(turns: list[dict[str, str]], existing_summary: str = "") -> str:
    lines = [existing_summary] if existing_summary else []
    lines.extend(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in turns)
    return "\n".join(lines)


def _infer_long_term_goal(user_messages: list[str], existing_summary: str) -> str:
    for text in [existing_summary, *user_messages]:
        if "目标" in text or "我要" in text or "构建" in text or "复刻" in text:
            sentence = _first_sentence_containing(text, ["目标", "我要", "构建", "复刻"])
            if sentence:
                return sentence
    return "构建企业内部技术文档的 Agentic RAG 知识库问答系统"


def _latest_nonempty(messages: list[str]) -> str:
    for message in reversed(messages):
        stripped = message.strip()
        if stripped:
            return stripped[:300]
    return ""


def _extract_confirmed_facts(text: str) -> list[str]:
    markers = ["已确认", "已经", "使用", "采用", "支持", "实现", "包含", "模块"]
    return _extract_sentences(text, markers, limit=8)


def _extract_pending_clarifications(text: str) -> list[str]:
    markers = ["待澄清", "不明确", "需要确认", "？", "?"]
    return _extract_sentences(text, markers, limit=5)


def _extract_constraints(text: str) -> list[str]:
    markers = ["必须", "不能", "不得", "不要", "要求", "约束", "不要简化", "不降低复杂度"]
    constraints = _extract_sentences(text, markers, limit=10)
    if not constraints:
        constraints.append("回答必须基于给定上下文，不得编造引用。")
    return constraints


def _extract_entities(text: str) -> list[str]:
    known_entities = [
        "Agentic RAG",
        "LangGraph",
        "Send API",
        "Qdrant",
        "Ollama",
        "Gradio",
        "RAGAS",
        "Cross-Encoder",
        "Dense",
        "Sparse",
        "BM25",
        "FastAPI",
        "Human-in-the-loop",
        "Parent-Child",
    ]
    entities = [entity for entity in known_entities if entity.lower() in text.lower()]
    acronyms = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)
    for acronym in acronyms:
        if acronym not in entities:
            entities.append(acronym)
    return entities[:16]


def _extract_sentences(text: str, markers: list[str], limit: int) -> list[str]:
    sentences = re.split(r"[\n。；;]", text)
    result: list[str] = []
    for sentence in sentences:
        cleaned = sentence.strip(" -\t\r\n")
        if not cleaned:
            continue
        if any(marker in cleaned for marker in markers) and cleaned not in result:
            result.append(cleaned[:220])
        if len(result) >= limit:
            break
    return result


def _first_sentence_containing(text: str, markers: list[str]) -> str:
    for sentence in re.split(r"[\n。；;]", text):
        cleaned = sentence.strip(" -\t\r\n")
        if any(marker in cleaned for marker in markers):
            return cleaned[:220]
    return ""


def _pad_summary_with_recent_details(text: str, summary: StructuredSummary, target_tokens: int) -> str:
    details = []
    for field_name, values in [
        ("已确认事实补充", summary.confirmed_facts),
        ("约束补充", summary.must_not_forget_constraints),
        ("实体补充", summary.important_entities),
    ]:
        for value in values:
            details.append(f"{field_name}：{value}")
    output = text
    for detail in details:
        if estimate_tokens(output) >= target_tokens:
            break
        output = f"{output}\n{detail}"
    return output


def _bullet_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- 无"]
