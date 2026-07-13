from __future__ import annotations

from app.utils.text import tokenize


def token_overlap(left: str, right: str) -> float:
    left_terms = set(tokenize(left))
    right_terms = set(tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def answer_correctness(answer: str, ground_truth: str) -> float:
    return token_overlap(answer, ground_truth)


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    return token_overlap(ground_truth, "\n".join(contexts))


def faithfulness(answer: str, contexts: list[str]) -> float:
    return token_overlap(answer, "\n".join(contexts))


def context_precision(question: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    scores = [token_overlap(question, context) for context in contexts]
    return sum(scores) / len(scores)

