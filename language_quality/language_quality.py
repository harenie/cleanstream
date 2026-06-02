"""Spelling and grammar quality checks for student answers."""

from __future__ import annotations

from functools import lru_cache
import re

from preprocessing.preprocessing import clean_text, tokenize_text
from reasoning.reasoning import COMMON_MISSPELLINGS


LANGUAGE_CHECK_BACKENDS = {"none", "simple", "languagetool"}


def analyze_language_quality(
    student_answer: object,
    backend: str = "simple",
    apply_penalty: bool = False,
    max_feedback_items: int = 3,
) -> dict[str, object]:
    """Return spelling, grammar, quality-score, and optional penalty features."""
    normalized_backend = backend.lower().replace("_", "-")
    if normalized_backend not in LANGUAGE_CHECK_BACKENDS:
        raise ValueError(
            "language check backend must be one of: "
            + ", ".join(sorted(LANGUAGE_CHECK_BACKENDS))
        )

    if normalized_backend == "none":
        result = empty_language_quality_result(normalized_backend)
    elif normalized_backend == "simple":
        result = analyze_simple_language_quality(student_answer, normalized_backend)
    else:
        result = analyze_languagetool_quality(
            student_answer,
            normalized_backend,
            max_feedback_items=max_feedback_items,
        )

    result["language_penalty"] = calculate_language_penalty(
        float(result["language_quality_score"]),
        enabled=apply_penalty,
    )
    result["language_penalty_applied"] = bool(apply_penalty)
    return result


def empty_language_quality_result(backend: str) -> dict[str, object]:
    return {
        "language_check_backend": backend,
        "spelling_error_count": 0,
        "grammar_error_count": 0,
        "language_error_count": 0,
        "language_error_rate": 0.0,
        "language_quality_score": 1.0,
        "language_feedback": "",
    }


def analyze_simple_language_quality(
    student_answer: object,
    backend: str,
) -> dict[str, object]:
    """Detect obvious spelling/noise patterns without external dependencies."""
    raw_text = "" if student_answer is None else str(student_answer)
    tokens = tokenize_text(raw_text)
    spelling_error_count = sum(1 for token in tokens if token in COMMON_MISSPELLINGS)
    repeated_punctuation_count = len(re.findall(r"([.!?,])\1+", raw_text))
    bracket_noise_count = raw_text.count("[[") + raw_text.count("]]")
    grammar_error_count = repeated_punctuation_count + bracket_noise_count
    return build_language_quality_result(
        backend=backend,
        student_answer=student_answer,
        spelling_error_count=spelling_error_count,
        grammar_error_count=grammar_error_count,
        feedback=[],
    )


def analyze_languagetool_quality(
    student_answer: object,
    backend: str,
    max_feedback_items: int,
) -> dict[str, object]:
    """Use language-tool-python when the optional dependency is installed."""
    raw_text = "" if student_answer is None else str(student_answer)
    tool = get_languagetool()
    matches = tool.check(raw_text)

    spelling_error_count = 0
    grammar_error_count = 0
    feedback: list[str] = []
    for match in matches:
        issue_type = str(getattr(match, "ruleIssueType", "")).lower()
        category = str(getattr(match, "category", "")).lower()
        if issue_type == "misspelling" or "typo" in category:
            spelling_error_count += 1
        else:
            grammar_error_count += 1

        message = str(getattr(match, "message", "")).strip()
        if message and len(feedback) < max_feedback_items:
            feedback.append(message)

    return build_language_quality_result(
        backend=backend,
        student_answer=student_answer,
        spelling_error_count=spelling_error_count,
        grammar_error_count=grammar_error_count,
        feedback=feedback,
    )


def build_language_quality_result(
    backend: str,
    student_answer: object,
    spelling_error_count: int,
    grammar_error_count: int,
    feedback: list[str],
) -> dict[str, object]:
    error_count = spelling_error_count + grammar_error_count
    word_count = max(1, len(tokenize_text(student_answer)))
    error_rate = round(error_count / word_count, 4)
    quality_score = max(0.0, 1.0 - min(error_rate * 5.0, 1.0))

    return {
        "language_check_backend": backend,
        "spelling_error_count": int(spelling_error_count),
        "grammar_error_count": int(grammar_error_count),
        "language_error_count": int(error_count),
        "language_error_rate": error_rate,
        "language_quality_score": round(quality_score, 4),
        "language_feedback": "; ".join(feedback),
    }


def calculate_language_penalty(
    language_quality_score: float,
    enabled: bool = False,
) -> float:
    """Return a small optional mark penalty out of 5."""
    if not enabled:
        return 0.0
    if language_quality_score >= 0.90:
        return 0.0
    if language_quality_score >= 0.75:
        return 0.10
    if language_quality_score >= 0.50:
        return 0.25
    return 0.50


@lru_cache(maxsize=1)
def get_languagetool() -> object:
    try:
        import language_tool_python
    except ImportError as exc:
        raise ImportError(
            "The 'languagetool' backend requires language-tool-python. "
            "Install it with: pip install language-tool-python"
        ) from exc

    return language_tool_python.LanguageTool("en-US")
