"""Reasoning, contradiction, and answer-noise features."""

from __future__ import annotations

import re

from preprocessing.preprocessing import clean_text, tokenize_text


REASONING_CONNECTIVES = [
    "because",
    "therefore",
    "as a result",
    "which means",
    "this means",
    "this leads to",
    "this causes",
    "this affects",
    "due to",
    "however",
    "although",
    "while",
    "thereby",
    "so",
]

REASONING_CONNECTIVE_PATTERNS = [
    re.compile(rf"\b{re.escape(phrase)}\b")
    for phrase in REASONING_CONNECTIVES
]

COMMON_MISSPELLINGS = {
    "internrt",
    "enternet",
    "busses",
    "busnisess",
    "bussenes",
    "plataform",
    "plataforms",
    "knowladge",
    "knouledge",
    "exprtise",
    "consummers",
    "relashionships",
    "magnament",
}

CONTRADICTION_TARGET_PATTERNS = [
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bmeaning of\b",
    r"\bconcept of\b",
    r"\bexplain the concept\b",
    r"\bdescribe the concept\b",
]

CONTRADICTION_EXCLUDED_PATTERNS = [
    r"\badvantages?\b",
    r"\bdisadvantages?\b",
    r"\bpros\b",
    r"\bcons\b",
    r"\bbenefits?\b",
    r"\bdrawbacks?\b",
    r"\bchallenges?\b",
    r"\blimitations?\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bcontrast\b",
]


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def assess_reasoning(student_answer: object) -> dict[str, object]:
    """Estimate reasoning quality using explicit logical connectives."""
    text = clean_text(student_answer)
    tokens = tokenize_text(text)
    connective_count = sum(
        len(pattern.findall(text))
        for pattern in REASONING_CONNECTIVE_PATTERNS
    )
    connective_density = round(connective_count / max(len(tokens), 1), 4)

    if connective_count >= 3:
        quality = "good"
    elif connective_count >= 1:
        quality = "partial"
    else:
        quality = "poor"

    return {
        "reasoning_quality": quality,
        "reasoning_connective_count": connective_count,
        "reasoning_connective_density": connective_density,
        "reasoning_quality_score": {"poor": 0.0, "partial": 0.5, "good": 1.0}[quality],
    }


def should_check_contradictions(question: object) -> dict[str, object]:
    """Return whether contradiction detection should run for this question type."""
    text = clean_text(question)
    if not text:
        return {
            "contradiction_check_applied": False,
            "contradiction_question_scope": "unknown",
            "contradiction_skip_reason": "No question text available.",
        }

    if any(re.search(pattern, text) for pattern in CONTRADICTION_EXCLUDED_PATTERNS):
        return {
            "contradiction_check_applied": False,
            "contradiction_question_scope": "excluded_balanced_answer",
            "contradiction_skip_reason": (
                "Skipped for advantages/disadvantages, pros/cons, comparison, "
                "or benefits/drawbacks style question."
            ),
        }

    if any(re.search(pattern, text) for pattern in CONTRADICTION_TARGET_PATTERNS):
        return {
            "contradiction_check_applied": True,
            "contradiction_question_scope": "target_definition_or_concept",
            "contradiction_skip_reason": "",
        }

    return {
        "contradiction_check_applied": False,
        "contradiction_question_scope": "non_target_question_type",
        "contradiction_skip_reason": (
            "Skipped because this is not a definition or concept-identification question."
        ),
    }


def detect_contradictions(student_answer: object, question: object = "") -> dict[str, object]:
    """Detect simple wrong-logic patterns only for targeted question types."""
    scope = should_check_contradictions(question)
    if not scope["contradiction_check_applied"]:
        return {
            **scope,
            "contradiction_detected": False,
            "contradiction_detail": "",
        }

    text = clean_text(student_answer)
    details: list[str] = []

    cost_higher_than_revenue = contains_any(
        text,
        [
            "cost is higher than revenue",
            "costs are higher than revenue",
            "cost exceeds revenue",
            "costs exceed revenue",
            "revenue is less than cost",
            "revenue lower than cost",
        ],
    )
    revenue_higher_than_cost = contains_any(
        text,
        [
            "revenue is higher than cost",
            "revenue exceeds cost",
            "revenue greater than cost",
            "cost is less than revenue",
            "cost lower than revenue",
        ],
    )
    profit_claim = contains_any(text, ["profit", "profitable", "made money"])
    loss_claim = contains_any(text, ["loss", "losing money", "not profitable"])

    if cost_higher_than_revenue and profit_claim:
        details.append("Claims costs are higher than revenue but concludes profit.")
    if revenue_higher_than_cost and loss_claim:
        details.append("Claims revenue is higher than cost but concludes loss.")
    if "information asymmetry" in text and "same information" in text and "seller knows more" in text:
        details.append("Conflicts while explaining information asymmetry.")
    if "disintermediation" in text and "more intermediaries" in text:
        details.append("Describes disintermediation as adding intermediaries.")

    return {
        **scope,
        "contradiction_detected": bool(details),
        "contradiction_detail": " ".join(details),
    }


def detect_noise(student_answer: object) -> dict[str, object]:
    """Count simple spelling/noise indicators without correcting the answer."""
    text = clean_text(student_answer)
    tokens = tokenize_text(text)
    misspelling_count = sum(1 for token in tokens if token in COMMON_MISSPELLINGS)
    repeated_punctuation_count = len(re.findall(r"([.!?,])\1+", str(student_answer)))
    bracket_noise_count = str(student_answer).count("[[") + str(student_answer).count("]]")

    return {
        "spelling_noise_count": misspelling_count,
        "grammar_noise_count": repeated_punctuation_count + bracket_noise_count,
        "noise_detected": bool(misspelling_count or repeated_punctuation_count or bracket_noise_count),
    }
