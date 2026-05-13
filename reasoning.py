"""Reasoning, contradiction, and answer-noise features."""

from __future__ import annotations

import re

from preprocessing import clean_text, tokenize_text


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


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def assess_reasoning(student_answer: object) -> dict[str, object]:
    """Estimate reasoning quality using explicit logical connectives."""
    text = clean_text(student_answer)
    connective_count = sum(1 for phrase in REASONING_CONNECTIVES if phrase in text)

    if connective_count >= 3:
        quality = "good"
    elif connective_count >= 1:
        quality = "partial"
    else:
        quality = "poor"

    return {
        "reasoning_quality": quality,
        "reasoning_connective_count": connective_count,
        "reasoning_quality_score": {"poor": 0.0, "partial": 0.5, "good": 1.0}[quality],
    }


def detect_contradictions(student_answer: object) -> dict[str, object]:
    """Detect simple wrong-logic patterns common in business/MIS answers."""
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
