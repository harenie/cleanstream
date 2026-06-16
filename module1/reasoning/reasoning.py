"""Reasoning, contradiction, and answer-noise features."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Protocol

from module1.preprocessing.preprocessing import clean_text, tokenize_text


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

DEFAULT_REASONING_MODEL_PATH = Path("module1") / "models" / "concept_coverage_model"
REASONING_MODEL_CONCEPT_TEXT = (
    "The student answer provides clear reasoning by explaining why or how the "
    "claim is true, linking causes and effects, and supporting statements with "
    "logical connectors such as because, therefore, however, or as a result."
)
REASONING_LABEL_TO_QUALITY = {"missing": "poor", "partial": "partial", "covered": "good"}
REASONING_QUALITY_TO_SCORE = {"poor": 0.0, "partial": 0.5, "good": 1.0}


class ReasoningPromptPredictor(Protocol):
    backend_name: str

    def predict_prompt(self, prompt: str) -> tuple[str, float]:
        ...


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


def build_reasoning_predictor(
    backend: str = "auto",
    model_path: str | Path | None = None,
) -> ReasoningPromptPredictor | None:
    """Build the optional DistilBERT reasoning predictor."""
    normalized_backend = backend.lower().replace("_", "-")
    if normalized_backend in {"rule-based", "rules", "simple", "none"}:
        return None
    if normalized_backend not in {"auto", "trained-llm", "llm", "transformer", "distilbert"}:
        raise ValueError("reasoning backend must be 'auto', 'rule-based', or 'trained-llm'")

    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_REASONING_MODEL_PATH
    if normalized_backend == "auto" and not resolved_model_path.exists():
        return None

    try:
        from module1.concept_coverage.llm_concept_coverage import ConceptCoveragePredictor

        return ConceptCoveragePredictor(resolved_model_path)
    except (FileNotFoundError, ImportError):
        if normalized_backend == "auto":
            return None
        raise


def assess_reasoning(
    student_answer: object,
    question: object = "",
    backend: str = "rule-based",
    predictor: ReasoningPromptPredictor | None = None,
    model_path: str | Path | None = None,
) -> dict[str, object]:
    """Estimate reasoning quality with rules or the shared DistilBERT classifier."""
    marker_stats = calculate_reasoning_marker_stats(student_answer)
    normalized_backend = backend.lower().replace("_", "-")
    if normalized_backend in {"auto", "trained-llm", "llm", "transformer", "distilbert"}:
        if predictor is None:
            predictor = build_reasoning_predictor(normalized_backend, model_path)
        if predictor is not None:
            prompt = build_reasoning_model_input(question, student_answer)
            label, confidence = predictor.predict_prompt(prompt)
            quality = REASONING_LABEL_TO_QUALITY.get(label, "partial")
            return {
                **marker_stats,
                "reasoning_quality": quality,
                "reasoning_quality_score": REASONING_QUALITY_TO_SCORE[quality],
                "reasoning_backend": getattr(predictor, "backend_name", "trained-llm"),
                "reasoning_model_label": label,
                "reasoning_model_confidence": round(float(confidence), 4),
            }

    quality = rule_based_reasoning_quality(marker_stats["reasoning_connective_count"])
    return {
        **marker_stats,
        "reasoning_quality": quality,
        "reasoning_quality_score": REASONING_QUALITY_TO_SCORE[quality],
        "reasoning_backend": "rule-based",
        "reasoning_model_label": "",
        "reasoning_model_confidence": 0.0,
    }


def calculate_reasoning_marker_stats(student_answer: object) -> dict[str, object]:
    """Count explicit reasoning markers for explainability."""
    text = clean_text(student_answer)
    tokens = tokenize_text(text)
    connective_count = sum(
        len(pattern.findall(text))
        for pattern in REASONING_CONNECTIVE_PATTERNS
    )
    connective_density = round(connective_count / max(len(tokens), 1), 4)
    return {
        "reasoning_connective_count": connective_count,
        "reasoning_connective_density": connective_density,
    }


def rule_based_reasoning_quality(connective_count: object) -> str:
    """Map reasoning-marker count to the legacy quality label."""
    count = int(connective_count)
    if count >= 3:
        return "good"
    if count >= 1:
        return "partial"
    return "poor"


def build_reasoning_model_input(question: object, student_answer: object) -> str:
    """Reuse the concept classifier prompt format for reasoning quality."""
    return (
        "Question: "
        + str(question or "")
        + "\nStudent Answer: "
        + str(student_answer or "")
        + "\nExpected Concept: "
        + REASONING_MODEL_CONCEPT_TEXT
        + "\nTask: Classify whether the student answer covers the expected reasoning quality concept as missing, partial, or covered."
    )


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
