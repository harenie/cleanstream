"""LLM/transformer concept coverage prediction."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

from module1.preprocessing.preprocessing import clean_text
from module1.preprocessing.preprocessing import clean_text


LABEL_TO_ID = {"missing": 0, "partial": 1, "covered": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_TO_SCORE = {"missing": 0.0, "partial": 0.5, "covered": 1.0}
NLI_COVERED_THRESHOLD = 0.62
NLI_PARTIAL_THRESHOLD = 0.35
NLI_PARTIAL_OVERLAP_THRESHOLD = 0.35
NLI_CONTRADICTION_THRESHOLD = 0.60
NLI_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "can",
    "from",
    "in",
    "is",
    "of",
    "or",
    "the",
    "through",
    "to",
    "with",
}


class WeakScoreConceptCoveragePredictor:
    """Temporary bootstrap predictor based on whole-answer score bands.

    This is not the final concept model. It exists so the pipeline can run before
    manually corrected concept-level labels and a trained transformer are ready.
    """

    backend_name = "weak-score"

    def __init__(self, target_score_column: str = "ai_score") -> None:
        self.target_score_column = target_score_column

    def predict_row(
        self,
        row: pd.Series,
        concepts: list[dict[str, object]],
        answer_column: str,
        question_id_column: str,
        question_column: str,
    ) -> list[dict[str, object]]:
        label = weak_label_from_score(row.get(self.target_score_column))
        return [
            build_prediction(
                concept=concept,
                label=label,
                confidence=0.50,
                source=self.backend_name,
            )
            for concept in concepts
        ]


class ConceptCoveragePredictor:
    """Transformer classifier for missing/partial/covered concept labels."""

    backend_name = "trained-llm"

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_length = max_length
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained concept coverage model not found: {self.model_path}. "
                "Train it with module1\\scripts\\train_concept_model.py."
            )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The trained-llm concept backend requires torch and transformers. "
                "Install them with: pip install -r requirements-llm.txt"
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_row(
        self,
        row: pd.Series,
        concepts: list[dict[str, object]],
        answer_column: str,
        question_id_column: str,
        question_column: str,
    ) -> list[dict[str, object]]:
        """Predict labels for a single row's concepts.

        Implements a quick cleaned-literal match: if the cleaned concept text
        appears verbatim in the cleaned student answer, mark it `covered`
        without calling the transformer. Otherwise call the model per concept.
        """
        predictions: list[dict[str, object]] = []
        question = str(row.get(question_column, ""))
        student_answer = str(row.get(answer_column, ""))
        student_answer_clean = clean_text(student_answer)

        for concept in concepts:
            concept_text = str(concept["concept_text"])
            concept_text_clean = clean_text(concept_text)
            if concept_text_clean and concept_text_clean in student_answer_clean:
                label, confidence = "covered", 0.99
            else:
                prompt = build_model_input(
                    question=question,
                    student_answer=student_answer,
                    concept_text=concept_text,
                )
                label, confidence = self.predict_prompt(prompt)

            predictions.append(
                build_prediction(
                    concept=concept,
                    label=label,
                    confidence=confidence,
                    source=self.backend_name,
                )
            )
        return predictions

    def predict_prompt(self, prompt: str) -> tuple[str, float]:
        results = self.predict_prompts([prompt])
        return results[0]

    def predict_prompts(self, prompts: list[str]) -> list[tuple[str, float]]:
        if not prompts:
            return []
        encoded = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = self.torch.softmax(logits, dim=-1)
            confidences, label_ids = self.torch.max(probabilities, dim=-1)

        results = []
        for i in range(len(prompts)):
            label_id = int(label_ids[i].item())
            confidence = float(confidences[i].item())
            results.append((ID_TO_LABEL[label_id], confidence))
        return results


class NLIConceptCoveragePredictor:
    """NLI predictor for missing/partial/covered concept labels."""

    backend_name = "nli"

    def __init__(
        self,
        model_name: str | None = None,
        nli_engine: object | None = None,
    ) -> None:
        if nli_engine is None:
            from module1.nli.nli import DEFAULT_NLI_MODEL_NAME, NLIEngine

            nli_engine = NLIEngine(model_name=model_name or DEFAULT_NLI_MODEL_NAME)
        self.nli_engine = nli_engine

    def predict_row(
        self,
        row: pd.Series,
        concepts: list[dict[str, object]],
        answer_column: str,
        question_id_column: str,
        question_column: str,
    ) -> list[dict[str, object]]:
        student_answer = str(row.get(answer_column, ""))
        concept_texts = [str(concept["concept_text"]) for concept in concepts]
        results = self.nli_engine.predict_many(
            [student_answer] * len(concept_texts),
            concept_texts,
        )
        predictions: list[dict[str, object]] = []
        for concept, result in zip(concepts, results):
            concept_text = str(concept["concept_text"])
            label = nli_result_to_concept_label(result, student_answer, concept_text)
            predictions.append(
                build_prediction(
                    concept=concept,
                    label=label,
                    confidence=result.entailment,
                    source=self.backend_name,
                    nli_label=result.label,
                    entailment_score=result.entailment,
                    neutral_score=result.neutral,
                    contradiction_score=result.contradiction,
                )
            )
        return predictions


def build_model_input(question: str, student_answer: str, concept_text: str) -> str:
    return (
        "Question: "
        + question
        + "\nStudent Answer: "
        + student_answer
        + "\nExpected Concept: "
        + concept_text
        + "\nTask: Classify whether the student answer covers the expected concept as missing, partial, or covered."
    )


def build_prediction(
    concept: dict[str, object],
    label: str,
    confidence: float,
    source: str,
    nli_label: str = "",
    entailment_score: float | None = None,
    neutral_score: float | None = None,
    contradiction_score: float | None = None,
) -> dict[str, object]:
    return {
        "concept_id": str(concept["concept_id"]),
        "concept_text": str(concept["concept_text"]),
        "max_mark": float(concept.get("max_mark", 1.0)),
        "concept_source": str(concept.get("concept_source", "reference")),
        "concept_generator_backend": str(concept.get("concept_generator_backend", "")),
        "concept_generator_model": str(concept.get("concept_generator_model", "")),
        "label": label,
        "score": LABEL_TO_SCORE[label],
        "confidence": float(confidence),
        "source": source,
        "nli_label": nli_label,
        "entailment_score": entailment_score,
        "neutral_score": neutral_score,
        "contradiction_score": contradiction_score,
    }


def nli_result_to_concept_label(
    result: object,
    student_answer: object = "",
    concept_text: object = "",
) -> str:
    entailment = float(getattr(result, "entailment", 0.0))
    contradiction = float(getattr(result, "contradiction", 0.0))
    if entailment >= NLI_COVERED_THRESHOLD and contradiction < NLI_CONTRADICTION_THRESHOLD:
        return "covered"
    if entailment >= NLI_PARTIAL_THRESHOLD or concept_token_overlap(student_answer, concept_text) >= NLI_PARTIAL_OVERLAP_THRESHOLD:
        return "partial"
    return "missing"


def concept_token_overlap(student_answer: object, concept_text: object) -> float:
    answer_tokens = content_tokens(student_answer)
    concept_tokens = content_tokens(concept_text)
    if not answer_tokens or not concept_tokens:
        return 0.0
    return len(answer_tokens.intersection(concept_tokens)) / len(concept_tokens)


def content_tokens(value: object) -> set[str]:
    return {
        token
        for token in clean_text(value).split()
        if len(token) > 2 and token not in NLI_STOP_WORDS
    }


def weak_label_from_score(score_value: Any) -> str:
    try:
        score = float(score_value)
    except (TypeError, ValueError):
        return "partial"

    if score >= 4.0:
        return "covered"
    if score >= 2.0:
        return "partial"
    return "missing"


def save_label_config(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "label_config.json").write_text(
        json.dumps(
            {
                "label_to_id": LABEL_TO_ID,
                "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
                "label_to_score": LABEL_TO_SCORE,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
