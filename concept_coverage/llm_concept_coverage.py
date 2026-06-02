"""LLM/transformer concept coverage prediction."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd


LABEL_TO_ID = {"missing": 0, "partial": 1, "covered": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_TO_SCORE = {"missing": 0.0, "partial": 0.5, "covered": 1.0}


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
                "Train it with scripts\\train_concept_model.py."
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
        predictions: list[dict[str, object]] = []
        question = str(row.get(question_column, ""))
        student_answer = str(row.get(answer_column, ""))
        for concept in concepts:
            prompt = build_model_input(
                question=question,
                student_answer=student_answer,
                concept_text=str(concept["concept_text"]),
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
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1)
            confidence, label_id = self.torch.max(probabilities, dim=-1)

        return ID_TO_LABEL[int(label_id.item())], float(confidence.item())


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
) -> dict[str, object]:
    return {
        "concept_id": str(concept["concept_id"]),
        "concept_text": str(concept["concept_text"]),
        "max_mark": float(concept.get("max_mark", 1.0)),
        "label": label,
        "score": LABEL_TO_SCORE[label],
        "confidence": float(confidence),
        "source": source,
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
