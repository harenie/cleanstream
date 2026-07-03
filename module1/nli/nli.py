"""Shared NLI scoring for concept, reasoning, and contradiction checks."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_NLI_MODEL_NAME = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"


@dataclass(frozen=True)
class NLIResult:
    """Scores from a three-way NLI model."""

    label: str
    entailment: float
    neutral: float
    contradiction: float


class NLIEngine:
    """Batch NLI scorer using an MNLI-style entailment/neutral/contradiction model."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL_NAME,
        device: str | None = None,
        max_length: int = 384,
        batch_size: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "NLI scoring requires torch and transformers. "
                "Install with: pip install -r requirements-llm.txt"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.label_index = self._build_label_index()

    def predict(self, premise: object, hypothesis: object) -> NLIResult:
        return self.predict_many([premise], [hypothesis])[0]

    def predict_many(
        self,
        premises: list[object],
        hypotheses: list[object],
    ) -> list[NLIResult]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length.")
        if not premises:
            return []

        results: list[NLIResult] = []
        with self.torch.no_grad():
            for start in range(0, len(premises), self.batch_size):
                premise_batch = [str(value or "") for value in premises[start : start + self.batch_size]]
                hypothesis_batch = [
                    str(value or "") for value in hypotheses[start : start + self.batch_size]
                ]
                encoded = self.tokenizer(
                    premise_batch,
                    hypothesis_batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)
                for row in probabilities:
                    entailment = float(row[self.label_index["entailment"]].item())
                    neutral = float(row[self.label_index["neutral"]].item())
                    contradiction = float(row[self.label_index["contradiction"]].item())
                    label = max(
                        {
                            "entailment": entailment,
                            "neutral": neutral,
                            "contradiction": contradiction,
                        },
                        key=lambda key: {
                            "entailment": entailment,
                            "neutral": neutral,
                            "contradiction": contradiction,
                        }[key],
                    )
                    results.append(
                        NLIResult(
                            label=label,
                            entailment=round(entailment, 4),
                            neutral=round(neutral, 4),
                            contradiction=round(contradiction, 4),
                        )
                    )
        return results

    def _build_label_index(self) -> dict[str, int]:
        label2id = {
            str(label).lower(): int(index)
            for label, index in getattr(self.model.config, "label2id", {}).items()
        }
        mapped: dict[str, int] = {}
        for target in ["entailment", "neutral", "contradiction"]:
            for label, index in label2id.items():
                if target in label:
                    mapped[target] = index
                    break
        if set(mapped) == {"entailment", "neutral", "contradiction"}:
            return mapped

        # Common MNLI fallback order used by many Hugging Face classifiers.
        return {"contradiction": 0, "neutral": 1, "entailment": 2}
