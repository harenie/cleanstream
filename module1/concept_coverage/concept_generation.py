"""Generate and cache expected concept statements from model answers."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from module1.concept_coverage.concepts_reference import (
    build_concept_reference_from_model_answers,
    load_concepts,
)
from module1.preprocessing.preprocessing import clean_text


DEFAULT_GENERATED_CONCEPT_REFERENCE_PATH = (
    Path("module1") / "generated_outputs" / "generated_concepts.csv"
)
DEFAULT_CONCEPT_GENERATOR_MODEL_NAME = "google/flan-t5-small"


class FlanT5ConceptGenerator:
    """Instruction-tuned concept generator for marking-schema/model answers."""

    backend_name = "flan-t5"

    def __init__(
        self,
        model_name: str = DEFAULT_CONCEPT_GENERATOR_MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "FLAN-T5 concept generation requires torch and transformers. "
                "Install with: pip install -r requirements-llm.txt"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, question: object, model_answer: object) -> list[str]:
        prompt = build_concept_generation_prompt(question, model_answer)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        concepts = split_generated_concepts(generated_text)
        fallback = build_concept_reference_from_model_answers(
            pd.DataFrame(
                [
                    {
                        "question_id": "TEMP",
                        "question": str(question or ""),
                        "model_answer": str(model_answer or ""),
                    }
                ]
            )
        )
        fallback_concepts = fallback["concept_text"].astype(str).tolist()
        if concepts and len(concepts) >= min_required_concept_count(fallback_concepts):
            return concepts

        return fallback_concepts or concepts


def build_concept_generation_prompt(question: object, model_answer: object) -> str:
    return (
        "Extract concise grading concept statements from the marking scheme.\n"
        "Write one self-contained concept per line. Include every major bullet or claim.\n"
        "Do not copy irrelevant wording. Do not add concepts that are not supported by the marking scheme.\n\n"
        f"Question: {question or ''}\n\n"
        f"Marking scheme / model answer:\n{model_answer or ''}\n\n"
        "Concept statements:"
    )


def min_required_concept_count(fallback_concepts: list[str]) -> int:
    if len(fallback_concepts) <= 1:
        return 1
    return min(3, len(fallback_concepts))


def split_generated_concepts(text: object) -> list[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []
    parts: list[str] = []
    for line in raw_text.splitlines():
        line = re.sub(r"^\s*(concept\s*\d+[:.)-]?|\d+[:.)-]?|[-*•])\s*", "", line, flags=re.I)
        line = re.sub(r"\s+", " ", line).strip()
        if clean_text(line):
            parts.append(line)
    if len(parts) <= 1:
        sentence_parts = [
            re.sub(r"\s+", " ", sentence).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", raw_text)
            if clean_text(sentence)
        ]
        if len(sentence_parts) > len(parts):
            parts = sentence_parts

    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = clean_text(part)
        if key not in seen:
            unique.append(part)
            seen.add(key)
    return unique


def load_or_generate_concepts(
    model_answers: dict[object, str],
    questions: dict[str, str],
    output_path: str | Path = DEFAULT_GENERATED_CONCEPT_REFERENCE_PATH,
    generator_backend: str = "flan-t5",
    generator_model_name: str = DEFAULT_CONCEPT_GENERATOR_MODEL_NAME,
    regenerate: bool = False,
    fallback_reference_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load cached generated concepts and generate any missing question ids."""
    output_file = Path(output_path)
    cached = load_cached_generated_concepts(output_file) if output_file.exists() and not regenerate else pd.DataFrame()
    cached_question_ids = set(cached["question_id"].astype(str)) if not cached.empty else set()
    required_question_ids = {str(question_id) for question_id in model_answers}
    missing_question_ids = sorted(required_question_ids.difference(cached_question_ids))

    generated_rows: list[dict[str, object]] = []
    if missing_question_ids:
        if generator_backend.lower().replace("_", "-") != "flan-t5":
            raise ValueError("concept generator backend must be 'flan-t5'")
        generator = FlanT5ConceptGenerator(model_name=generator_model_name)
        for question_id in missing_question_ids:
            model_answer = model_answers[question_id]
            question = questions.get(question_id, "")
            generated = generator.generate(question, model_answer)
            for index, concept_text in enumerate(generated, start=1):
                generated_rows.append(
                    {
                        "question_id": question_id,
                        "concept_id": f"{question_id}_G{index}",
                        "question": question,
                        "concept_text": concept_text,
                        "max_mark": 1.0,
                        "concept_source": "generated",
                        "concept_generator_backend": generator.backend_name,
                        "concept_generator_model": generator_model_name,
                    }
                )

    if generated_rows:
        generated_df = pd.DataFrame(generated_rows)
        combined = pd.concat([cached, generated_df], ignore_index=True) if not cached.empty else generated_df
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_file, index=False)
        return combined

    if not cached.empty:
        return cached

    if fallback_reference_path is not None:
        return load_concepts(fallback_reference_path)

    raise ValueError("No generated concepts were available.")


def load_cached_generated_concepts(path: Path) -> pd.DataFrame:
    concepts = load_concepts(path)
    if "concept_source" not in concepts.columns:
        concepts["concept_source"] = "generated"
    if "concept_generator_backend" not in concepts.columns:
        concepts["concept_generator_backend"] = "unknown"
    if "concept_generator_model" not in concepts.columns:
        concepts["concept_generator_model"] = ""
    return concepts
