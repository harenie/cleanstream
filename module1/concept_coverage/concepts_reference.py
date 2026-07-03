"""Expected-concept reference loading and generation helpers."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from module1.preprocessing.preprocessing import clean_text, load_dataset


DEFAULT_CONCEPT_REFERENCE_PATH = Path("data") / "reference" / "concepts.csv"
REQUIRED_CONCEPT_COLUMNS = {"question_id", "concept_id", "concept_text"}


def load_concepts(path: str | Path = DEFAULT_CONCEPT_REFERENCE_PATH) -> pd.DataFrame:
    """Load expected concepts from a CSV/XLSX reference file."""
    concept_path = Path(path)
    if not concept_path.exists():
        raise FileNotFoundError(
            f"Concept reference not found: {concept_path}. "
            "Create it with module1\\scripts\\build_concept_reference.py."
        )

    concepts = load_dataset(concept_path)
    concepts.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in concepts.columns
    ]
    missing = REQUIRED_CONCEPT_COLUMNS.difference(concepts.columns)
    if missing:
        raise ValueError(f"Concept reference is missing columns: {sorted(missing)}")

    output = concepts.copy()
    if "max_mark" not in output.columns:
        output["max_mark"] = 1.0
    output["question_id"] = output["question_id"].astype(str)
    output["concept_id"] = output["concept_id"].astype(str)
    output["concept_text"] = output["concept_text"].astype(str).str.strip()
    output["max_mark"] = pd.to_numeric(output["max_mark"], errors="coerce").fillna(1.0)
    output = output[output["concept_text"].apply(clean_text) != ""].copy()
    return output


def load_concepts_by_question(
    path: str | Path = DEFAULT_CONCEPT_REFERENCE_PATH,
) -> dict[str, list[dict[str, object]]]:
    """Return expected concepts keyed by question_id."""
    concepts = load_concepts(path)
    grouped: dict[str, list[dict[str, object]]] = {}
    for question_id, group in concepts.groupby("question_id", sort=False):
        grouped[str(question_id)] = [
            {
                "concept_id": str(row["concept_id"]),
                "concept_text": str(row["concept_text"]),
                "max_mark": float(row["max_mark"]),
                "concept_source": str(row.get("concept_source", "reference")),
                "concept_generator_backend": str(row.get("concept_generator_backend", "")),
                "concept_generator_model": str(row.get("concept_generator_model", "")),
            }
            for _, row in group.iterrows()
        ]
    return grouped


def build_concept_reference_from_model_answers(
    model_answers: pd.DataFrame,
    question_id_column: str = "question_id",
    answer_column: str = "model_answer",
    question_column: str = "question",
) -> pd.DataFrame:
    """Build expected concepts from marking-scheme bullet points, not keywords."""
    rows: list[dict[str, object]] = []
    for question_id, group in model_answers.groupby(question_id_column, sort=False):
        first = group.iloc[0]
        question = str(first.get(question_column, "")).strip()
        model_answer = str(first[answer_column])
        concept_texts = split_marking_scheme_concepts(model_answer)
        for index, concept_text in enumerate(concept_texts, start=1):
            rows.append(
                {
                    "question_id": str(question_id),
                    "concept_id": f"{question_id}_C{index}",
                    "question": question,
                    "concept_text": concept_text,
                    "max_mark": 1.0,
                }
            )
    return pd.DataFrame(rows)


def split_marking_scheme_concepts(model_answer: object) -> list[str]:
    """Split a model answer into expected concept statements."""
    raw_text = "" if model_answer is None else str(model_answer)
    bullet_lines = [
        normalize_concept_line(line)
        for line in raw_text.splitlines()
        if line.strip().startswith(("•", "-", "*"))
    ]
    concepts = [line for line in bullet_lines if clean_text(line)]
    if concepts:
        return concepts

    sentences = [
        normalize_concept_line(sentence)
        for sentence in re.split(r"[.!?]+", raw_text)
        if clean_text(sentence)
    ]
    return sentences


def normalize_concept_line(line: str) -> str:
    line = re.sub(r"^\s*[•*\-]\s*", "", line).strip()
    line = re.sub(r"\s+", " ", line)
    return line
