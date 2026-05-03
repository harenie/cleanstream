"""Text and dataset preprocessing helpers for the synthetic answer dataset."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ANSWER_COLUMNS = ("student_answer", "synthetic_answer", "generated_answer", "answer")


def clean_text(value: object) -> str:
    """Clean one answer string for later NLP processing."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\[\[|\]\]", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[^a-z0-9.%$'\"\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase snake_case."""
    renamed = {
        column: (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        for column in dataframe.columns
    }
    renamed = {
        source: re.sub(r"_+", "_", target).strip("_")
        for source, target in renamed.items()
    }
    return dataframe.rename(columns=renamed)


def preprocess_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a preprocessed copy of the dataset."""
    processed = normalize_column_names(dataframe).copy()

    for column in ANSWER_COLUMNS:
        if column in processed.columns:
            processed[f"{column}_clean"] = processed[column].apply(clean_text)

    if "question" in processed.columns:
        processed["question_clean"] = processed["question"].apply(clean_text)

    if "chapter" in processed.columns:
        processed["chapter"] = processed["chapter"].astype(str).str.strip()

    if "difficulty" in processed.columns:
        processed["difficulty"] = processed["difficulty"].astype(str).str.strip().str.title()

    return processed


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load an Excel or CSV dataset file."""
    dataset_path = Path(path)
    if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    if dataset_path.suffix.lower() == ".csv":
        return pd.read_csv(dataset_path)
    raise ValueError("Dataset must be an .xlsx, .xls, or .csv file.")


def save_preprocessed_dataset(dataframe: pd.DataFrame, path: str | Path) -> None:
    """Save preprocessed data as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def build_preprocessing_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """Create a small summary for group review."""
    summary: dict[str, object] = {
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "missing_values": dataframe.isna().sum().to_dict(),
    }

    if "question_id" in dataframe.columns:
        summary["question_count"] = int(dataframe["question_id"].nunique())

    if "answer_id" in dataframe.columns:
        summary["answer_count"] = int(dataframe["answer_id"].nunique())

    return summary
