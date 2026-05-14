"""Helpers for attaching marking-schema model answers to datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing.preprocessing import clean_text, load_dataset, normalize_column_names


MODEL_ANSWER_CANDIDATES = [
    "model_answer",
    "scheme",
    "marking_schema_answer",
    "marking_scheme_answer",
    "answer_scheme",
    "rubric_answer",
]


def attach_model_answers(
    dataframe: pd.DataFrame,
    model_answers_path: Path,
    question_id_column: str = "question_id",
    model_answer_column: str = "model_answer",
    require_complete: bool = False,
) -> pd.DataFrame:
    """Attach marking-schema/model answers from a reference file by question id.

    The reference may use either ``model_answer`` or the dataset's ``scheme`` column.
    The attached output column is always named by ``model_answer_column``.
    """
    main = normalize_column_names(dataframe)
    reference = normalize_column_names(load_dataset(model_answers_path))

    if question_id_column not in reference.columns:
        raise ValueError(f"Model answer reference is missing column: {question_id_column}")
    if question_id_column not in main.columns:
        raise ValueError(f"Dataset is missing question id column: {question_id_column}")

    reference_answer_column = choose_reference_answer_column(reference, model_answer_column)
    answer_map: dict[str, str] = {}
    for question_id, group in reference.groupby(question_id_column, dropna=False):
        answers = [
            str(value).strip()
            for value in group[reference_answer_column].tolist()
            if clean_text(value)
        ]
        if answers:
            answer_map[str(question_id)] = answers[0]

    output = main.copy()
    attached = output[question_id_column].astype(str).map(answer_map)
    if require_complete:
        missing_ids = sorted(
            output.loc[attached.apply(clean_text) == "", question_id_column]
            .astype(str)
            .unique()
            .tolist()
        )
        if missing_ids:
            raise ValueError(
                "Missing model answers for question_id values: "
                + ", ".join(missing_ids)
            )

    if model_answer_column in output.columns:
        output[model_answer_column] = attached.combine_first(output[model_answer_column])
    else:
        output[model_answer_column] = attached
    return output


def choose_reference_answer_column(
    reference: pd.DataFrame,
    preferred_column: str = "model_answer",
) -> str:
    """Choose the populated answer column from a normalized reference dataframe."""
    candidates = [preferred_column, *MODEL_ANSWER_CANDIDATES]
    seen: set[str] = set()
    for column in candidates:
        if column in seen:
            continue
        seen.add(column)
        if column in reference.columns and reference[column].apply(clean_text).any():
            return column
    raise ValueError(
        "Model answer reference needs a populated answer column. "
        f"Tried: {list(seen)}"
    )
