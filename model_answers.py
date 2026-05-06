"""Helpers for attaching marking-schema model answers to datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing import clean_text, load_dataset, normalize_column_names


def attach_model_answers(
    dataframe: pd.DataFrame,
    model_answers_path: Path,
    question_id_column: str = "question_id",
    model_answer_column: str = "model_answer",
) -> pd.DataFrame:
    """Attach marking-schema/model answers from a reference file by question id."""
    main = normalize_column_names(dataframe)
    reference = normalize_column_names(load_dataset(model_answers_path))

    missing = [
        column
        for column in [question_id_column, model_answer_column]
        if column not in reference.columns
    ]
    if missing:
        raise ValueError(f"Model answer reference is missing columns: {missing}")
    if question_id_column not in main.columns:
        raise ValueError(f"Dataset is missing question id column: {question_id_column}")

    answer_map: dict[str, str] = {}
    for question_id, group in reference.groupby(question_id_column, dropna=False):
        answers = [
            str(value).strip()
            for value in group[model_answer_column].tolist()
            if clean_text(value)
        ]
        if answers:
            answer_map[str(question_id)] = answers[0]

    output = main.copy()
    attached = output[question_id_column].astype(str).map(answer_map)
    if model_answer_column in output.columns:
        output[model_answer_column] = attached.combine_first(output[model_answer_column])
    else:
        output[model_answer_column] = attached
    return output
