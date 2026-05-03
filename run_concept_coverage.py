"""Command-line runner for concept keyword coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from concept_coverage import add_concept_coverage_columns
from preprocessing import (
    build_preprocessing_summary,
    clean_text,
    load_dataset,
    normalize_column_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add concept keyword, missing concept, and coverage columns."
    )
    parser.add_argument("input", type=Path, help="Path to preprocessed .csv/.xlsx dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("concept_coverage_output.xlsx"),
        help="Output .xlsx or .csv path.",
    )
    parser.add_argument(
        "--model-answer-column",
        default="model_answer",
        help="Column containing the marking-schema/model answer.",
    )
    parser.add_argument(
        "--model-answers-file",
        type=Path,
        help="Optional CSV/XLSX reference file with question_id and model_answer columns.",
    )
    parser.add_argument(
        "--student-answer-column",
        default="synthetic_answer",
        help="Column containing student answers. The current dataset uses synthetic_answer.",
    )
    parser.add_argument(
        "--question-id-column",
        default="question_id",
        help="Question id column used for grouping answers.",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=20,
        help="Maximum number of concept keywords per model answer.",
    )
    parser.add_argument(
        "--strict-model-answers",
        action="store_true",
        help="Fail when any question is missing a marking-schema/model answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.input)
    if args.model_answers_file:
        dataframe = attach_model_answers(
            dataframe,
            args.model_answers_file,
            question_id_column=args.question_id_column,
            model_answer_column=args.model_answer_column,
        )

    output = add_concept_coverage_columns(
        dataframe,
        model_answer_column=args.model_answer_column,
        student_answer_column=args.student_answer_column,
        question_id_column=args.question_id_column,
        max_concepts=args.max_concepts,
        require_model_answer=args.strict_model_answers,
    )

    save_output(output, args.output)
    summary = build_summary(output)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"\nSaved concept coverage output to: {args.output}")


def save_output(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        dataframe.to_csv(path, index=False)
        return
    dataframe.to_excel(path, index=False)


def attach_model_answers(
    dataframe: pd.DataFrame,
    model_answers_path: Path,
    question_id_column: str,
    model_answer_column: str,
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

    answer_map: dict[object, str] = {}
    for question_id, group in reference.groupby(question_id_column, dropna=False):
        answers = [
            clean_text(value)
            for value in group[model_answer_column].tolist()
            if clean_text(value)
        ]
        if answers:
            answer_map[question_id] = answers[0]

    output = main.copy()
    attached = output[question_id_column].map(answer_map)
    if model_answer_column in output.columns:
        output[model_answer_column] = attached.combine_first(output[model_answer_column])
    else:
        output[model_answer_column] = attached
    return output


def build_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    summary = build_preprocessing_summary(dataframe)
    summary["average_concepts_covered_ratio"] = round(
        float(dataframe["concepts_covered_ratio"].mean()),
        4,
    )
    if "missing_model_answer" in dataframe.columns:
        summary["rows_missing_model_answer"] = int(dataframe["missing_model_answer"].sum())
    return summary


if __name__ == "__main__":
    main()
