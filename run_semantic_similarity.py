"""Command-line runner for semantic similarity features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model_answers import attach_model_answers
from preprocessing import load_dataset
from semantic_similarity import add_semantic_similarity_columns, build_semantic_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add semantic similarity between student answers and model answers."
    )
    parser.add_argument("input", type=Path, help="Path to .csv/.xlsx dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("semantic_similarity_output.xlsx"),
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
        help="Question id column used for attaching model answers.",
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

    output = add_semantic_similarity_columns(
        dataframe,
        model_answer_column=args.model_answer_column,
        student_answer_column=args.student_answer_column,
        question_id_column=args.question_id_column,
        require_model_answer=args.strict_model_answers,
    )
    save_output(output, args.output)
    print(json.dumps(build_semantic_summary(output), indent=2, ensure_ascii=True))
    print(f"\nSaved semantic similarity output to: {args.output}")


def save_output(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        dataframe.to_csv(path, index=False)
        return
    dataframe.to_excel(path, index=False)


if __name__ == "__main__":
    main()
