"""Command-line runner for concept keyword coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from concept_coverage import add_concept_coverage_columns
from preprocessing import build_preprocessing_summary, load_dataset


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
        default="generated_answer",
        help="Column containing candidate model answers.",
    )
    parser.add_argument(
        "--student-answer-column",
        default="synthetic_answer",
        help="Column containing student answers.",
    )
    parser.add_argument(
        "--score-column",
        default="ai_score",
        help="Score column used to choose the best model answer per question.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.input)
    output = add_concept_coverage_columns(
        dataframe,
        model_answer_column=args.model_answer_column,
        student_answer_column=args.student_answer_column,
        score_column=args.score_column,
        question_id_column=args.question_id_column,
        max_concepts=args.max_concepts,
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


def build_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    summary = build_preprocessing_summary(dataframe)
    summary["average_concepts_covered_ratio"] = round(
        float(dataframe["concepts_covered_ratio"].mean()),
        4,
    )
    return summary


if __name__ == "__main__":
    main()
