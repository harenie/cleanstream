"""Command-line runner for reference-concept coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concept_coverage.concept_coverage import add_concept_coverage_columns
from model_answers.model_answers import attach_model_answers
from preprocessing.preprocessing import build_preprocessing_summary, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add reference-concept missing/partial/covered coverage columns."
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
        help="Maximum number of expected concepts per model answer.",
    )
    parser.add_argument(
        "--concept-reference",
        type=Path,
        default=Path("data/reference/concepts.csv"),
        help="CSV/XLSX file with question_id, concept_id, concept_text, and max_mark.",
    )
    parser.add_argument(
        "--concept-backend",
        choices=["weak-score", "trained-llm"],
        default="weak-score",
        help="Concept coverage backend. Use trained-llm after training the transformer.",
    )
    parser.add_argument(
        "--concept-model-path",
        type=Path,
        default=Path("models/concept_coverage_model"),
        help="Path to trained concept coverage transformer model.",
    )
    parser.add_argument(
        "--target-score-column",
        default="ai_score",
        help="Score column used only by the weak-score bootstrap backend.",
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
        concept_reference_path=args.concept_reference,
        concept_backend=args.concept_backend,
        concept_model_path=args.concept_model_path,
        target_score_column=args.target_score_column,
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
    if "missing_model_answer" in dataframe.columns:
        summary["rows_missing_model_answer"] = int(dataframe["missing_model_answer"].sum())
    return summary


if __name__ == "__main__":
    main()
