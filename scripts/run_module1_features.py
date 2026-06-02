"""Command-line runner for the full Module 1 feature pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_answers.model_answers import attach_model_answers
from module1.module1_features import build_module1_features
from preprocessing.preprocessing import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Module 1 understanding features for student answers."
    )
    parser.add_argument("input", type=Path, help="Path to .csv/.xlsx dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/module1_features.csv"),
        help="Output .csv or .xlsx path.",
    )
    parser.add_argument(
        "--model-answers-file",
        type=Path,
        help="Optional CSV/XLSX reference file with question_id and model_answer columns.",
    )
    parser.add_argument(
        "--model-answer-column",
        default="model_answer",
        help="Column containing the marking-schema/model answer.",
    )
    parser.add_argument(
        "--student-answer-column",
        default="synthetic_answer",
        help="Column containing student answers.",
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
        help="Maximum number of expected reference concepts per model answer.",
    )
    parser.add_argument(
        "--concept-reference",
        type=Path,
        default=Path("data/reference/concepts.csv"),
        help="CSV/XLSX file with expected concepts.",
    )
    parser.add_argument(
        "--concept-backend",
        choices=["weak-score", "trained-llm"],
        default="weak-score",
        help="Concept coverage backend.",
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
        help="Score column used by weak-score concept bootstrapping and summaries.",
    )
    parser.add_argument(
        "--similarity-backend",
        choices=["tfidf", "sentence-bert"],
        default="tfidf",
        help="Semantic similarity backend.",
    )
    parser.add_argument(
        "--language-check",
        choices=["none", "simple", "languagetool"],
        default="simple",
        help="Spelling and grammar check backend.",
    )
    parser.add_argument(
        "--apply-language-penalty",
        action="store_true",
        help="Add a small optional language_penalty value based on language quality.",
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
            require_complete=args.strict_model_answers,
        )

    output = build_module1_features(
        dataframe,
        model_answer_column=args.model_answer_column,
        student_answer_column=args.student_answer_column,
        question_id_column=args.question_id_column,
        max_concepts=args.max_concepts,
        require_model_answer=args.strict_model_answers,
        similarity_backend=args.similarity_backend,
        language_check_backend=args.language_check,
        apply_language_penalty=args.apply_language_penalty,
        concept_reference_path=args.concept_reference,
        concept_backend=args.concept_backend,
        concept_model_path=args.concept_model_path,
        target_score_column=args.target_score_column,
    )
    save_output(output, args.output)
    print(json.dumps(build_summary(output), indent=2, ensure_ascii=True))
    print(f"\nSaved Module 1 features to: {args.output}")


def save_output(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        dataframe.to_csv(path, index=False)
        return
    dataframe.to_excel(path, index=False)


def build_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataframe)),
        "average_semantic_similarity_score": round(
            float(dataframe["semantic_similarity_score"].mean()),
            4,
        ),
        "average_concept_coverage_ratio": round(
            float(dataframe["concepts_covered_ratio"].mean()),
            4,
        ),
        "average_language_quality_score": round(
            float(dataframe["language_quality_score"].mean()),
            4,
        ),
        "rows_with_language_errors": int((dataframe["language_error_count"] > 0).sum()),
        "rows_with_contradictions": int(dataframe["contradiction_detected"].sum()),
        "rows_cross_question_flagged": int(dataframe["cross_question_flag"].sum()),
    }


if __name__ == "__main__":
    main()
