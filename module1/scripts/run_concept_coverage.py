"""Command-line runner for reference-concept coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1.concept_coverage.concept_coverage import add_concept_coverage_columns
from module1.model_answers.model_answers import attach_model_answers
from module1.preprocessing.preprocessing import build_preprocessing_summary, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add reference-concept missing/partial/covered coverage columns."
    )
    parser.add_argument("input", type=Path, help="Path to preprocessed .csv/.xlsx dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("module1/outputs/concept_coverage_output.xlsx"),
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
        help="Fallback CSV/XLSX file with question_id, concept_id, concept_text, and max_mark.",
    )
    parser.add_argument(
        "--concept-source",
        choices=["generated", "reference", "auto"],
        default="generated",
        help="Use FLAN-T5 generated concepts, manual reference concepts, or generated with fallback.",
    )
    parser.add_argument(
        "--generated-concepts",
        type=Path,
        default=Path("module1/generated_outputs/generated_concepts.csv"),
        help="Cache file for FLAN-T5 generated concept statements.",
    )
    parser.add_argument(
        "--concept-generator-model",
        default="google/flan-t5-small",
        help="Hugging Face model used to generate concept statements.",
    )
    parser.add_argument(
        "--regenerate-concepts",
        action="store_true",
        help="Regenerate concepts even when the generated concept cache exists.",
    )
    parser.add_argument(
        "--concept-backend",
        choices=["weak-score", "trained-llm", "nli"],
        default="nli",
        help="Concept coverage backend. NLI uses a DeBERTa-style entailment model.",
    )
    parser.add_argument(
        "--concept-model-path",
        type=Path,
        default=Path("module1/models/concept_coverage_model"),
        help="Path to trained concept coverage transformer model.",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/deberta-v3-base-mnli-fever-anli",
        help="Hugging Face MNLI/NLI model used by the nli concept backend.",
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
        concept_source=args.concept_source,
        generated_concepts_path=args.generated_concepts,
        concept_generator_model_name=args.concept_generator_model,
        regenerate_concepts=args.regenerate_concepts,
        concept_backend=args.concept_backend,
        concept_model_path=args.concept_model_path,
        concept_nli_model_name=args.nli_model,
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
