"""Command-line runner for the full Module 1 feature pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1.model_answers.model_answers import attach_model_answers
from module1.module1_features import build_module1_features
from module1.preprocessing.preprocessing import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Module 1 understanding features for student answers."
    )
    parser.add_argument("input", type=Path, help="Path to .csv/.xlsx dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("module1/generated_outputs/module1_features.csv"),
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
        help="Fallback CSV/XLSX file with expected concepts.",
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
        "--concept-generator-backend",
        choices=["flan-t5"],
        default="flan-t5",
        help="Concept generation backend.",
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
        choices=["auto", "weak-score", "trained-llm", "nli"],
        default="nli",
        help=(
            "Concept coverage backend. NLI uses DeBERTa-style entailment scoring; "
            "trained-llm uses the earlier DistilBERT classifier."
        ),
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
        help="Hugging Face MNLI/NLI model used by concept, reasoning, and contradiction checks.",
    )
    parser.add_argument(
        "--target-score-column",
        default="ai_score",
        help="Score column used by weak-score concept bootstrapping and summaries.",
    )
    parser.add_argument(
        "--similarity-backend",
        choices=["tfidf", "sentence-bert"],
        default="sentence-bert",
        help="Semantic similarity backend.",
    )
    parser.add_argument(
        "--reasoning-backend",
        choices=["auto", "rule-based", "trained-llm", "nli"],
        default="nli",
        help=(
            "Reasoning quality backend. NLI checks a reasoning hypothesis when "
            "the question requires reasoning."
        ),
    )
    parser.add_argument(
        "--reasoning-model-path",
        type=Path,
        default=Path("module1/models/concept_coverage_model"),
        help="Path to the DistilBERT model reused for reasoning quality.",
    )
    parser.add_argument(
        "--contradiction-backend",
        choices=["rule-based", "nli"],
        default="nli",
        help="Contradiction detection backend.",
    )
    parser.add_argument(
        "--question-requirements",
        type=Path,
        default=Path("data/reference/question_requirements.csv"),
        help="CSV/XLSX file that marks whether each question requires reasoning.",
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
        reasoning_backend=args.reasoning_backend,
        reasoning_model_path=args.reasoning_model_path,
        nli_model_name=args.nli_model,
        contradiction_backend=args.contradiction_backend,
        question_requirements_path=args.question_requirements,
        language_check_backend=args.language_check,
        apply_language_penalty=args.apply_language_penalty,
        concept_reference_path=args.concept_reference,
        concept_source=args.concept_source,
        generated_concepts_path=args.generated_concepts,
        concept_generator_backend=args.concept_generator_backend,
        concept_generator_model_name=args.concept_generator_model,
        regenerate_concepts=args.regenerate_concepts,
        concept_backend=args.concept_backend,
        concept_model_path=args.concept_model_path,
        concept_nli_model_name=args.nli_model,
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
        "rows_reasoning_required": int(dataframe["reasoning_required"].sum()),
        "rows_reasoning_not_required": int((~dataframe["reasoning_required"]).sum()),
        "rows_with_language_errors": int((dataframe["language_error_count"] > 0).sum()),
        "rows_with_contradictions": int(dataframe["contradiction_detected"].sum()),
        "average_contradiction_score": round(
            float(dataframe["contradiction_score"].mean()),
            4,
        ),
        "rows_cross_question_flagged": int(dataframe["cross_question_flag"].sum()),
    }


if __name__ == "__main__":
    main()
