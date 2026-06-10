"""Prepare concept-answer training pairs for the transformer coverage model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1.concept_coverage.concepts_reference import load_concepts_by_question
from module1.concept_coverage.llm_concept_coverage import build_model_input, weak_label_from_score
from module1.model_answers.model_answers import attach_model_answers
from module1.preprocessing.preprocessing import load_dataset, preprocess_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create concept-level training rows from answers and expected concepts."
    )
    parser.add_argument("input", type=Path, help="Path to answer dataset.")
    parser.add_argument(
        "--concept-reference",
        type=Path,
        default=Path("data/reference/concepts.csv"),
        help="Concept reference CSV/XLSX path.",
    )
    parser.add_argument(
        "--model-answers-file",
        type=Path,
        default=Path("data/reference/model_answers.csv"),
        help="Optional model-answer reference to attach.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/training/concept_coverage_training.csv"))
    parser.add_argument("--question-id-column", default="question_id")
    parser.add_argument("--student-answer-column", default="synthetic_answer")
    parser.add_argument("--target-score-column", default="ai_score")
    parser.add_argument("--model-answer-column", default="model_answer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.input)
    if args.model_answers_file and args.model_answers_file.exists():
        dataframe = attach_model_answers(
            dataframe,
            args.model_answers_file,
            question_id_column=args.question_id_column,
            model_answer_column=args.model_answer_column,
            require_complete=True,
        )
    processed = preprocess_dataframe(dataframe)
    concepts_by_question = load_concepts_by_question(args.concept_reference)

    rows: list[dict[str, object]] = []
    answer_column = choose_answer_column(processed, args.student_answer_column)
    for _, row in processed.iterrows():
        question_id = str(row[args.question_id_column])
        concepts = concepts_by_question.get(question_id, [])
        label = weak_label_from_score(row.get(args.target_score_column))
        for concept in concepts:
            prompt = build_model_input(
                question=str(row.get("question", "")),
                student_answer=str(row[answer_column]),
                concept_text=str(concept["concept_text"]),
            )
            rows.append(
                {
                    "question_id": question_id,
                    "answer_id": row.get("answer_id", ""),
                    "concept_id": concept["concept_id"],
                    "question": row.get("question", ""),
                    "student_answer": row[answer_column],
                    "concept_text": concept["concept_text"],
                    "label": label,
                    "label_source": "weak_ai_score",
                    "target_score": row.get(args.target_score_column, ""),
                    "model_input": prompt,
                }
            )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": int(len(output)),
                "questions": int(output["question_id"].nunique()) if len(output) else 0,
                "label_counts": output["label"].value_counts().to_dict() if len(output) else {},
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def choose_answer_column(dataframe: pd.DataFrame, preferred_column: str) -> str:
    for column in [f"{preferred_column}_clean", preferred_column, "student_answer_clean", "student_answer"]:
        if column in dataframe.columns:
            return column
    raise ValueError(f"Missing student answer column. Tried {preferred_column} and student_answer.")


if __name__ == "__main__":
    main()
