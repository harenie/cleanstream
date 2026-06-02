"""Build expected concept reference rows from marking-scheme bullet points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concept_coverage.concepts_reference import build_concept_reference_from_model_answers
from preprocessing.preprocessing import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/reference/concepts.csv from model-answer bullet points."
    )
    parser.add_argument(
        "--model-answers",
        type=Path,
        default=Path("data/reference/model_answers.csv"),
        help="Reference file with question_id, question, and model_answer columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reference/concepts.csv"),
        help="Output concept reference CSV path.",
    )
    parser.add_argument("--question-id-column", default="question_id")
    parser.add_argument("--answer-column", default="model_answer")
    parser.add_argument("--question-column", default="question")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_answers = load_dataset(args.model_answers)
    concepts = build_concept_reference_from_model_answers(
        model_answers,
        question_id_column=args.question_id_column,
        answer_column=args.answer_column,
        question_column=args.question_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    concepts.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "questions": int(concepts["question_id"].nunique()),
                "concepts": int(len(concepts)),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
