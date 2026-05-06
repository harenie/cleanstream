"""Train the first answer-score baseline for cleanstream."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from concept_coverage import add_concept_coverage_columns
from model_answers import attach_model_answers
from preprocessing import clean_text, load_dataset
from semantic_similarity import add_semantic_similarity_columns


FEATURE_COLUMNS = [
    "semantic_similarity_score",
    "concepts_covered_ratio",
    "student_answer_length",
    "model_answer_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Ridge regression baseline to predict ai_score."
    )
    parser.add_argument("input", type=Path, help="Path to .csv/.xlsx dataset.")
    parser.add_argument(
        "--model-answers-file",
        type=Path,
        help="Optional CSV/XLSX reference file with question_id and model_answer columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_results"),
        help="Directory for split files, metrics, and trained model.",
    )
    parser.add_argument(
        "--model-answer-column",
        default="model_answer",
        help="Column containing the marking-schema/model answer.",
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
        "--target-column",
        default="ai_score",
        help="Numeric score column to predict.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
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

    prepared = prepare_training_dataframe(
        dataframe,
        model_answer_column=args.model_answer_column,
        student_answer_column=args.student_answer_column,
        question_id_column=args.question_id_column,
        target_column=args.target_column,
    )
    eligible = prepared[
        (~prepared["missing_model_answer"])
        & prepared[args.target_column].notna()
        & (prepared["model_answer"].apply(clean_text) != "")
    ].copy()
    if len(eligible) < 2:
        raise ValueError("Need at least two eligible rows with model answers and ai_score.")

    train_df, test_df, stratified = split_train_test(
        eligible,
        target_column=args.target_column,
        random_state=args.random_state,
    )

    model = Ridge(alpha=1.0)
    model.fit(train_df[FEATURE_COLUMNS], train_df[args.target_column])

    test_predictions = model.predict(test_df[FEATURE_COLUMNS])
    test_output = test_df.copy()
    test_output["predicted_score_raw"] = test_predictions
    test_output["predicted_score"] = test_output["predicted_score_raw"].clip(0.0, 5.0)
    test_output["absolute_error"] = (
        test_output[args.target_column] - test_output["predicted_score"]
    ).abs()

    metrics = build_metrics(
        actual=test_output[args.target_column],
        predicted=test_output["predicted_score"],
        train_count=len(train_df),
        test_count=len(test_df),
        missing_model_answer_count=int(prepared["missing_model_answer"].sum()),
    )
    split_summary = build_split_summary(
        prepared=prepared,
        eligible=eligible,
        train_df=train_df,
        test_df=test_df,
        target_column=args.target_column,
        stratified=stratified,
        random_state=args.random_state,
    )

    save_training_outputs(
        output_dir=args.output_dir,
        train_df=train_df,
        test_output=test_output,
        model=model,
        metrics=metrics,
        split_summary=split_summary,
        target_column=args.target_column,
    )

    print(json.dumps({"split_summary": split_summary, "training_metrics": metrics}, indent=2))
    print(f"\nSaved training outputs to: {args.output_dir}")


def prepare_training_dataframe(
    dataframe: pd.DataFrame,
    model_answer_column: str,
    student_answer_column: str,
    question_id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Build concept and semantic features for training."""
    concepts = add_concept_coverage_columns(
        dataframe,
        model_answer_column=model_answer_column,
        student_answer_column=student_answer_column,
        question_id_column=question_id_column,
    )
    prepared = add_semantic_similarity_columns(
        concepts,
        model_answer_column="model_answer",
        student_answer_column="student_answer",
        question_id_column=question_id_column,
    )
    if target_column not in prepared.columns:
        raise ValueError(f"Dataset is missing target column: {target_column}")

    prepared[target_column] = pd.to_numeric(prepared[target_column], errors="coerce")
    for column in FEATURE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)
    return prepared


def split_train_test(
    dataframe: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Split data into 2/3 train and 1/3 test, stratifying when possible."""
    target_counts = dataframe[target_column].value_counts()
    test_count = math.ceil(len(dataframe) / 3)
    train_count = len(dataframe) - test_count
    can_stratify = (
        target_counts.min() >= 2
        and len(target_counts) <= test_count
        and len(target_counts) <= train_count
    )
    stratify = dataframe[target_column] if can_stratify else None
    train_df, test_df = train_test_split(
        dataframe,
        test_size=1 / 3,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df.sort_index().copy(), test_df.sort_index().copy(), can_stratify


def build_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    train_count: int,
    test_count: int,
    missing_model_answer_count: int,
) -> dict[str, float | int]:
    """Build training metric summary."""
    return {
        "train_row_count": int(train_count),
        "test_row_count": int(test_count),
        "missing_model_answer_row_count": int(missing_model_answer_count),
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(actual, predicted))), 4),
        "r2_score": round(float(r2_score(actual, predicted)), 4),
        "within_0_5_marks_count": int(((actual - predicted).abs() <= 0.5).sum()),
        "within_0_5_marks_ratio": round(float(((actual - predicted).abs() <= 0.5).mean()), 4),
    }


def build_split_summary(
    prepared: pd.DataFrame,
    eligible: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    stratified: bool,
    random_state: int,
) -> dict[str, object]:
    """Build split details for audit and presentation."""
    return {
        "total_rows": int(len(prepared)),
        "eligible_rows": int(len(eligible)),
        "rows_missing_model_answer": int(prepared["missing_model_answer"].sum()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_ratio": round(float(len(train_df) / len(eligible)), 4),
        "test_ratio": round(float(len(test_df) / len(eligible)), 4),
        "target_column": target_column,
        "feature_columns": FEATURE_COLUMNS,
        "algorithm": "Ridge regression",
        "random_state": int(random_state),
        "stratified_by_target": bool(stratified),
        "train_score_distribution": score_distribution(train_df, target_column),
        "test_score_distribution": score_distribution(test_df, target_column),
    }


def score_distribution(dataframe: pd.DataFrame, target_column: str) -> dict[str, int]:
    """Return a stable score-count dictionary."""
    counts = dataframe[target_column].value_counts().sort_index()
    return {str(score): int(count) for score, count in counts.items()}


def save_training_outputs(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_output: pd.DataFrame,
    model: Ridge,
    metrics: dict[str, float | int],
    split_summary: dict[str, object],
    target_column: str,
) -> None:
    """Persist split files, metrics, and model artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_columns = existing_columns(
        train_df,
        [
            "id",
            "question_id",
            "answer_id",
            "student_answer",
            "model_answer",
            target_column,
            *FEATURE_COLUMNS,
        ],
    )
    train_df[output_columns].to_csv(output_dir / "train_split.csv", index=False)
    test_columns = existing_columns(
        test_output,
        [
            *output_columns,
            "predicted_score_raw",
            "predicted_score",
            "absolute_error",
        ],
    )
    test_output[test_columns].to_csv(output_dir / "test_predictions.csv", index=False)

    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": target_column,
            "split_summary": split_summary,
            "metrics": metrics,
        },
        output_dir / "score_model.joblib",
    )


def existing_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that exist in the dataframe, preserving requested order."""
    return [column for column in columns if column in dataframe.columns]


if __name__ == "__main__":
    main()
