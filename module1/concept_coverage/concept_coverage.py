"""Reference-concept coverage calculation with trainable LLM support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from module1.concept_coverage.concepts_reference import (
    DEFAULT_CONCEPT_REFERENCE_PATH,
    load_concepts_by_question,
)
from module1.concept_coverage.llm_concept_coverage import (
    ConceptCoveragePredictor,
    WeakScoreConceptCoveragePredictor,
)
from module1.preprocessing.preprocessing import clean_text, preprocess_dataframe


def add_concept_coverage_columns(
    dataframe: pd.DataFrame,
    model_answer_column: str = "model_answer",
    student_answer_column: str = "synthetic_answer",
    question_id_column: str = "question_id",
    max_concepts: int = 20,
    require_model_answer: bool = False,
    concept_reference_path: str | Path = DEFAULT_CONCEPT_REFERENCE_PATH,
    concept_backend: str = "weak-score",
    concept_model_path: str | Path | None = None,
    target_score_column: str = "ai_score",
) -> pd.DataFrame:
    """Add model-vs-student concept coverage columns using expected concepts.

    Keyword extraction is intentionally not used here. Expected concepts must come
    from ``data/reference/concepts.csv`` or another concept-reference file.
    """
    processed = preprocess_dataframe(dataframe)
    model_answer_column = choose_model_answer_column(processed, model_answer_column)
    model_answers = infer_model_answers(
        processed,
        model_answer_column=model_answer_column,
        question_id_column=question_id_column,
        require_model_answer=require_model_answer,
    )
    concepts_by_question = load_concepts_by_question(concept_reference_path)

    output = processed.copy()
    answer_column = choose_student_answer_column(output, student_answer_column)
    if student_answer_column in output.columns:
        output = output.rename(columns={student_answer_column: "student_answer"})
    if f"{student_answer_column}_clean" in output.columns:
        output = output.rename(columns={f"{student_answer_column}_clean": "student_answer_clean"})

    if answer_column == student_answer_column:
        answer_column = "student_answer"
    elif answer_column == f"{student_answer_column}_clean":
        answer_column = "student_answer_clean"

    predictor = build_concept_predictor(
        concept_backend=concept_backend,
        concept_model_path=concept_model_path,
        target_score_column=target_score_column,
    )

    concept_values: list[str] = []
    present_values: list[str] = []
    partial_values: list[str] = []
    missing_values: list[str] = []
    ratio_values: list[float] = []
    model_answer_values: list[str] = []
    missing_model_answer_values: list[bool] = []
    detail_values: list[str] = []
    backend_values: list[str] = []

    for _, row in output.iterrows():
        question_id = str(row[question_id_column])
        model_answer = model_answers[row[question_id_column]]
        concepts = concepts_by_question.get(question_id, [])[:max_concepts]
        if not concepts:
            raise ValueError(
                f"No expected concepts found for question_id={question_id}. "
                f"Update {concept_reference_path} before calculating coverage."
            )

        predictions = predictor.predict_row(
            row=row,
            concepts=concepts,
            answer_column=answer_column,
            question_id_column=question_id_column,
            question_column="question",
        )
        summary = summarize_concept_predictions(predictions)

        model_answer_values.append(model_answer)
        missing_model_answer_values.append(not bool(clean_text(model_answer)))
        concept_values.append("; ".join(item["concept_text"] for item in concepts))
        present_values.append("; ".join(summary["present"]))
        partial_values.append("; ".join(summary["partial"]))
        missing_values.append("; ".join(summary["missing"]))
        ratio_values.append(summary["coverage_ratio"])
        detail_values.append(summary["details"])
        backend_values.append(predictor.backend_name)

    output = drop_source_model_answer_columns(output, keep_column=model_answer_column)
    output["model_answer"] = model_answer_values
    output["missing_model_answer"] = missing_model_answer_values
    output["concept_backend"] = backend_values
    output["concepts"] = concept_values
    output["concepts_present"] = present_values
    output["concepts_partial"] = partial_values
    output["concepts_missing"] = missing_values
    output["concept_prediction_details"] = detail_values
    output["concepts_covered_ratio"] = ratio_values
    output["concept_coverage_ratio"] = ratio_values
    return output


def build_concept_predictor(
    concept_backend: str,
    concept_model_path: str | Path | None,
    target_score_column: str,
) -> ConceptCoveragePredictor | WeakScoreConceptCoveragePredictor:
    normalized = concept_backend.lower().replace("_", "-")
    if normalized in {"weak-score", "weak"}:
        return WeakScoreConceptCoveragePredictor(target_score_column=target_score_column)
    if normalized in {"trained-llm", "llm", "transformer"}:
        if concept_model_path is None:
            concept_model_path = Path("module1") / "models" / "concept_coverage_model"
        return ConceptCoveragePredictor(concept_model_path)
    raise ValueError("concept_backend must be 'weak-score' or 'trained-llm'")


def summarize_concept_predictions(predictions: list[dict[str, object]]) -> dict[str, object]:
    present: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    weighted_score = 0.0
    total_weight = 0.0
    detail_parts: list[str] = []

    for prediction in predictions:
        concept_text = str(prediction["concept_text"])
        label = str(prediction["label"])
        score = float(prediction["score"])
        max_mark = float(prediction.get("max_mark", 1.0))
        confidence = float(prediction.get("confidence", 0.0))

        if label == "covered":
            present.append(concept_text)
        elif label == "partial":
            partial.append(concept_text)
        else:
            missing.append(concept_text)

        weighted_score += score * max_mark
        total_weight += max_mark
        detail_parts.append(f"{concept_text}={label}:{confidence:.4f}")

    coverage_ratio = round(weighted_score / total_weight, 4) if total_weight else 0.0
    return {
        "present": present,
        "partial": partial,
        "missing": missing,
        "coverage_ratio": coverage_ratio,
        "details": "; ".join(detail_parts),
    }


def infer_model_answers(
    dataframe: pd.DataFrame,
    model_answer_column: str,
    question_id_column: str,
    require_model_answer: bool,
) -> dict[object, str]:
    """Use the marking-schema/model answer for each question."""
    missing = [
        column
        for column in [question_id_column, model_answer_column]
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model_answers: dict[object, str] = {}
    for question_id, group in dataframe.groupby(question_id_column, dropna=False):
        answers = [
            str(value).strip()
            for value in group[model_answer_column].tolist()
            if clean_text(value)
        ]
        if not answers:
            if require_model_answer:
                raise ValueError(
                    f"No marking-schema/model answer found for question_id {question_id!r} "
                    f"in column {model_answer_column!r}."
                )
            model_answers[question_id] = ""
        else:
            model_answers[question_id] = answers[0]

    return model_answers


def choose_model_answer_column(dataframe: pd.DataFrame, preferred_column: str) -> str:
    """Choose the marking-schema/model-answer column, never generated answers by default."""
    candidates = [
        preferred_column,
        "model_answer",
        "marking_schema_answer",
        "marking_scheme_answer",
        "scheme",
        "answer_scheme",
        "rubric_answer",
    ]
    for column in candidates:
        if column in dataframe.columns and dataframe[column].apply(clean_text).any():
            return column
    raise ValueError(
        "No populated marking-schema/model-answer column found. "
        f"Tried: {candidates}. The generated_answer column is not used for concept coverage."
    )


def choose_student_answer_column(dataframe: pd.DataFrame, preferred_column: str) -> str:
    """Prefer cleaned student-answer text when available."""
    candidates = [preferred_column, "student_answer", "synthetic_answer", "answer"]
    for column in candidates:
        clean_column = f"{column}_clean"
        if clean_column in dataframe.columns:
            return clean_column
        if column in dataframe.columns:
            return column
    raise ValueError(f"Missing student answer column. Tried: {candidates}")


def drop_source_model_answer_columns(dataframe: pd.DataFrame, keep_column: str) -> pd.DataFrame:
    """Remove raw generated-answer and schema-source columns from the final output."""
    columns_to_drop = {
        "generated_answer",
        "generated_answer_clean",
        "scheme",
        "scheme_clean",
        "model_answer_clean",
        "marking_schema_answer",
        "marking_schema_answer_clean",
        "marking_scheme_answer",
        "marking_scheme_answer_clean",
        "answer_scheme",
        "answer_scheme_clean",
        "rubric_answer",
        "rubric_answer_clean",
    }
    return dataframe.drop(columns=[col for col in columns_to_drop if col in dataframe.columns])
