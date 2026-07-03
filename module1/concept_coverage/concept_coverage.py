"""Reference-concept coverage calculation with trainable LLM support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from module1.concept_coverage.concepts_reference import (
    DEFAULT_CONCEPT_REFERENCE_PATH,
    load_concepts_by_question,
)
from module1.concept_coverage.concept_generation import (
    DEFAULT_CONCEPT_GENERATOR_MODEL_NAME,
    DEFAULT_GENERATED_CONCEPT_REFERENCE_PATH,
    load_or_generate_concepts,
)
from module1.concept_coverage.llm_concept_coverage import (
    ConceptCoveragePredictor,
    NLIConceptCoveragePredictor,
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
    concept_source: str = "reference",
    generated_concepts_path: str | Path = DEFAULT_GENERATED_CONCEPT_REFERENCE_PATH,
    concept_generator_backend: str = "flan-t5",
    concept_generator_model_name: str = DEFAULT_CONCEPT_GENERATOR_MODEL_NAME,
    regenerate_concepts: bool = False,
    concept_backend: str = "auto",
    concept_model_path: str | Path | None = None,
    concept_nli_model_name: str | None = None,
    concept_nli_engine: object | None = None,
    target_score_column: str = "ai_score",
) -> pd.DataFrame:
    """Add model-vs-student concept coverage columns using expected concepts.

    Keyword extraction is intentionally not used here. Expected concepts come
    from generated model-answer concepts or a concept-reference fallback file.
    """
    processed = preprocess_dataframe(dataframe)
    model_answer_column = choose_model_answer_column(processed, model_answer_column)
    model_answers = infer_model_answers(
        processed,
        model_answer_column=model_answer_column,
        question_id_column=question_id_column,
        require_model_answer=require_model_answer,
    )
    questions_by_id = infer_questions_by_id(processed, question_id_column)
    concepts_by_question = build_concepts_by_question(
        concept_source=concept_source,
        processed=processed,
        model_answers=model_answers,
        questions_by_id=questions_by_id,
        concept_reference_path=concept_reference_path,
        generated_concepts_path=generated_concepts_path,
        concept_generator_backend=concept_generator_backend,
        concept_generator_model_name=concept_generator_model_name,
        regenerate_concepts=regenerate_concepts,
    )

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
        concept_nli_model_name=concept_nli_model_name,
        concept_nli_engine=concept_nli_engine,
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
    source_values: list[str] = []

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
        source_values.append("; ".join(sorted({str(item.get("concept_source", "reference")) for item in concepts})))

    output = drop_source_model_answer_columns(output, keep_column=model_answer_column)
    output["model_answer"] = model_answer_values
    output["missing_model_answer"] = missing_model_answer_values
    output["concept_backend"] = backend_values
    output["concept_source"] = source_values
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
    concept_nli_model_name: str | None,
    concept_nli_engine: object | None,
    target_score_column: str,
) -> ConceptCoveragePredictor | WeakScoreConceptCoveragePredictor | NLIConceptCoveragePredictor:
    normalized = concept_backend.lower().replace("_", "-")
    if normalized in {"weak-score", "weak"}:
        return WeakScoreConceptCoveragePredictor(target_score_column=target_score_column)
    if normalized in {"nli", "deberta", "deberta-mnli"}:
        return NLIConceptCoveragePredictor(
            model_name=concept_nli_model_name,
            nli_engine=concept_nli_engine,
        )
    if normalized in {"auto", "trained-llm", "llm", "transformer", "distilbert"}:
        resolved_model_path = (
            Path(concept_model_path)
            if concept_model_path is not None
            else Path("module1") / "models" / "concept_coverage_model"
        )
        try:
            return ConceptCoveragePredictor(resolved_model_path)
        except (FileNotFoundError, ImportError):
            if normalized == "auto":
                return WeakScoreConceptCoveragePredictor(target_score_column=target_score_column)
            raise
    raise ValueError("concept_backend must be 'auto', 'weak-score', 'trained-llm', or 'nli'")


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
        if prediction.get("source") == "nli":
            detail_parts[-1] += (
                f":E{float(prediction.get('entailment_score') or 0.0):.4f}"
                f"/N{float(prediction.get('neutral_score') or 0.0):.4f}"
                f"/C{float(prediction.get('contradiction_score') or 0.0):.4f}"
            )

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


def infer_questions_by_id(
    dataframe: pd.DataFrame,
    question_id_column: str,
    question_column: str = "question",
) -> dict[str, str]:
    if question_column not in dataframe.columns:
        return {}
    questions: dict[str, str] = {}
    for question_id, group in dataframe.groupby(question_id_column, dropna=False):
        values = [str(value).strip() for value in group[question_column].tolist() if clean_text(value)]
        questions[str(question_id)] = values[0] if values else ""
    return questions


def build_concepts_by_question(
    concept_source: str,
    processed: pd.DataFrame,
    model_answers: dict[object, str],
    questions_by_id: dict[str, str],
    concept_reference_path: str | Path,
    generated_concepts_path: str | Path,
    concept_generator_backend: str,
    concept_generator_model_name: str,
    regenerate_concepts: bool,
) -> dict[str, list[dict[str, object]]]:
    normalized = concept_source.lower().replace("_", "-")
    if normalized == "reference":
        return load_concepts_by_question(concept_reference_path)
    if normalized not in {"generated", "auto"}:
        raise ValueError("concept_source must be 'reference', 'generated', or 'auto'")

    try:
        generated = load_or_generate_concepts(
            model_answers={str(key): value for key, value in model_answers.items()},
            questions=questions_by_id,
            output_path=generated_concepts_path,
            generator_backend=concept_generator_backend,
            generator_model_name=concept_generator_model_name,
            regenerate=regenerate_concepts,
            fallback_reference_path=concept_reference_path if normalized == "auto" else None,
        )
        return concepts_dataframe_to_dict(generated)
    except (ImportError, OSError, ValueError):
        if normalized == "auto":
            return load_concepts_by_question(concept_reference_path)
        raise


def concepts_dataframe_to_dict(concepts: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    normalized = concepts.copy()
    if "max_mark" not in normalized.columns:
        normalized["max_mark"] = 1.0
    grouped: dict[str, list[dict[str, object]]] = {}
    for question_id, group in normalized.groupby("question_id", sort=False):
        grouped[str(question_id)] = [
            {
                "concept_id": str(row["concept_id"]),
                "concept_text": str(row["concept_text"]),
                "max_mark": float(row.get("max_mark", 1.0)),
                "concept_source": str(row.get("concept_source", "generated")),
                "concept_generator_backend": str(row.get("concept_generator_backend", "")),
                "concept_generator_model": str(row.get("concept_generator_model", "")),
            }
            for _, row in group.iterrows()
            if clean_text(row["concept_text"])
        ]
    return grouped


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
