"""Full Module 1 feature extraction pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from module1.concept_coverage.concept_coverage import add_concept_coverage_columns
from module1.concept_coverage.concept_generation import (
    DEFAULT_CONCEPT_GENERATOR_MODEL_NAME,
    DEFAULT_GENERATED_CONCEPT_REFERENCE_PATH,
)
from module1.concept_coverage.concepts_reference import DEFAULT_CONCEPT_REFERENCE_PATH
from module1.language_quality.language_quality import analyze_language_quality
from module1.nli.nli import DEFAULT_NLI_MODEL_NAME, NLIEngine
from module1.preprocessing.preprocessing import clean_text, preprocess_dataframe
from module1.reasoning.reasoning import (
    DEFAULT_QUESTION_REQUIREMENTS_PATH,
    assess_reasoning,
    build_reasoning_predictor,
    detect_contradictions,
    detect_noise,
    load_question_requirements,
    resolve_question_requirement,
)
from module1.semantic_similarity.semantic_similarity import (
    add_semantic_similarity_columns,
    build_similarity_engine,
)


def build_module1_features(
    dataframe: pd.DataFrame,
    model_answer_column: str = "model_answer",
    student_answer_column: str = "synthetic_answer",
    question_id_column: str = "question_id",
    max_concepts: int = 20,
    require_model_answer: bool = True,
    similarity_backend: str = "sentence-bert",
    cross_question_margin: float = 0.05,
    reasoning_backend: str = "nli",
    reasoning_model_path: str | Path | None = None,
    nli_model_name: str = DEFAULT_NLI_MODEL_NAME,
    contradiction_backend: str = "nli",
    question_requirements_path: str | Path = DEFAULT_QUESTION_REQUIREMENTS_PATH,
    language_check_backend: str = "simple",
    apply_language_penalty: bool = False,
    concept_reference_path: str | Path = DEFAULT_CONCEPT_REFERENCE_PATH,
    concept_source: str = "generated",
    generated_concepts_path: str | Path = DEFAULT_GENERATED_CONCEPT_REFERENCE_PATH,
    concept_generator_backend: str = "flan-t5",
    concept_generator_model_name: str = DEFAULT_CONCEPT_GENERATOR_MODEL_NAME,
    regenerate_concepts: bool = False,
    concept_backend: str = "nli",
    concept_model_path: str | Path | None = None,
    concept_nli_model_name: str | None = None,
    target_score_column: str = "ai_score",
) -> pd.DataFrame:
    """Build semantic, concept, reasoning, contradiction, and relevance features."""
    processed = preprocess_dataframe(dataframe)
    require_columns(processed, [question_id_column, student_answer_column])
    shared_nli_engine = build_shared_nli_engine(
        concept_backend=concept_backend,
        reasoning_backend=reasoning_backend,
        contradiction_backend=contradiction_backend,
        nli_model_name=concept_nli_model_name or nli_model_name,
    )

    output = add_concept_coverage_columns(
        processed,
        model_answer_column=model_answer_column,
        student_answer_column=student_answer_column,
        question_id_column=question_id_column,
        max_concepts=max_concepts,
        require_model_answer=require_model_answer,
        concept_reference_path=concept_reference_path,
        concept_source=concept_source,
        generated_concepts_path=generated_concepts_path,
        concept_generator_backend=concept_generator_backend,
        concept_generator_model_name=concept_generator_model_name,
        regenerate_concepts=regenerate_concepts,
        concept_backend=concept_backend,
        concept_model_path=concept_model_path,
        concept_nli_model_name=concept_nli_model_name or nli_model_name,
        concept_nli_engine=shared_nli_engine,
        target_score_column=target_score_column,
    )
    output = add_semantic_similarity_columns(
        output,
        model_answer_column="model_answer",
        student_answer_column="student_answer",
        question_id_column=question_id_column,
        require_model_answer=require_model_answer,
        similarity_backend=similarity_backend,
    )

    answer_column = "student_answer_clean" if "student_answer_clean" in output.columns else "student_answer"
    question_requirements = load_question_requirements(question_requirements_path)
    reasoning_predictor = build_reasoning_predictor(
        backend=reasoning_backend,
        model_path=reasoning_model_path,
        nli_model_name=nli_model_name,
        nli_engine=shared_nli_engine,
    )
    reasoning_values = []
    for _, row in output.iterrows():
        answer = row[answer_column]
        question = row["question"] if "question" in output.columns else ""
        question_id = row[question_id_column]
        requirement = resolve_question_requirement(
            question_id=question_id,
            question=question,
            requirements_by_question=question_requirements,
        )
        reasoning_values.append(
            assess_reasoning(
                answer,
                question,
                backend=reasoning_backend,
                predictor=reasoning_predictor,
                model_path=reasoning_model_path,
                nli_model_name=nli_model_name,
                nli_engine=shared_nli_engine,
                nli_support_score=(
                    float(row.get("concept_coverage_ratio", 0.0))
                    if concept_backend.lower().replace("_", "-") == "nli"
                    else None
                ),
                reasoning_required=bool(requirement["reasoning_required"]),
                reasoning_expected_type=str(requirement["reasoning_expected_type"]),
                reasoning_requirement_source=str(requirement["reasoning_requirement_source"]),
                reasoning_skip_reason=str(requirement["reasoning_skip_reason"]),
            )
        )
    contradiction_values = []
    for _, row in output.iterrows():
        contradiction_values.append(
            detect_contradictions(
                row[answer_column],
                row.get("question", ""),
                backend=contradiction_backend,
                nli_engine=shared_nli_engine,
                concepts=split_concepts(row.get("concepts")),
                model_answer=row.get("model_answer", ""),
            )
        )
    noise_values = [detect_noise(value) for value in output[answer_column]]
    language_values = [
        analyze_language_quality(
            value,
            backend=language_check_backend,
            apply_penalty=apply_language_penalty,
        )
        for value in output[answer_column]
    ]

    for key in [
        "reasoning_required",
        "reasoning_expected_type",
        "reasoning_requirement_source",
        "reasoning_skip_reason",
        "reasoning_quality",
        "reasoning_connective_count",
        "reasoning_connective_density",
        "reasoning_quality_score",
        "reasoning_backend",
        "reasoning_model_label",
        "reasoning_model_confidence",
        "reasoning_nli_label",
        "reasoning_nli_entailment_score",
        "reasoning_nli_neutral_score",
        "reasoning_nli_contradiction_score",
    ]:
        output[key] = [item[key] for item in reasoning_values]
    for key in [
        "contradiction_check_applied",
        "contradiction_question_scope",
        "contradiction_skip_reason",
        "contradiction_detected",
        "contradiction_detail",
        "contradiction_backend",
        "contradiction_score",
        "contradiction_source_concept",
        "contradiction_nli_entailment_score",
        "contradiction_nli_neutral_score",
        "contradiction_nli_contradiction_score",
    ]:
        output[key] = [item[key] for item in contradiction_values]
    for key in ["spelling_noise_count", "grammar_noise_count", "noise_detected"]:
        output[key] = [item[key] for item in noise_values]
    for key in [
        "language_check_backend",
        "spelling_error_count",
        "grammar_error_count",
        "language_error_count",
        "language_error_rate",
        "language_quality_score",
        "language_feedback",
        "language_penalty",
        "language_penalty_applied",
    ]:
        output[key] = [item[key] for item in language_values]

    output = add_cross_question_features(
        output,
        question_id_column=question_id_column,
        answer_column=answer_column,
        similarity_backend=similarity_backend,
        margin=cross_question_margin,
    )
    output["answer_word_count"] = [len(clean_text(value).split()) for value in output[answer_column]]
    return output


def build_shared_nli_engine(
    concept_backend: str,
    reasoning_backend: str,
    contradiction_backend: str,
    nli_model_name: str,
) -> NLIEngine | None:
    uses_nli = any(
        backend.lower().replace("_", "-") in {"nli", "deberta", "deberta-mnli"}
        for backend in [concept_backend, reasoning_backend, contradiction_backend]
    )
    return NLIEngine(model_name=nli_model_name) if uses_nli else None


def split_concepts(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("; ") if part.strip()]


def require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def add_cross_question_features(
    dataframe: pd.DataFrame,
    question_id_column: str,
    answer_column: str,
    similarity_backend: str,
    margin: float,
) -> pd.DataFrame:
    """Flag answers that match another question's model answer more strongly."""
    output = dataframe.copy()
    model_answers = {
        str(question_id): str(group["model_answer"].iloc[0])
        for question_id, group in output.groupby(question_id_column, dropna=True)
        if clean_text(group["model_answer"].iloc[0])
    }
    engine = build_similarity_engine(
        similarity_backend,
        list(model_answers.values()),
    )

    flags: list[bool] = []
    best_ids: list[str] = []
    best_scores: list[float] = []
    actual_scores: list[float] = []

    for _, row in output.iterrows():
        actual_question_id = str(row[question_id_column])
        answer = row[answer_column]
        if not model_answers:
            flags.append(False)
            best_ids.append("")
            best_scores.append(0.0)
            actual_scores.append(0.0)
            continue

        scores = {
            question_id: engine.similarity(answer, model_answer)
            for question_id, model_answer in model_answers.items()
        }
        best_question_id = max(scores, key=scores.get)
        actual_score = scores.get(actual_question_id, 0.0)
        best_score = scores[best_question_id]

        flags.append(best_question_id != actual_question_id and best_score >= actual_score + margin)
        best_ids.append(best_question_id)
        best_scores.append(best_score)
        actual_scores.append(actual_score)

    output["cross_question_flag"] = flags
    output["best_matching_question_id"] = best_ids
    output["best_cross_question_score"] = best_scores
    output["actual_question_similarity"] = actual_scores
    return output
