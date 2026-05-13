"""Full Module 1 feature extraction pipeline."""

from __future__ import annotations

import pandas as pd

from concept_coverage import add_concept_coverage_columns
from preprocessing import clean_text, preprocess_dataframe
from reasoning import assess_reasoning, detect_contradictions, detect_noise
from semantic_similarity import add_semantic_similarity_columns, build_similarity_engine


def build_module1_features(
    dataframe: pd.DataFrame,
    model_answer_column: str = "model_answer",
    student_answer_column: str = "synthetic_answer",
    question_id_column: str = "question_id",
    max_concepts: int = 20,
    require_model_answer: bool = True,
    similarity_backend: str = "tfidf",
    cross_question_margin: float = 0.05,
) -> pd.DataFrame:
    """Build semantic, concept, reasoning, contradiction, and relevance features."""
    processed = preprocess_dataframe(dataframe)
    require_columns(processed, [question_id_column, student_answer_column])

    model_values = choose_model_corpus_values(processed, model_answer_column, question_id_column)
    answer_values = processed[student_answer_column].fillna("").astype(str).tolist()
    similarity_engine = build_similarity_engine(
        similarity_backend,
        [*model_values, *answer_values],
    )

    output = add_concept_coverage_columns(
        processed,
        model_answer_column=model_answer_column,
        student_answer_column=student_answer_column,
        question_id_column=question_id_column,
        max_concepts=max_concepts,
        require_model_answer=require_model_answer,
        similarity_engine=similarity_engine,
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
    reasoning_values = [assess_reasoning(value) for value in output[answer_column]]
    contradiction_values = [detect_contradictions(value) for value in output[answer_column]]
    noise_values = [detect_noise(value) for value in output[answer_column]]

    for key in ["reasoning_quality", "reasoning_connective_count", "reasoning_quality_score"]:
        output[key] = [item[key] for item in reasoning_values]
    for key in ["contradiction_detected", "contradiction_detail"]:
        output[key] = [item[key] for item in contradiction_values]
    for key in ["spelling_noise_count", "grammar_noise_count", "noise_detected"]:
        output[key] = [item[key] for item in noise_values]

    output = add_cross_question_features(
        output,
        question_id_column=question_id_column,
        answer_column=answer_column,
        similarity_backend=similarity_backend,
        margin=cross_question_margin,
    )
    output["answer_word_count"] = [len(clean_text(value).split()) for value in output[answer_column]]
    return output


def require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def choose_model_corpus_values(
    dataframe: pd.DataFrame,
    model_answer_column: str,
    question_id_column: str,
) -> list[str]:
    """Collect available model-answer text for fitting local similarity models."""
    for column in [
        model_answer_column,
        "model_answer",
        "scheme",
        "marking_schema_answer",
        "marking_scheme_answer",
        "answer_scheme",
        "rubric_answer",
    ]:
        if column in dataframe.columns and dataframe[column].apply(clean_text).any():
            return dataframe[column].fillna("").astype(str).tolist()
    raise ValueError(
        "No populated model-answer column found. Expected model_answer or scheme."
    )


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
        [*model_answers.values(), *output[answer_column].fillna("").astype(str).tolist()],
    )

    flags: list[bool] = []
    best_ids: list[str] = []
    best_scores: list[float] = []
    actual_scores: list[float] = []

    for _, row in output.iterrows():
        actual_question_id = str(row[question_id_column])
        answer = row[answer_column]
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
