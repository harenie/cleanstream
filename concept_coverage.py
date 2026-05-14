"""Concept keyword extraction and coverage calculation."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from preprocessing.preprocessing import clean_text, preprocess_dataframe


STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "other",
    "should",
    "so",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "too",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def split_sentences(text: object) -> list[str]:
    """Split model-answer text into simple sentence units."""
    cleaned = clean_text(text)
    return [sentence.strip() for sentence in re.split(r"[.!?]+", cleaned) if sentence.strip()]


def tokenize(text: object) -> list[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", clean_text(text))


def extract_keywords_from_sentence(sentence: str, max_keywords: int = 6) -> list[str]:
    """Extract important keyword candidates from one sentence."""
    words = [
        word
        for word in tokenize(sentence)
        if len(word) > 2 and word not in STOPWORDS and not word.isdigit()
    ]
    counts = Counter(words)
    ordered_words = sorted(counts, key=lambda word: (-counts[word], words.index(word)))
    return ordered_words[:max_keywords]


def extract_concept_keywords(model_answer: object, max_concepts: int = 20) -> list[str]:
    """Extract concept keywords from model-answer sentences."""
    concepts: list[str] = []
    seen: set[str] = set()

    for sentence in split_sentences(model_answer):
        for keyword in extract_keywords_from_sentence(sentence):
            if keyword not in seen:
                concepts.append(keyword)
                seen.add(keyword)
            if len(concepts) >= max_concepts:
                return concepts

    return concepts


def compare_concepts_with_answer(
    concept_keywords: list[str],
    student_answer: object,
    similarity_engine: object | None = None,
    semantic_threshold: float = 0.12,
) -> dict[str, object]:
    """Calculate present concepts, missing concepts, and coverage ratio."""
    student_words = set(tokenize(student_answer))
    present: list[str] = []
    missing: list[str] = []

    for concept in concept_keywords:
        exact_match = concept in student_words
        semantic_match = False
        if similarity_engine is not None:
            semantic_match = (
                similarity_engine.similarity(concept, student_answer) >= semantic_threshold
            )
        if exact_match or semantic_match:
            present.append(concept)
        else:
            missing.append(concept)

    ratio = round(len(present) / len(concept_keywords), 4) if concept_keywords else 0.0

    return {
        "concepts_present": present,
        "concepts_missing": missing,
        "concept_coverage_ratio": ratio,
    }


def add_concept_coverage_columns(
    dataframe: pd.DataFrame,
    model_answer_column: str = "model_answer",
    student_answer_column: str = "synthetic_answer",
    question_id_column: str = "question_id",
    max_concepts: int = 20,
    require_model_answer: bool = False,
    similarity_engine: object | None = None,
) -> pd.DataFrame:
    """Add model-vs-student concept keyword and coverage columns."""
    processed = preprocess_dataframe(dataframe)
    model_answer_column = choose_model_answer_column(processed, model_answer_column)
    model_answers = infer_model_answers(
        processed,
        model_answer_column=model_answer_column,
        question_id_column=question_id_column,
        require_model_answer=require_model_answer,
    )

    output = processed.copy()
    answer_column = choose_student_answer_column(output, student_answer_column)

    if student_answer_column in output.columns:
        output = output.rename(columns={student_answer_column: "student_answer"})
    if f"{student_answer_column}_clean" in output.columns:
        output = output.rename(columns={f"{student_answer_column}_clean": "student_answer_clean"})

    concept_keywords_values: list[str] = []
    present_values: list[str] = []
    missing_values: list[str] = []
    ratio_values: list[float] = []
    model_answer_values: list[str] = []
    missing_model_answer_values: list[bool] = []

    if answer_column == student_answer_column:
        answer_column = "student_answer"
    elif answer_column == f"{student_answer_column}_clean":
        answer_column = "student_answer_clean"

    for _, row in output.iterrows():
        question_id = row[question_id_column]
        model_answer = model_answers[question_id]
        concepts = extract_concept_keywords(model_answer, max_concepts=max_concepts)
        comparison = compare_concepts_with_answer(
            concepts,
            row[answer_column],
            similarity_engine=similarity_engine,
        )

        model_answer_values.append(model_answer)
        missing_model_answer_values.append(not bool(model_answer))
        concept_keywords_values.append("; ".join(concepts))
        present_values.append("; ".join(comparison["concepts_present"]))
        missing_values.append("; ".join(comparison["concepts_missing"]))
        ratio_values.append(comparison["concept_coverage_ratio"])

    output = drop_source_model_answer_columns(output, keep_column=model_answer_column)
    output["model_answer"] = model_answer_values
    output["missing_model_answer"] = missing_model_answer_values
    output["concepts"] = concept_keywords_values
    output["concepts_present"] = present_values
    output["concepts_missing"] = missing_values
    output["concepts_covered_ratio"] = ratio_values
    output["concept_coverage_ratio"] = ratio_values
    return output


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
