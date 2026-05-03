"""Concept keyword extraction and coverage calculation."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from preprocessing import clean_text, preprocess_dataframe


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
) -> dict[str, object]:
    """Calculate present concepts, missing concepts, and coverage ratio."""
    student_words = set(tokenize(student_answer))
    present = [concept for concept in concept_keywords if concept in student_words]
    missing = [concept for concept in concept_keywords if concept not in student_words]
    ratio = round(len(present) / len(concept_keywords), 4) if concept_keywords else 0.0

    return {
        "concepts_present": present,
        "concepts_missing": missing,
        "concept_coverage_ratio": ratio,
    }


def add_concept_coverage_columns(
    dataframe: pd.DataFrame,
    model_answer_column: str = "generated_answer",
    student_answer_column: str = "synthetic_answer",
    score_column: str = "ai_score",
    question_id_column: str = "question_id",
    max_concepts: int = 20,
) -> pd.DataFrame:
    """Add concept keyword and coverage columns to a preprocessed dataset."""
    processed = preprocess_dataframe(dataframe)
    model_answers = infer_model_answers(
        processed,
        model_answer_column=model_answer_column,
        score_column=score_column,
        question_id_column=question_id_column,
    )

    output = processed.copy()
    concept_keywords_values: list[str] = []
    present_values: list[str] = []
    missing_values: list[str] = []
    ratio_values: list[float] = []
    model_answer_values: list[str] = []

    answer_column = choose_student_answer_column(output, student_answer_column)

    for _, row in output.iterrows():
        question_id = row[question_id_column]
        model_answer = model_answers[question_id]
        concepts = extract_concept_keywords(model_answer, max_concepts=max_concepts)
        comparison = compare_concepts_with_answer(concepts, row[answer_column])

        model_answer_values.append(model_answer)
        concept_keywords_values.append("; ".join(concepts))
        present_values.append("; ".join(comparison["concepts_present"]))
        missing_values.append("; ".join(comparison["concepts_missing"]))
        ratio_values.append(comparison["concept_coverage_ratio"])

    output["model_answer_used"] = model_answer_values
    output["concepts"] = concept_keywords_values
    output["concept_keywords"] = concept_keywords_values
    output["concepts_present"] = present_values
    output["concepts_missing"] = missing_values
    output["concepts_covered_ratio"] = ratio_values
    output["concept_coverage_ratio"] = ratio_values
    return output


def infer_model_answers(
    dataframe: pd.DataFrame,
    model_answer_column: str,
    score_column: str,
    question_id_column: str,
) -> dict[object, str]:
    """Use the highest-scoring generated answer as the model answer per question."""
    missing = [
        column
        for column in [question_id_column, model_answer_column]
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model_answers: dict[object, str] = {}
    for question_id, group in dataframe.groupby(question_id_column, dropna=False):
        ordered = group.copy()
        if score_column in ordered.columns and ordered[score_column].notna().any():
            ordered = ordered.sort_values(score_column, ascending=False, na_position="last")
        model_answers[question_id] = str(ordered.iloc[0][model_answer_column])

    return model_answers


def choose_student_answer_column(dataframe: pd.DataFrame, preferred_column: str) -> str:
    """Prefer cleaned student-answer text when available."""
    clean_column = f"{preferred_column}_clean"
    if clean_column in dataframe.columns:
        return clean_column
    if preferred_column in dataframe.columns:
        return preferred_column
    raise ValueError(f"Missing student answer column: {preferred_column}")
