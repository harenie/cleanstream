"""Semantic similarity features for student answers and model answers."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from concept_coverage import (
    choose_model_answer_column,
    choose_student_answer_column,
    drop_source_model_answer_columns,
    infer_model_answers,
)
from preprocessing import clean_text, preprocess_dataframe


class SemanticSimilarityEngine:
    """Lightweight semantic baseline using TF-IDF word n-grams."""

    def __init__(self, corpus: list[str] | None = None) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.fit(corpus or ["empty answer"])

    def fit(self, corpus: list[str]) -> None:
        cleaned = [clean_text(text) for text in corpus if clean_text(text)]
        if not cleaned:
            cleaned = ["empty answer"]
        self.vectorizer.fit(cleaned)

    def similarity(self, text_a: object, text_b: object) -> float:
        """Return cosine similarity from 0.0 to 1.0."""
        a = clean_text(text_a)
        b = clean_text(text_b)
        if not a or not b:
            return 0.0

        vectors = self.vectorizer.transform([a, b])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(float(score), 4)


def add_semantic_similarity_columns(
    dataframe: pd.DataFrame,
    model_answer_column: str = "model_answer",
    student_answer_column: str = "synthetic_answer",
    question_id_column: str = "question_id",
    require_model_answer: bool = False,
) -> pd.DataFrame:
    """Add model-vs-student semantic similarity columns."""
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

    if answer_column == student_answer_column:
        answer_column = "student_answer"
    elif answer_column == f"{student_answer_column}_clean":
        answer_column = "student_answer_clean"

    model_answer_values = [model_answers[row[question_id_column]] for _, row in output.iterrows()]
    student_answer_values = output[answer_column].fillna("").astype(str).tolist()
    corpus = [answer for answer in model_answer_values if clean_text(answer)]
    corpus.extend(student_answer_values)
    engine = SemanticSimilarityEngine(corpus)

    output = drop_source_model_answer_columns(output, keep_column=model_answer_column)
    output["model_answer"] = model_answer_values
    output["missing_model_answer"] = [not bool(clean_text(value)) for value in model_answer_values]
    output["semantic_similarity_score"] = [
        engine.similarity(model_answer, student_answer)
        for model_answer, student_answer in zip(model_answer_values, student_answer_values)
    ]
    output["student_answer_length"] = [len(clean_text(value).split()) for value in student_answer_values]
    output["model_answer_length"] = [len(clean_text(value).split()) for value in model_answer_values]
    return output


def build_semantic_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    """Create a compact summary for semantic-similarity output."""
    eligible = dataframe[~dataframe["missing_model_answer"]]
    return {
        "rows": int(len(dataframe)),
        "eligible_rows": int(len(eligible)),
        "rows_missing_model_answer": int(dataframe["missing_model_answer"].sum()),
        "average_semantic_similarity_score": round(
            float(eligible["semantic_similarity_score"].mean()) if len(eligible) else 0.0,
            4,
        ),
    }
