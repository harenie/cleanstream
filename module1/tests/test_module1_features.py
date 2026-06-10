from __future__ import annotations

import pandas as pd

from module1.module1_features import build_module1_features
from module1.reasoning.reasoning import assess_reasoning, detect_contradictions
from module1.semantic_similarity import semantic_similarity as semantic_module


def test_semantic_similarity_fits_reference_answers_only(monkeypatch):
    captured: dict[str, list[str]] = {}
    original_builder = semantic_module.build_similarity_engine

    def capture_builder(backend: str, corpus: list[str] | None = None):
        captured["backend"] = [backend]
        captured["corpus"] = list(corpus or [])
        return original_builder(backend, corpus)

    monkeypatch.setattr(semantic_module, "build_similarity_engine", capture_builder)
    dataframe = pd.DataFrame(
        {
            "question_id": ["Q1"],
            "model_answer": ["E-commerce is available anywhere and anytime."],
            "synthetic_answer": ["rare_student_only_token appears in this answer."],
        }
    )

    output = semantic_module.add_semantic_similarity_columns(dataframe)

    assert captured["backend"] == ["tfidf"]
    assert captured["corpus"] == ["E-commerce is available anywhere and anytime."]
    assert output["similarity_backend"].tolist() == ["tfidf"]
    assert output["semantic_similarity_score"].between(0.0, 1.0).all()


def test_reasoning_connectives_do_not_match_inside_words():
    result = assess_reasoning("A software solution can support online commerce.")

    assert result["reasoning_connective_count"] == 0
    assert result["reasoning_quality"] == "poor"


def test_reasoning_quality_detects_explicit_explanation_markers():
    result = assess_reasoning(
        "Because consumers can access the store anywhere, search costs fall. "
        "Therefore bargaining power increases. However, firms need reliable systems."
    )

    assert result["reasoning_quality"] == "good"
    assert result["reasoning_connective_count"] == 3
    assert result["reasoning_connective_density"] > 0


def test_contradiction_detection_is_scoped_by_question_type():
    contradiction = detect_contradictions(
        "The cost is higher than revenue, but the business is profitable.",
        "What is profit?",
    )
    skipped = detect_contradictions(
        "The cost is higher than revenue, but the business is profitable.",
        "Discuss the advantages and disadvantages of this model.",
    )

    assert contradiction["contradiction_check_applied"] is True
    assert contradiction["contradiction_detected"] is True
    assert skipped["contradiction_check_applied"] is False
    assert skipped["contradiction_detected"] is False


def test_full_module1_feature_pipeline_smoke(tmp_path):
    concept_reference = tmp_path / "concepts.csv"
    pd.DataFrame(
        [
            {
                "question_id": "Q1",
                "concept_id": "Q1_C1",
                "question": "What is ubiquity in e-commerce?",
                "concept_text": "E-commerce is available anywhere and anytime.",
                "max_mark": 1.0,
            },
            {
                "question_id": "Q2",
                "concept_id": "Q2_C1",
                "question": "Explain the concept of information asymmetry.",
                "concept_text": "Sellers may know more product information than buyers.",
                "max_mark": 1.0,
            },
        ]
    ).to_csv(concept_reference, index=False)
    dataframe = pd.DataFrame(
        [
            {
                "question_id": "Q1",
                "question": "What is ubiquity in e-commerce?",
                "scheme": "E-commerce is available anywhere and anytime.",
                "synthetic_answer": "Because e-commerce is available anywhere, consumers can shop anytime.",
                "ai_score": 4.0,
            },
            {
                "question_id": "Q2",
                "question": "Explain the concept of information asymmetry.",
                "scheme": "Sellers may know more product information than buyers.",
                "synthetic_answer": "Information asymmetry means sellers know more than buyers.",
                "ai_score": 3.0,
            },
        ]
    )

    output = build_module1_features(
        dataframe,
        model_answer_column="scheme",
        student_answer_column="synthetic_answer",
        concept_reference_path=concept_reference,
        concept_backend="weak-score",
        language_check_backend="simple",
        require_model_answer=True,
    )

    expected_columns = {
        "student_answer",
        "model_answer",
        "concept_backend",
        "concept_coverage_ratio",
        "similarity_backend",
        "semantic_similarity_score",
        "reasoning_quality",
        "reasoning_connective_density",
        "contradiction_check_applied",
        "language_quality_score",
        "cross_question_flag",
        "answer_word_count",
    }
    assert expected_columns.issubset(output.columns)
    assert len(output) == 2
    assert output["missing_model_answer"].eq(False).all()
    assert output["semantic_similarity_score"].between(0.0, 1.0).all()
    assert output["concept_coverage_ratio"].between(0.0, 1.0).all()
