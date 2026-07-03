from __future__ import annotations

import pandas as pd

from module1.concept_coverage import concept_coverage
from module1.concept_coverage.concept_coverage import build_concept_predictor
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

    output = semantic_module.add_semantic_similarity_columns(dataframe, similarity_backend="tfidf")

    assert captured["backend"] == ["tfidf"]
    assert captured["corpus"] == ["E-commerce is available anywhere and anytime."]
    assert output["similarity_backend"].tolist() == ["tfidf"]
    assert output["semantic_similarity_score"].between(0.0, 1.0).all()


def test_concept_backend_auto_uses_trained_model_when_available(monkeypatch):
    class FakeConceptPredictor:
        backend_name = "trained-llm"

        def __init__(self, model_path):
            self.model_path = model_path

    monkeypatch.setattr(concept_coverage, "ConceptCoveragePredictor", FakeConceptPredictor)

    predictor = build_concept_predictor(
        concept_backend="auto",
        concept_model_path="module1/models/concept_coverage_model",
        concept_nli_model_name=None,
        concept_nli_engine=None,
        target_score_column="ai_score",
    )

    assert predictor.backend_name == "trained-llm"


def test_concept_backend_auto_falls_back_to_weak_score(monkeypatch):
    class MissingConceptPredictor:
        def __init__(self, model_path):
            raise FileNotFoundError("missing model")

    monkeypatch.setattr(concept_coverage, "ConceptCoveragePredictor", MissingConceptPredictor)

    predictor = build_concept_predictor(
        concept_backend="auto",
        concept_model_path="missing-model",
        concept_nli_model_name=None,
        concept_nli_engine=None,
        target_score_column="ai_score",
    )

    assert predictor.backend_name == "weak-score"


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


def test_reasoning_skips_when_question_does_not_require_it():
    result = assess_reasoning(
        "An ERP implementation challenge is user resistance.",
        "Identify four ERP implementation challenges.",
        reasoning_required=False,
        reasoning_expected_type="not_required",
        reasoning_requirement_source="question_requirements",
        reasoning_skip_reason="Identify only question can be answered as a list.",
    )

    assert result["reasoning_required"] is False
    assert result["reasoning_quality"] == "not_applicable"
    assert result["reasoning_quality_score"] is None
    assert result["reasoning_backend"] == "not-required"
    assert result["reasoning_skip_reason"] == "Identify only question can be answered as a list."


def test_reasoning_can_use_trained_llm_style_predictor():
    class FakeReasoningPredictor:
        backend_name = "trained-llm"

        def predict_prompt(self, prompt: str) -> tuple[str, float]:
            assert "Expected Concept:" in prompt
            assert "reasoning quality concept" in prompt
            return "covered", 0.91

    result = assess_reasoning(
        "The answer explains the result because the cause changes demand.",
        "Explain the business impact.",
        backend="trained-llm",
        predictor=FakeReasoningPredictor(),
    )

    assert result["reasoning_quality"] == "good"
    assert result["reasoning_quality_score"] == 1.0
    assert result["reasoning_backend"] == "trained-llm"
    assert result["reasoning_model_label"] == "covered"
    assert result["reasoning_model_confidence"] == 0.91


def test_reasoning_can_use_nli_predictor():
    class FakeNLIResult:
        label = "entailment"
        entailment = 0.8
        neutral = 0.1
        contradiction = 0.1

    class FakeNLIEngine:
        def predict(self, premise, hypothesis):
            assert "explains why or how" in hypothesis
            return FakeNLIResult()

    result = assess_reasoning(
        "The internet reduces global cost by lowering communication expenses.",
        "How has the Internet reduced the cost of operating globally?",
        backend="nli",
        nli_engine=FakeNLIEngine(),
        reasoning_expected_type="causal_explanation",
    )

    assert result["reasoning_quality"] == "good"
    assert result["reasoning_backend"] == "nli"
    assert result["reasoning_nli_entailment_score"] == 0.8


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
    question_requirements = tmp_path / "question_requirements.csv"
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
    pd.DataFrame(
        [
            {
                "question_id": "Q1",
                "reasoning_required": True,
                "reasoning_expected_type": "descriptive_explanation",
                "reasoning_notes": "Requires explanation.",
            },
            {
                "question_id": "Q2",
                "reasoning_required": False,
                "reasoning_expected_type": "not_required",
                "reasoning_notes": "Definition only.",
            },
        ]
    ).to_csv(question_requirements, index=False)
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
        concept_source="reference",
        concept_backend="weak-score",
        similarity_backend="tfidf",
        reasoning_backend="rule-based",
        contradiction_backend="rule-based",
        question_requirements_path=question_requirements,
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
        "reasoning_required",
        "reasoning_expected_type",
        "reasoning_requirement_source",
        "reasoning_skip_reason",
        "reasoning_quality",
        "reasoning_connective_density",
        "contradiction_check_applied",
        "contradiction_backend",
        "contradiction_score",
        "language_quality_score",
        "cross_question_flag",
        "answer_word_count",
    }
    assert expected_columns.issubset(output.columns)
    assert len(output) == 2
    assert output["missing_model_answer"].eq(False).all()
    assert output["semantic_similarity_score"].between(0.0, 1.0).all()
    assert output["concept_coverage_ratio"].between(0.0, 1.0).all()
    assert output.loc[output["question_id"] == "Q2", "reasoning_quality"].iloc[0] == "not_applicable"
