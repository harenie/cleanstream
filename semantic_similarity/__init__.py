"""Semantic similarity package exports."""

from semantic_similarity.semantic_similarity import (
    SemanticSimilarityEngine,
    SentenceBertSimilarityEngine,
    add_semantic_similarity_columns,
    build_semantic_summary,
    build_similarity_engine,
)

__all__ = [
    "SemanticSimilarityEngine",
    "SentenceBertSimilarityEngine",
    "add_semantic_similarity_columns",
    "build_semantic_summary",
    "build_similarity_engine",
]
