"""Reasoning package exports."""

from module1.reasoning.reasoning import (
    COMMON_MISSPELLINGS,
    CONTRADICTION_EXCLUDED_PATTERNS,
    CONTRADICTION_TARGET_PATTERNS,
    REASONING_CONNECTIVES,
    assess_reasoning,
    contains_any,
    detect_contradictions,
    detect_noise,
    should_check_contradictions,
)

__all__ = [
    "COMMON_MISSPELLINGS",
    "CONTRADICTION_EXCLUDED_PATTERNS",
    "CONTRADICTION_TARGET_PATTERNS",
    "REASONING_CONNECTIVES",
    "assess_reasoning",
    "contains_any",
    "detect_contradictions",
    "detect_noise",
    "should_check_contradictions",
]
