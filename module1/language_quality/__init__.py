"""Language quality package exports."""

from module1.language_quality.language_quality import (
    LANGUAGE_CHECK_BACKENDS,
    analyze_language_quality,
    calculate_language_penalty,
)

__all__ = [
    "LANGUAGE_CHECK_BACKENDS",
    "analyze_language_quality",
    "calculate_language_penalty",
]
