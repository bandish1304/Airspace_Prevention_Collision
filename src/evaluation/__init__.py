"""Shared evaluation utilities for model comparison."""

from .harness import (
    build_evaluation_harness,
    compute_classification_metrics,
    precision_recall_curve_data,
)

__all__ = [
    "build_evaluation_harness",
    "compute_classification_metrics",
    "precision_recall_curve_data",
]
