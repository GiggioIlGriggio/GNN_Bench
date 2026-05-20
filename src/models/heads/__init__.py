"""Prediction head sub-package."""

from src.models.heads.base_head import PredictionHead
from src.models.heads.regression_head import RegressionHead

__all__ = [
    "PredictionHead",
    "RegressionHead",
]
