"""Regression metrics: MAE, RMSE, R², Pearson r."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MetricDict = Dict[str, float]
"""Mapping from metric name to scalar value."""


@dataclass
class AggregatedMetrics:
    """Cross-fold aggregated metrics.

    Attributes
    ----------
    mean : MetricDict
        Mean of each metric across folds.
    std : MetricDict
        Standard deviation of each metric across folds.
    ci_95_lower : MetricDict
        Lower bound of 95 % confidence interval.
    ci_95_upper : MetricDict
        Upper bound of 95 % confidence interval.
    """

    mean: MetricDict = field(default_factory=dict)
    std: MetricDict = field(default_factory=dict)
    ci_95_lower: MetricDict = field(default_factory=dict)
    ci_95_upper: MetricDict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricDict:
    """Compute regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values, shape ``[N]``.
    y_pred : np.ndarray
        Predicted values, shape ``[N]``.

    Returns
    -------
    MetricDict
        Keys: ``"mae"``, ``"rmse"``, ``"r2"``, ``"pearson_r"``.
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    _EPS = 1e-8
    if len(y_true) > 1 and np.std(y_pred) > _EPS and np.std(y_true) > _EPS:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
        # Guard against NaN from corrcoef edge cases
        if np.isnan(pearson_r):
            warnings.warn(
                "pearson_r computation resulted in NaN. Setting to 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            pearson_r = 0.0
    else:
        if len(y_true) > 1 and np.std(y_pred) <= _EPS:
            warnings.warn(
                "pearson_r is undefined: all predictions are identical "
                f"(constant={y_pred[0]:.6f}). The model may have collapsed.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif len(y_true) > 1 and np.std(y_true) <= _EPS:
            warnings.warn(
                "pearson_r is undefined: all ground-truth labels are identical "
                f"(constant={y_true[0]:.6f}). The target has zero variance.",
                RuntimeWarning,
                stacklevel=2,
            )
        pearson_r = 0.0
    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": pearson_r}
    nan_metrics = {k: v for k, v in metrics.items() if np.isnan(v)}
    if nan_metrics:
        raise ValueError(
            f"NaN detected in metrics: {list(nan_metrics.keys())}\n"
            f"  All metrics: {metrics}\n"
            f"  y_true — shape={y_true.shape}, min={np.nanmin(y_true):.6f}, "
            f"max={np.nanmax(y_true):.6f}, mean={np.nanmean(y_true):.6f}, "
            f"std={np.nanstd(y_true):.6f}, n_nan={np.isnan(y_true).sum()}\n"
            f"  y_pred — shape={y_pred.shape}, min={np.nanmin(y_pred):.6f}, "
            f"max={np.nanmax(y_pred):.6f}, mean={np.nanmean(y_pred):.6f}, "
            f"std={np.nanstd(y_pred):.6f}, n_nan={np.isnan(y_pred).sum()}"
        )
    return metrics


def aggregate_fold_metrics(fold_metrics: List[MetricDict]) -> AggregatedMetrics:
    """Aggregate per-fold metrics into mean ± std and 95 % CI.

    Parameters
    ----------
    fold_metrics : List[MetricDict]
        One ``MetricDict`` per fold.

    Returns
    -------
    AggregatedMetrics
    """
    n = len(fold_metrics)
    keys = fold_metrics[0].keys()
    result = AggregatedMetrics()
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ci = 1.96 * std / np.sqrt(n)
        result.mean[k] = mean
        result.std[k] = std
        result.ci_95_lower[k] = mean - ci
        result.ci_95_upper[k] = mean + ci
    return result
