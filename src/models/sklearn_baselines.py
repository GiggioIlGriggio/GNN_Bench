"""Classical-ML estimator registry for the sklearn baseline runner.

Each builder returns a leakage-safe sklearn Pipeline:
StandardScaler (fit on train fold) → estimator. Required for ElasticNet;
harmless for XGBoost. Hyperparameters arrive as a flat dict (the model's
``model_params`` after TrialOverrides) so the search-space DSL can sweep
``model.model_params.<hp>`` exactly like the GNN sweeps ``model.<field>``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_elasticnet(params: Dict[str, Any], seed: int):
    p = {"max_iter": 5000, "random_state": seed}
    p.update(params)
    return ElasticNet(**p)


def _build_xgboost(params: Dict[str, Any], seed: int):
    from xgboost import XGBRegressor

    p = {
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        "objective": "reg:squarederror",
    }
    p.update(params)
    return XGBRegressor(**p)


SKLEARN_ESTIMATORS: Dict[str, Callable[[Dict[str, Any], int], Any]] = {
    "elasticnet": _build_elasticnet,
    "xgboost": _build_xgboost,
}


def build_estimator(name: str, params: Dict[str, Any], *, seed: int) -> Pipeline:
    """Return a StandardScaler→estimator Pipeline for ``name`` with ``params``."""
    if name not in SKLEARN_ESTIMATORS:
        raise ValueError(
            f"Unknown sklearn estimator {name!r}; "
            f"expected one of {sorted(SKLEARN_ESTIMATORS)}."
        )
    estimator = SKLEARN_ESTIMATORS[name](params, seed)
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
