import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models.sklearn_baselines import build_estimator, SKLEARN_ESTIMATORS


def test_registry_keys():
    assert set(SKLEARN_ESTIMATORS) == {"xgboost", "elasticnet"}


def test_elasticnet_built_with_params():
    pipe = build_estimator("elasticnet", {"alpha": 0.5, "l1_ratio": 0.3}, seed=0)
    assert isinstance(pipe, Pipeline)
    enet = pipe.steps[-1][1]
    assert enet.alpha == 0.5 and enet.l1_ratio == 0.3


def test_xgboost_built_with_params():
    pipe = build_estimator("xgboost", {"n_estimators": 50, "max_depth": 3}, seed=0)
    xgb = pipe.steps[-1][1]
    assert xgb.n_estimators == 50 and xgb.max_depth == 3


def test_unknown_estimator_raises():
    with pytest.raises(ValueError):
        build_estimator("randomforest", {}, seed=0)


def test_estimator_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)); y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    pipe = build_estimator("elasticnet", {"alpha": 0.01, "l1_ratio": 0.5}, seed=0)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (60,)
