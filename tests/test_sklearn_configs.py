from src.configs.model_config import ModelConfig


def test_kind_defaults_to_gnn():
    assert ModelConfig(name="gcn").kind == "gnn"


def test_kind_accepts_sklearn():
    assert ModelConfig(name="xgboost", kind="sklearn").kind == "sklearn"
