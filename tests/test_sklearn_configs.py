import yaml

from src.configs.model_config import ModelConfig
from src.training.search_space import load_sweeper_params, parse_search_space


def test_kind_defaults_to_gnn():
    assert ModelConfig(name="gcn").kind == "gnn"


def test_kind_accepts_sklearn():
    assert ModelConfig(name="xgboost", kind="sklearn").kind == "sklearn"


def _load_model(path):
    with open(path) as f:
        return ModelConfig(**yaml.safe_load(f))


def test_model_configs_load_as_sklearn():
    for name in ("xgboost", "elasticnet"):
        cfg = _load_model(f"configs/model/{name}.yaml")
        assert cfg.kind == "sklearn" and cfg.name == name


def test_sweeper_configs_parse():
    for name in ("xgboost", "elasticnet"):
        specs = parse_search_space(load_sweeper_params(f"configs/sweeper/{name}.yaml"))
        assert specs and all(s.name.startswith("model.model_params.") for s in specs)
