import numpy as np
import torch
from torch_geometric.data import Data

from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from src.training.nested_cross_validation import NestedCVResult
from src.training.sklearn_nested_cv import SklearnNestedCrossValidator


class _NullLogger:
    class _Cfg:
        enabled = False
    def __init__(self): self.cfg = self._Cfg()
    def __getattr__(self, name):
        def _noop(*a, **k): return None
        return _noop


def _signal_dataset(n=80, nodes=6, seed=0):
    """Node features carry a linear signal so r2 should be clearly positive."""
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    ea = torch.ones(ei.shape[1], 1)
    for _ in range(n):
        x = torch.tensor(rng.normal(size=(nodes, 1)), dtype=torch.float32)
        y = float(x.sum().item() + rng.normal(scale=0.05))
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=nodes))
        labels.append(y)
    return graphs, np.asarray(labels)


def test_sklearn_nested_cv_runs_and_recovers_signal(tmp_path):
    graphs, labels = _signal_dataset()
    trainer_cfg = TrainerConfig(
        n_repetitions=1, n_outer_folds=2, inner_hpo_trials=2,
        hpo_metric="val_r2", stratify_bins=4, seed=42,
        checkpoint_dir=str(tmp_path),
        search_space="configs/sweeper/elasticnet.yaml",
    )
    model_cfg = ModelConfig(
        name="elasticnet", kind="sklearn", mlp_input="node_features",
        model_params={"alpha": 0.1, "l1_ratio": 0.5},
    )
    scv = SklearnNestedCrossValidator(
        cfg=trainer_cfg,
        search_space_path="configs/sweeper/elasticnet.yaml",
    )
    result = scv.run(
        estimator_name="elasticnet",
        model_cfg=model_cfg,
        dataset=graphs,
        labels=labels,
        logger=_NullLogger(),
        run_name="test_run",
        glm_col_range=None,
        glm_normalize=False,
        num_nodes=6,
    )
    assert isinstance(result, NestedCVResult)
    assert len(result.fold_results) == 2
    assert "r2" in result.mean_metrics
    # node-feature sum is highly learnable → pooled-ish mean r2 clearly positive
    assert result.mean_metrics["r2"] > 0.5
    # result JSON persisted + reloadable
    saved = NestedCVResult.load(tmp_path / "test_run" / "nested_cv_result.json")
    assert saved.model_name == "elasticnet"
