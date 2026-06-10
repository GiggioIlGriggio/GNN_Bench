import numpy as np
import torch
from torch_geometric.data import Data

from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from scripts.run_experiment import run_sklearn


class _NullLogger:
    class _Cfg:
        enabled = False
    def __init__(self): self.cfg = self._Cfg()
    def __getattr__(self, name):
        def _noop(*a, **k): return None
        return _noop


def test_run_sklearn_helper(tmp_path):
    rng = np.random.default_rng(0)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    ea = torch.ones(3, 1)
    graphs, labels = [], []
    for _ in range(40):
        x = torch.tensor(rng.normal(size=(4, 1)), dtype=torch.float32)
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=4))
        labels.append(float(x.sum()))
    labels = np.asarray(labels)
    trainer_cfg = TrainerConfig(
        n_repetitions=1, n_outer_folds=2, inner_hpo_trials=0,
        hpo_metric="val_r2", stratify_bins=4, checkpoint_dir=str(tmp_path),
    )
    model_cfg = ModelConfig(
        name="elasticnet", kind="sklearn", mlp_input="node_features",
        model_params={"alpha": 0.05, "l1_ratio": 0.5},
    )
    result = run_sklearn(
        model_cfg=model_cfg, trainer_cfg=trainer_cfg, graphs=graphs, labels=labels,
        logger=_NullLogger(), run_name="t", glm_col_range=None, glm_normalize=False,
        label_builder=None, label_components=None, feature_config={},
    )
    assert "r2" in result.mean_metrics
