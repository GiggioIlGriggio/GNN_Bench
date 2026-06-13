import torch
import torch.nn as nn
from src.finetuning.transfer_nested import SourceBackboneProvider
from src.training.nested_cross_validation import NestedCrossValidator
from src.training.transfer_ops import freeze_layers, reinit_head


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.head = nn.Linear(3, 1)


def test_wrap_factory_loads_freezes_and_reinits(tmp_path, monkeypatch):
    # Build a fake source provider whose weights are all 7.0 on the backbone.
    root = tmp_path / "src"
    (root / "rep_0" / "fold_2").mkdir(parents=True)
    sd = {"backbone.weight": torch.full((3, 3), 7.0), "backbone.bias": torch.full((3,), 7.0)}
    torch.save(sd, root / "rep_0" / "fold_2" / "model_best.pt")
    import json
    (root / "fold_indices.json").write_text(json.dumps(
        {"folds": [{"rep": 0, "fold": 2, "train_val_idx": [0, 1], "test_idx": [2]}]}
    ))
    provider = SourceBackboneProvider(root)

    from src.configs.trainer_config import TrainerConfig
    ncv = NestedCrossValidator(cfg=TrainerConfig(), source_provider=provider,
                               frozen_layers=["backbone"])
    base_factory = lambda cfg=None: _Tiny()
    wrapped = ncv._wrap_factory_for_fold(
        base_factory, rep=0, fold=2, train_val_idx=[1, 0], test_idx=[2],
    )
    model = wrapped(None)
    # backbone loaded from source (==7.0) and frozen; head reinit'd & trainable
    assert torch.allclose(model.backbone.weight, torch.full((3, 3), 7.0))
    assert not model.backbone.weight.requires_grad
    assert model.head.weight.requires_grad


def test_wrap_factory_raises_on_misalignment(tmp_path):
    import json
    root = tmp_path / "src"
    (root / "rep_0" / "fold_0").mkdir(parents=True)
    torch.save({"backbone.weight": torch.zeros(3, 3)},
               root / "rep_0" / "fold_0" / "model_best.pt")
    (root / "fold_indices.json").write_text(json.dumps(
        {"folds": [{"rep": 0, "fold": 0, "train_val_idx": [0, 1], "test_idx": [2]}]}
    ))
    from src.configs.trainer_config import TrainerConfig
    ncv = NestedCrossValidator(cfg=TrainerConfig(),
                               source_provider=SourceBackboneProvider(root),
                               frozen_layers=[])
    wrapped = ncv._wrap_factory_for_fold(
        lambda cfg=None: _Tiny(), rep=0, fold=0, train_val_idx=[0, 9], test_idx=[2],
    )
    import pytest
    with pytest.raises(ValueError, match="fold-index mismatch"):
        wrapped(None)
