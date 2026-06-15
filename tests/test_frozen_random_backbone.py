"""Frozen-random control: freeze a randomly-initialised backbone, no checkpoint.

_wrap_factory_for_fold must, when there is NO source provider but frozen_layers
is non-empty, return a factory that builds a fresh model and freezes the backbone
(head stays trainable) — without reading any checkpoint.
"""

from __future__ import annotations

import torch

from src.training.nested_cross_validation import NestedCrossValidator


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 1)


def _ncv(frozen_layers):
    # cfg is stored but unused by _wrap_factory_for_fold; pass None to avoid
    # constructing a full TrainerConfig.
    return NestedCrossValidator(cfg=None, source_provider=None, frozen_layers=frozen_layers)


def test_no_provider_no_frozen_returns_original_factory():
    ncv = _ncv([])
    orig = lambda cfg: _FakeModel()
    wrapped = ncv._wrap_factory_for_fold(orig, rep=0, fold=0, train_val_idx=[1, 2], test_idx=[0])
    assert wrapped is orig  # identity passthrough — from-scratch (transfer=none)


def test_frozen_random_freezes_backbone_only():
    ncv = _ncv(["backbone"])
    wrapped = ncv._wrap_factory_for_fold(
        lambda cfg: _FakeModel(), rep=0, fold=0, train_val_idx=[1, 2], test_idx=[0]
    )
    model = wrapped(None)
    bb = {n: p.requires_grad for n, p in model.named_parameters() if n.startswith("backbone")}
    hd = {n: p.requires_grad for n, p in model.named_parameters() if n.startswith("head")}
    assert bb and not any(bb.values()), f"backbone must be frozen, got {bb}"
    assert hd and all(hd.values()), f"head must stay trainable, got {hd}"


from pathlib import Path
from omegaconf import OmegaConf


def test_frozen_random_config_shape():
    cfg = OmegaConf.load(Path("configs/transfer/frozen_random.yaml"))
    # enabled=False => run_experiment builds NO SourceBackboneProvider.
    assert cfg.enabled is False
    assert cfg.source_checkpoint_root is None
    # non-empty frozen_layers => triggers the freeze-only path.
    assert list(cfg.frozen_layers) == ["backbone"]
