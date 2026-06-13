import torch
import torch.nn as nn
from src.training.transfer_ops import freeze_layers, reinit_head
from src.configs.trainer_config import TrainerConfig
from src.training.trainer import Trainer


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def test_freeze_layers_freezes_only_matching_prefixes():
    m = _Tiny()
    n = freeze_layers(m, ["backbone"])
    assert n == 2  # weight + bias
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())


def test_freeze_layers_empty_list_is_noop():
    m = _Tiny()
    assert freeze_layers(m, []) == 0
    assert all(p.requires_grad for p in m.parameters())


def test_reinit_head_changes_head_weights_only():
    m = _Tiny()
    torch.nn.init.constant_(m.head[0].weight, 0.123)
    backbone_before = m.backbone.weight.detach().clone()
    reinit_head(m)
    assert not torch.allclose(m.head[0].weight, torch.full_like(m.head[0].weight, 0.123))
    assert torch.allclose(m.backbone.weight, backbone_before)


class _NullLogger:
    class _Cfg:
        enabled = False

    def __init__(self):
        self.cfg = self._Cfg()

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_build_optimizer_excludes_frozen_params():
    m = _Tiny()
    freeze_layers(m, ["backbone"])
    trainer = Trainer(cfg=TrainerConfig(), logger=_NullLogger())
    opt = trainer._build_optimizer(m)
    opt_param_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    assert all(id(p) not in opt_param_ids for p in m.backbone.parameters())
    assert all(id(p) in opt_param_ids for p in m.head.parameters())
