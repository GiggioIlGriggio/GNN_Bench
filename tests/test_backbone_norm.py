"""Tests for the configurable `model.norm` knob and the build_norm factory.

All tests use synthetic data — no real datasets.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch_geometric.data

from src.configs.model_config import ModelConfig
from src.models.backbones.base_backbone import build_norm
from src.models.backbones.gcn import GCNBackbone
from src.models.backbones.gat import GATBackbone
from src.models.backbones.gin import GINBackbone

BACKBONES = {"gcn": GCNBackbone, "gat": GATBackbone, "gin": GINBackbone}
NORM_TYPES = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "none": nn.Identity}


def _cfg(norm: str = "batch") -> ModelConfig:
    """Small backbone config; hidden_dim divisible by heads for GAT."""
    return ModelConfig(
        name="test",
        hidden_dim=16,
        num_layers=2,
        heads=4,
        dropout=0.0,
        norm=norm,
    )


def _make_synthetic_data(
    num_nodes: int = 20,
    node_feat_dim: int = 3,
    num_edges: int = 60,
    edge_feat_dim: int = 1,
) -> torch_geometric.data.Data:
    return torch_geometric.data.Data(
        x=torch.randn(num_nodes, node_feat_dim),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        # Positive edge weights: GCNConv's symmetric D^{-1/2} A D^{-1/2}
        # normalization is undefined (NaN) for negative/zero node degrees.
        edge_attr=torch.rand(num_edges, edge_feat_dim),
        y=torch.tensor([1.0], dtype=torch.float32),
    )


# --- Config validation -----------------------------------------------------

def test_norm_defaults_to_batch() -> None:
    assert ModelConfig(name="test").norm == "batch"


def test_norm_accepts_known_kinds() -> None:
    for kind in ("batch", "layer", "none"):
        assert ModelConfig(name="test", norm=kind).norm == kind


def test_norm_rejects_unknown_kind() -> None:
    with pytest.raises(Exception):  # pydantic.ValidationError
        ModelConfig(name="test", norm="bogus")


# --- build_norm factory ----------------------------------------------------

def test_build_norm_returns_expected_modules() -> None:
    assert isinstance(build_norm("batch", 16), nn.BatchNorm1d)
    assert isinstance(build_norm("layer", 16), nn.LayerNorm)
    assert isinstance(build_norm("none", 16), nn.Identity)


def test_build_norm_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_norm("bogus", 16)


# --- backbone wiring: norm module type per (backbone, kind) ----------------

@pytest.mark.parametrize("backbone_name", list(BACKBONES))
@pytest.mark.parametrize("kind", list(NORM_TYPES))
def test_backbone_norm_module_type(backbone_name: str, kind: str) -> None:
    cfg = _cfg(norm=kind)
    backbone = BACKBONES[backbone_name](cfg, in_channels=3)
    assert len(backbone.norms) == cfg.num_layers
    for norm_layer in backbone.norms:
        assert isinstance(norm_layer, NORM_TYPES[kind])


# --- default-unchanged regression -----------------------------------------

@pytest.mark.parametrize("backbone_name", list(BACKBONES))
def test_default_norm_is_batchnorm(backbone_name: str) -> None:
    """Omitting `norm` must reproduce today's BatchNorm1d (prior results valid)."""
    cfg = ModelConfig(name="test", hidden_dim=16, num_layers=2, heads=4)
    backbone = BACKBONES[backbone_name](cfg, in_channels=3)
    assert isinstance(backbone.norms[0], nn.BatchNorm1d)


# --- forward smoke per (backbone, kind) -----------------------------------

@pytest.mark.parametrize("backbone_name", list(BACKBONES))
@pytest.mark.parametrize("kind", list(NORM_TYPES))
def test_backbone_forward_smoke(backbone_name: str, kind: str) -> None:
    cfg = _cfg(norm=kind)
    backbone = BACKBONES[backbone_name](cfg, in_channels=3)
    backbone.eval()
    data = _make_synthetic_data(node_feat_dim=3)
    with torch.no_grad():
        out = backbone(data)
    assert out.shape == (data.x.size(0), backbone.get_output_dim())
    assert torch.isfinite(out).all()
