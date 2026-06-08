"""GIN backbone using :class:`torch_geometric.nn.GINConv`."""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import GINConv

from src.configs.model_config import ModelConfig
from src.models.backbones.base_backbone import GNNBackbone, build_norm


class GINBackbone(GNNBackbone):
    """Graph Isomorphism Network backbone.

    Parameters
    ----------
    cfg : ModelConfig
        Model hyperparameters (``hidden_dim``, ``num_layers``, ``dropout``,
        ``jk_mode``).
    in_channels : int
        Dimensionality of input node features.
    """

    def __init__(self, cfg: ModelConfig, in_channels: int) -> None:
        super().__init__(cfg, in_channels)

        mlp = nn.Sequential(
            nn.Linear(in_channels, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.convs.append(GINConv(mlp))

        for _ in range(1, cfg.num_layers):
            mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        for _ in range(cfg.num_layers):
            self.norms.append(build_norm(cfg.norm, cfg.hidden_dim))

        self._build_jk()
