"""GCN backbone using :class:`torch_geometric.nn.GCNConv`."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from src.configs.model_config import ModelConfig
from src.models.backbones.base_backbone import GNNBackbone


class GCNBackbone(GNNBackbone):
    """Graph Convolutional Network backbone.

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

        self.convs.append(GCNConv(in_channels, cfg.hidden_dim))
        for _ in range(1, cfg.num_layers):
            self.convs.append(GCNConv(cfg.hidden_dim, cfg.hidden_dim))

        for _ in range(cfg.num_layers):
            self.norms.append(nn.BatchNorm1d(cfg.hidden_dim))

        self._build_jk()

    def _conv_forward(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pass scalar edge weights to :class:`GCNConv` via *edge_weight*.

        ``GCNConv`` expects a 1-D ``edge_weight`` tensor, so we extract the
        first channel of ``edge_attr`` when it is available.
        """
        if edge_attr is not None:
            edge_weight = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
            return conv(x, edge_index, edge_weight=edge_weight)
        return conv(x, edge_index)
