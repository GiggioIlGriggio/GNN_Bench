"""Graph Transformer backbone using :class:`torch_geometric.nn.TransformerConv`."""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import TransformerConv

from src.configs.model_config import ModelConfig
from src.models.backbones.base_backbone import GNNBackbone


class GraphTransformerBackbone(GNNBackbone):
    """Graph Transformer backbone.

    Parameters
    ----------
    cfg : ModelConfig
        Model hyperparameters (``hidden_dim``, ``num_layers``, ``dropout``,
        ``heads``, ``jk_mode``).
    in_channels : int
        Dimensionality of input node features.
    """

    def __init__(self, cfg: ModelConfig, in_channels: int) -> None:
        super().__init__(cfg, in_channels)

        assert cfg.hidden_dim % cfg.heads == 0, (
            f"hidden_dim ({cfg.hidden_dim}) must be divisible by heads ({cfg.heads})"
        )
        per_head_dim = cfg.hidden_dim // cfg.heads

        self.convs.append(TransformerConv(in_channels, per_head_dim, heads=cfg.heads))
        for _ in range(1, cfg.num_layers):
            self.convs.append(TransformerConv(cfg.hidden_dim, per_head_dim, heads=cfg.heads))

        for _ in range(cfg.num_layers):
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))

        self._build_jk()
