"""GAT backbone using :class:`torch_geometric.nn.GATConv`."""

from __future__ import annotations

from torch_geometric.nn import GATConv

from src.configs.model_config import ModelConfig
from src.models.backbones.base_backbone import GNNBackbone, build_norm


class GATBackbone(GNNBackbone):
    """Graph Attention Network backbone.

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

        self.convs.append(GATConv(in_channels, per_head_dim, heads=cfg.heads))
        for _ in range(1, cfg.num_layers):
            self.convs.append(GATConv(cfg.hidden_dim, per_head_dim, heads=cfg.heads))

        for _ in range(cfg.num_layers):
            self.norms.append(build_norm(cfg.norm, cfg.hidden_dim))

        self._build_jk()
