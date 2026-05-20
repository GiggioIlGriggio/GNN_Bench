"""Single-modality BrainGNN models (no fusion)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_geometric.data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from src.configs.model_config import ModelConfig
from src.models.backbones.gat import GATBackbone
from src.models.backbones.gcn import GCNBackbone
from src.models.backbones.gin import GINBackbone
from src.models.backbones.transformer import GraphTransformerBackbone
from src.models.base_model import BrainGNN
from src.models.heads.regression_head import RegressionHead
from src.models.registry import register_model

# ---------------------------------------------------------------------------
# Internal lookup tables
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY = {
    "gcn": GCNBackbone,
    "gat": GATBackbone,
    "gin": GINBackbone,
    "transformer": GraphTransformerBackbone,
}

_POOL_FNS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "add": global_add_pool,
}

# ---------------------------------------------------------------------------
# Base unimodal model
# ---------------------------------------------------------------------------


@register_model("unimodal")
class UnimodalBrainGNN(BrainGNN):
    """Single-modality GNN model: backbone → global pool → regression head (no fusion).

    Backbone type determined by ``cfg.backbone`` (gcn, gat, gin, transformer).

    Parameters
    ----------
    cfg : ModelConfig
        Full model configuration.  ``cfg.backbone`` selects the GNN layer type;
        ``cfg.pooling`` selects the graph-level aggregation.
    node_feat_dim : int
        Number of input node features.
    edge_feat_dim : int
        Number of input edge features.  Accepted for interface consistency;
        the default backbone convolutions do not use edge attributes.

    Example YAML usage::

        name: unimodal
        backbone: gcn      # or gat, gin, transformer
        hidden_dim: 64
        ...
    """

    def __init__(
        self,
        cfg: ModelConfig,
        node_feat_dim: int,
        edge_feat_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()

        # -- Backbone --------------------------------------------------------
        backbone_cls = _BACKBONE_REGISTRY.get(cfg.backbone)
        if backbone_cls is None:
            raise ValueError(
                f"Unknown backbone '{cfg.backbone}'. "
                f"Available: {list(_BACKBONE_REGISTRY)}"
            )
        self.backbone = backbone_cls(cfg, in_channels=node_feat_dim)

        # -- Global pooling --------------------------------------------------
        node_dim = self.backbone.get_output_dim()

        if cfg.pooling == "attention":
            from torch_geometric.nn import GlobalAttention

            self.pool = GlobalAttention(gate_nn=nn.Linear(node_dim, 1))
        elif cfg.pooling in _POOL_FNS:
            # Plain function: no parameters, stored as a regular attribute.
            self.pool = _POOL_FNS[cfg.pooling]
        else:
            raise ValueError(
                f"Unknown pooling '{cfg.pooling}'. "
                f"Available: {list(_POOL_FNS) + ['attention']}"
            )

        # -- Prediction head -------------------------------------------------
        self.head = RegressionHead(cfg, embedding_dim=node_dim)

    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Backbone forward pass + global pooling → graph embedding.

        Returns
        -------
        torch.Tensor
            Shape ``[B, node_dim]``.
        """
        node_emb = self.backbone(data)  # [N_total, node_dim]

        batch = getattr(data, "batch", None)
        if batch is None:
            # Single graph (no DataLoader batching)
            batch = node_emb.new_zeros(node_emb.size(0), dtype=torch.long)

        return self.pool(node_emb, batch)  # [B, node_dim]

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Regression head: graph embedding → scalar prediction.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1]``.
        """
        return self.head(embedding)

