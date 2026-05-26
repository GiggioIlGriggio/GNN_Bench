"""BrainGNN (Li et al. 2021) — thin adapter over vendored upstream layers.

The scientifically-critical conv/message-passing code is vendored verbatim
under ``src/models/vendor/braingnn`` (see its PROVENANCE.md). This adapter
re-expresses the upstream ``Network.forward`` to fit the repo's
``encode``/``decode`` contract and swaps in the configurable RegressionHead.

Reference: https://github.com/xxlya/BrainGNN_Pytorch @ 1e337e7
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import TopKPooling, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_sparse import spspmm

from src.configs.model_config import ModelConfig
from src.models.base_model import BrainGNN
from src.models.heads.regression_head import RegressionHead
from src.models.registry import register_model
from src.models.vendor.braingnn import MyNNConv
from src.models.vendor.braingnn.losses import consist_loss, topk_loss


@register_model("braingnn")
class BrainGNNModel(BrainGNN):
    """BrainGNN adapter: vendored ROI-aware GConv + ROI-TopK pooling.

    Reads BrainGNN-specific parameters from ``cfg.model_params``:

    - ``pool_ratio`` (float, default 0.5): fraction of ROIs kept per stage.
    - ``roi_embed_dim`` (int, default 8): number of communities ``k`` in the
      per-ROI weight network.
    - ``lambda_topk`` (float, default 0.1): weight on the topk loss.
    - ``lambda_unit`` (float, default 0.0): weight on the unit loss.
    - ``lambda_consist`` (float, default 0.1): weight on the consistency loss.
    - ``consist_n_bins`` (int, default 4): target-quantile bins for the
      regression consistency loss.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        node_feat_dim: int,
        edge_feat_dim: int,
        num_nodes: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        if cfg.fusion is not None:
            raise NotImplementedError(
                "BrainGNN does not support multimodal fusion. Set model.fusion=null."
            )
        if num_nodes <= 0:
            raise ValueError(
                "BrainGNN requires num_nodes > 0. "
                "Ensure the dataset provides graphs with a fixed ROI count."
            )

        p = cfg.model_params
        self.pool_ratio: float = float(p.get("pool_ratio", 0.5))
        k: int = int(p.get("roi_embed_dim", 8))
        self.lambda_topk: float = float(p.get("lambda_topk", 0.1))
        self.lambda_unit: float = float(p.get("lambda_unit", 0.0))
        self.lambda_consist: float = float(p.get("lambda_consist", 0.1))
        self.consist_n_bins: int = int(p.get("consist_n_bins", 4))

        self.num_nodes = num_nodes
        hidden = cfg.hidden_dim
        R = num_nodes

        # Stage 1: per-ROI weight network -> vendored MyNNConv -> TopK pool.
        self.n1 = nn.Sequential(
            nn.Linear(R, k, bias=False), nn.ReLU(), nn.Linear(k, hidden * node_feat_dim)
        )
        self.conv1 = MyNNConv(node_feat_dim, hidden, self.n1, normalize=False)
        self.pool1 = TopKPooling(
            hidden, ratio=self.pool_ratio, multiplier=1, nonlinearity=torch.sigmoid
        )

        # Stage 2.
        self.n2 = nn.Sequential(
            nn.Linear(R, k, bias=False), nn.ReLU(), nn.Linear(k, hidden * hidden)
        )
        self.conv2 = MyNNConv(hidden, hidden, self.n2, normalize=False)
        self.pool2 = TopKPooling(
            hidden, ratio=self.pool_ratio, multiplier=1, nonlinearity=torch.sigmoid
        )

        self.head = RegressionHead(cfg, embedding_dim=hidden * 4)

        # Aux-loss tensors stashed during the last encode().
        self._s1: Optional[torch.Tensor] = None
        self._s2: Optional[torch.Tensor] = None
        self._w1: Optional[torch.Tensor] = None
        self._w2: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None

    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        raise NotImplementedError  # implemented in Task 5

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # implemented in Task 5

    def auxiliary_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError  # implemented in Task 6

    def _augment_adj(self, edge_index, edge_weight, num_nodes):
        """A² on the pooled subgraph (verbatim from upstream Network.augment_adj).

        Uses plain ``add_self_loops`` (default fill) — NOT
        ``add_remaining_self_loops(fill=1)``, which lives inside MyNNConv.
        """
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes)
        edge_index, edge_weight = spspmm(
            edge_index, edge_weight, edge_index, edge_weight,
            num_nodes, num_nodes, num_nodes,
        )
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
