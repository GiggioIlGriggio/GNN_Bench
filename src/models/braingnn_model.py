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
        """Faithful port of upstream Network.forward, truncated at the readout.

        Returns the hierarchical dual readout ``[B, hidden*4]`` and stashes
        the pooling weights/scores + targets for ``auxiliary_loss``.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1

        if edge_attr is not None:
            edge_weight = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
        else:
            edge_weight = x.new_ones(edge_index.size(1))

        # Synthesize upstream's data.pos: per-ROI identity, repeated per graph.
        pos = F.one_hot(
            torch.arange(self.num_nodes, device=x.device).repeat(num_graphs),
            self.num_nodes,
        ).float()

        # --- Stage 1 ---
        x = self.conv1(x, edge_index, edge_weight, pos)
        x, edge_index, edge_weight, batch, perm, score1 = self.pool1(
            x, edge_index, edge_weight, batch
        )
        pos = pos[perm]
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # A² adjacency augmentation on the pooled subgraph.
        edge_weight = edge_weight.squeeze()
        edge_index, edge_weight = self._augment_adj(edge_index, edge_weight, x.size(0))

        # --- Stage 2 ---
        x = self.conv2(x, edge_index, edge_weight, pos)
        x, edge_index, edge_weight, batch, perm, score2 = self.pool2(
            x, edge_index, edge_weight, batch
        )
        pos = pos[perm]
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # Stash for auxiliary_loss. The extra sigmoid here matches upstream
        # Network.forward (see PROVENANCE: this is the 2nd of 3 sigmoids).
        self._w1 = self.pool1.select.weight
        self._w2 = self.pool2.select.weight
        self._s1 = torch.sigmoid(score1).view(num_graphs, -1)
        self._s2 = torch.sigmoid(score2).view(num_graphs, -1)
        self._y = data.y.detach().view(-1)

        return torch.cat([x1, x2], dim=1)  # [B, hidden*4]

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Configurable RegressionHead: [B, hidden*4] -> [B, 1]."""
        return self.head(embedding)

    def auxiliary_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        """Upstream topk + unit + (binned) consistency losses, pre-scaled.

        Returns None in eval or before the first forward.
        """
        if not self.training or self._s1 is None:
            return None

        topk = topk_loss(self._s1, self.pool_ratio) + topk_loss(self._s2, self.pool_ratio)
        unit = (torch.norm(self._w1, p=2) - 1) ** 2 \
            + (torch.norm(self._w2, p=2) - 1) ** 2
        consist = self._binned_consist_loss(self._s1, self._y, self.consist_n_bins)

        return {
            "topk_loss": self.lambda_topk * topk,
            "unit_loss": self.lambda_unit * unit,
            "consist_loss": self.lambda_consist * consist,
        }

    def _binned_consist_loss(
        self, s: torch.Tensor, y: torch.Tensor, n_bins: int
    ) -> torch.Tensor:
        """Consistency loss applied per target-quantile bin (regression analogue
        of upstream's per-class loop). ``s`` is [B, n_kept], ``y`` is [B]."""
        y = y.view(-1).float()
        total = s.new_zeros(())
        if y.numel() == 0 or n_bins <= 1:
            return total + consist_loss(s)

        qs = torch.linspace(0, 1, n_bins + 1, device=y.device)[1:-1]
        edges = torch.quantile(y, qs)
        bins = torch.bucketize(y, edges)  # values in [0, n_bins-1]
        for b in range(n_bins):
            mask = bins == b
            if int(mask.sum()) >= 1:
                total = total + consist_loss(s[mask])
        return total

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
