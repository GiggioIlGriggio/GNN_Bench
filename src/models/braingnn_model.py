"""BrainGNN model (Li et al. 2021) with ROI-aware GConv and ROI-TopK pooling.

Reference: https://github.com/xxlya/BrainGNN_Pytorch
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import TopKPooling, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_sparse import spspmm

from src.configs.model_config import ModelConfig
from src.models.backbones.roi_aware_conv import ROIAwareConv
from src.models.base_model import BrainGNN
from src.models.heads.regression_head import RegressionHead
from src.models.registry import register_model

_EPS = 1e-8


@register_model("braingnn")
class BrainGNNModel(BrainGNN):
    """BrainGNN: ROI-aware graph convolution with hierarchical ROI-TopK pooling.

    Two alternating stages of GCNConv + TopKPooling are applied before a
    global mean pool and regression head.  Auxiliary losses (unit loss and
    topk/consist loss) are emitted during training and consumed by the trainer.

    Parameters
    ----------
    cfg : ModelConfig
        Model configuration.  BrainGNN-specific parameters are read from
        ``cfg.model_params``:

        - ``pool_ratio`` (float, default 0.5): fraction of ROIs kept per stage.
        - ``roi_embed_dim`` (int, default 8): size of the learned ROI embedding
          appended to node features before each GConv layer.
        - ``unit_loss_weight`` (float, default 0.3): weight on the unit loss.
        - ``topk_loss_weight`` (float, default 0.5): weight on the topk/consist loss.

    node_feat_dim : int
        Number of input node features.
    edge_feat_dim : int
        Number of input edge features (first channel used as edge weight).
    num_nodes : int
        Number of ROIs per graph (atlas size, constant across subjects).
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
        pool_ratio: float = float(p.get("pool_ratio", 0.5))
        roi_k: int = int(p.get("roi_embed_dim", 8))
        self.unit_loss_weight: float = float(p.get("unit_loss_weight", 0.3))
        self.topk_loss_weight: float = float(p.get("topk_loss_weight", 0.5))

        self.num_nodes = num_nodes
        self.pool_ratio = pool_ratio
        self.dropout = cfg.dropout

        # Stage 1: GConv + TopK pool
        self.conv1 = ROIAwareConv(node_feat_dim, cfg.hidden_dim, num_nodes, k=roi_k)
        self.bn1 = nn.BatchNorm1d(cfg.hidden_dim)
        self.pool1 = TopKPooling(cfg.hidden_dim, ratio=pool_ratio, nonlinearity=torch.sigmoid)

        # Stage 2: GConv + TopK pool
        self.conv2 = ROIAwareConv(cfg.hidden_dim, cfg.hidden_dim, num_nodes, k=roi_k)
        self.bn2 = nn.BatchNorm1d(cfg.hidden_dim)
        self.pool2 = TopKPooling(cfg.hidden_dim, ratio=pool_ratio, nonlinearity=torch.sigmoid)

        self.head = RegressionHead(cfg, embedding_dim=cfg.hidden_dim * 4)

        # Full-node pooling scores cached during the last encode() for aux loss.
        # Each entry is (full_score [N_total], num_graphs).
        self._pool_scores: List[Tuple[torch.Tensor, int]] = []

    # ------------------------------------------------------------------
    # BrainGNN interface
    # ------------------------------------------------------------------

    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Two-stage ROI-aware GConv + ROI-TopK pooling → hierarchical dual readout.

        After each TopK pooling stage, both global max-pool and global mean-pool
        are computed and concatenated.  The two stage embeddings are then
        concatenated, yielding a ``hidden_dim * 4`` graph representation:
        ``[max1 | mean1 | max2 | mean2]``.

        Caches full pre-selection pooling scores in ``self._pool_scores`` for
        use by ``auxiliary_loss()`` during training.

        Returns
        -------
        torch.Tensor
            Shape ``[B, hidden_dim * 4]``.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        num_graphs = int(batch.max().item()) + 1

        # ROI indices for each node in the batch: [0..N-1, 0..N-1, ...]
        roi_idx = torch.arange(self.num_nodes, device=x.device).repeat(num_graphs)

        self._pool_scores = []

        # --- Stage 1 ---
        x, edge_index, edge_attr, batch, roi_idx = self._conv_pool_stage(
            x, edge_index, edge_attr, batch, roi_idx,
            self.conv1, self.bn1, self.pool1, num_graphs,
        )

        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # Augment adjacency: A² = A@A on the pooled subgraph.
        n = x.size(0)
        if edge_attr is not None:
            ew = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
        else:
            ew = x.new_ones(edge_index.size(1))
        edge_index, ew = add_self_loops(edge_index, ew, num_nodes=n)
        edge_index, ew = sort_edge_index(edge_index, ew, num_nodes=n)
        edge_index, ew = spspmm(edge_index, ew, edge_index, ew, n, n, n)
        edge_index, ew = remove_self_loops(edge_index, ew)
        edge_attr = ew.unsqueeze(-1)

        # --- Stage 2 ---
        x, edge_index, edge_attr, batch, roi_idx = self._conv_pool_stage(
            x, edge_index, edge_attr, batch, roi_idx,
            self.conv2, self.bn2, self.pool2, num_graphs,
        )

        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        return torch.cat([x1, x2], dim=1)  # [B, hidden_dim * 4]

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Regression head: graph embedding → scalar prediction.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1]``.
        """
        return self.head(embedding)

    def auxiliary_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return unit loss and topk/consist loss accumulated during the last encode.

        Returns ``None`` when not in training mode or before the first forward pass.
        """
        if not self.training or not self._pool_scores:
            return None

        unit_loss = torch.stack(
            [self._unit_loss(score, self.pool_ratio) for score, _ in self._pool_scores]
        ).mean()

        topk_loss = torch.stack(
            [self._consist_loss(score, n_graphs) for score, n_graphs in self._pool_scores]
        ).mean()

        return {
            "unit_loss": self.unit_loss_weight * unit_loss,
            "topk_loss": self.topk_loss_weight * topk_loss,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _conv_pool_stage(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
        roi_idx: torch.Tensor,
        conv: ROIAwareConv,
        bn: nn.BatchNorm1d,
        pool: TopKPooling,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """One ROI-aware GConv + TopK pooling stage."""
        edge_weight: Optional[torch.Tensor] = None
        if edge_attr is not None:
            edge_weight = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr

        x = conv(x, edge_index, edge_weight, roi_idx)
        x = bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Compute full-node scores before pooling (PyG only returns selected nodes' scores).
        # Mirrors SelectTopK.forward: score = act((X @ p) / ||p||); scores in [-1, 1].
        w = pool.select.weight  # [1, hidden_dim]
        full_score_raw = (x * w).sum(dim=-1)
        full_score = pool.select.act(full_score_raw / (w.norm(p=2, dim=-1) + _EPS))

        x, edge_index, edge_attr, batch, perm, _ = pool(x, edge_index, edge_attr, batch)

        # Cache all-node scores for auxiliary loss.
        self._pool_scores.append((full_score.detach() if not self.training else full_score, num_graphs))

        # Track which ROI each surviving node corresponds to.
        roi_idx = roi_idx[perm]

        return x, edge_index, edge_attr, batch, roi_idx

    @staticmethod
    def _unit_loss(score: torch.Tensor, ratio: float) -> torch.Tensor:
        """Penalise pooling scores that are neither 0 nor 1 (sigmoid extremes).

        Scores from sigmoid TopKPooling are in (0, 1). Applies cross-entropy
        against hard 0/1 targets: top-ratio nodes should score near 1,
        bottom-ratio nodes near 0.
        """
        score_sorted = score.sort().values
        n = score_sorted.size(0)
        ratio_used = min(ratio, 1.0 - ratio)
        n_group = max(int(n * ratio_used), 1)

        loss = -torch.log(score_sorted[-n_group:] + _EPS).mean()
        loss = loss - torch.log(1.0 - score_sorted[:n_group] + _EPS).mean()
        return loss

    @staticmethod
    def _consist_loss(score: torch.Tensor, num_graphs: int) -> torch.Tensor:
        """Graph Laplacian regularisation: subjects should agree on which ROIs matter.

        A fully-connected subject graph with unit weights is used.
        Minimising the Laplacian quadratic form pulls each subject's score
        profile toward the group mean.
        """
        score_prob = torch.sigmoid(score)   # [-1, 1] → (0, 1)
        n_total = score_prob.size(0)
        n_per_graph, remainder = divmod(n_total, num_graphs)

        if remainder != 0 or n_per_graph == 0:
            # Non-uniform node counts — fall back to variance penalty.
            return score_prob.var() if score_prob.numel() > 1 else score_prob.new_zeros(1).squeeze()

        # s: [B, N_per_graph] — one score profile per subject.
        s = score_prob.reshape(num_graphs, n_per_graph)

        # Laplacian quadratic with W = ones(B, B): tr(S^T L S)
        #   = B * ||S||_F^2 - ||col_sum(S)||_2^2
        s_sq = (s * s).sum()
        col_sum = s.sum(dim=0)  # [N_per_graph]
        laplacian_trace = num_graphs * s_sq - (col_sum * col_sum).sum()
        return laplacian_trace / (num_graphs * num_graphs)
