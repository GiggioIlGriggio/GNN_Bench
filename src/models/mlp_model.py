"""MLP model that operates on the flattened upper-triangular adjacency matrix."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data

from src.configs.model_config import ModelConfig
from src.models.base_model import BrainGNN
from src.models.heads.regression_head import RegressionHead
from src.models.registry import register_model


@register_model("mlp")
class MLPBrainModel(BrainGNN):
    """MLP model that reconstructs the dense adjacency matrix from the PyG
    sparse representation, extracts the upper triangle, flattens it, and
    passes it through an MLP encoder followed by a regression head.

    Input modes (``cfg.mlp_input``):

    - ``"adjacency"``: flattened upper-triangular adjacency matrix only.
    - ``"node_features"``: flattened node features only.
    - ``"both"``: concatenation of both.

    Adjacency type (``cfg.mlp_adjacency_type``):

    - ``"weighted"``: use edge weights from ``edge_attr``.
    - ``"binary"``: use 1.0 for every existing edge.

    Parameters
    ----------
    cfg : ModelConfig
        Full model configuration.
    node_feat_dim : int
        Number of input node features per node.
    edge_feat_dim : int
        Number of edge features (only first channel used for weighted adj).
    num_nodes : int
        Number of nodes in every graph (must be constant across subjects).
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
        self.cfg = cfg
        self.num_nodes = num_nodes

        # --- Compute input dimension ---
        adj_dim = num_nodes * (num_nodes - 1) // 2
        feat_dim = num_nodes * node_feat_dim

        if cfg.mlp_input == "adjacency":
            input_dim = adj_dim
        elif cfg.mlp_input == "node_features":
            input_dim = feat_dim
        else:  # "both"
            input_dim = adj_dim + feat_dim

        # --- Encoder MLP ---
        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(cfg.num_layers):
            encoder_layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(cfg.hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=cfg.dropout))
            in_dim = cfg.hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Prediction head ---
        self.head = RegressionHead(cfg, embedding_dim=cfg.hidden_dim)

    # ------------------------------------------------------------------
    # Adjacency reconstruction helpers
    # ------------------------------------------------------------------

    def _build_adjacency_vector(
        self, data: torch_geometric.data.Data
    ) -> torch.Tensor:
        """Reconstruct the upper-triangular adjacency vector for a batch.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Batched PyG data with ``edge_index``, optionally ``edge_attr``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N*(N-1)/2]``.
        """
        N = self.num_nodes
        tri_size = N * (N - 1) // 2

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.edge_index.new_zeros(data.num_nodes, dtype=torch.long)

        num_graphs = int(batch.max().item()) + 1
        device = data.edge_index.device

        # Pre-compute upper-triangular index mapping: (i, j) with i < j → flat index
        row_idx, col_idx = torch.triu_indices(N, N, offset=1, device=device)
        # Map (i, j) pair to flat position in the tri_size vector
        flat_map = torch.zeros(N, N, dtype=torch.long, device=device)
        flat_map[row_idx, col_idx] = torch.arange(tri_size, device=device)

        result = torch.zeros(num_graphs, tri_size, device=device)

        src, dst = data.edge_index  # [2, E]
        # Determine which graph each edge belongs to
        edge_batch = batch[src]

        # Local node indices within each graph
        # In PyG batching, node indices are offset by cumulative num_nodes
        node_offsets = torch.zeros(num_graphs, dtype=torch.long, device=device)
        for g in range(1, num_graphs):
            node_offsets[g] = node_offsets[g - 1] + N
        local_src = src - node_offsets[edge_batch]
        local_dst = dst - node_offsets[edge_batch]

        # Only keep upper-triangular edges (local_src < local_dst)
        upper_mask = local_src < local_dst
        local_src = local_src[upper_mask]
        local_dst = local_dst[upper_mask]
        edge_batch_filtered = edge_batch[upper_mask]

        flat_positions = flat_map[local_src, local_dst]

        if self.cfg.mlp_adjacency_type == "binary":
            values = torch.ones(flat_positions.size(0), device=device)
        else:
            # Use the first channel of edge_attr as weight
            edge_attr = data.edge_attr[upper_mask]
            values = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr

        result[edge_batch_filtered, flat_positions] = values
        return result

    # ------------------------------------------------------------------
    # BrainGNN interface
    # ------------------------------------------------------------------

    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Flatten adjacency / node features → MLP → embedding.

        Returns
        -------
        torch.Tensor
            Shape ``[B, hidden_dim]``.
        """
        parts: list[torch.Tensor] = []

        if self.cfg.mlp_input in ("adjacency", "both"):
            parts.append(self._build_adjacency_vector(data))

        if self.cfg.mlp_input in ("node_features", "both"):
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = data.x.new_zeros(data.x.size(0), dtype=torch.long)
            num_graphs = int(batch.max().item()) + 1

            # Reshape [N_total, feat] → [B, N*feat]
            x_flat = data.x.reshape(num_graphs, -1)
            parts.append(x_flat)

        mlp_input = torch.cat(parts, dim=1)
        return self.encoder(mlp_input)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Regression head: embedding → scalar prediction.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1]``.
        """
        return self.head(embedding)
