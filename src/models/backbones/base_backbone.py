"""Base class for GNN backbones with shared forward logic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from src.configs.model_config import ModelConfig


def _apply_adjacency_type(
    edge_attr: torch.Tensor | None,
    adjacency_type: str,
) -> torch.Tensor | None:
    """Return ``edge_attr`` processed according to *adjacency_type*.

    - ``"weighted"``: return *edge_attr* unchanged.
    - ``"binary"``: replace every edge weight with 1, preserving shape.
    """
    if edge_attr is None or adjacency_type == "weighted":
        return edge_attr
    return torch.ones_like(edge_attr)


class GNNBackbone(nn.Module):
    """Base for all GNN backbone architectures.

    A backbone produces **node-level** embeddings.  Global pooling is handled
    by :class:`~src.models.base_model.BrainGNN`, not the backbone.

    Subclasses must:

    1. Call ``super().__init__(cfg, in_channels)``.
    2. Populate ``self.convs`` and ``self.norms`` (same length).
    3. Call ``self._build_jk()`` at the end of ``__init__``.
    """

    def __init__(self, cfg: ModelConfig, in_channels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.convs: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_jk(self) -> None:
        """Build a :class:`JumpingKnowledge` module when *jk_mode* is not ``'last'``."""
        if self.cfg.jk_mode != "last":
            self.jk = JumpingKnowledge(self.cfg.jk_mode)

    def _conv_forward(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Call a single conv layer.

        Override in subclasses to pass edge weights / attributes.
        """
        return conv(x, edge_index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Compute node-level embeddings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must have ``x: [N, in_channels]``, ``edge_index: [2, E]``,
            and optionally ``edge_attr: [E, edge_feat_dim]``.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, out_dim]``.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        edge_attr = _apply_adjacency_type(edge_attr, self.cfg.gnn_adjacency_type)

        xs: list[torch.Tensor] = []
        for i, conv in enumerate(self.convs):
            x = self._conv_forward(conv, x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)
            xs.append(x)

        if hasattr(self, "jk"):
            x = self.jk(xs)

        return x

    def get_output_dim(self) -> int:
        """Return the dimensionality of the node embeddings this backbone produces."""
        if self.cfg.jk_mode == "cat":
            return self.cfg.num_layers * self.cfg.hidden_dim
        return self.cfg.hidden_dim
