"""Abstract base class for all brain GNN models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch_geometric.data


class BrainGNN(nn.Module, ABC):
    """Abstract base for all brain GNN models.

    Subclasses compose a :class:`GNNBackbone`, an optional
    :class:`ModalityFusion` module, and a :class:`PredictionHead`.

    The ``forward`` method is final — override ``encode`` and ``decode``
    in subclasses.  Models with auxiliary training losses (e.g. BrainGNN)
    may also override ``auxiliary_loss``.
    """

    @abstractmethod
    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Produce a graph-level embedding from a PyG Data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must have ``x: [N, node_feat_dim]`` and
            ``edge_attr: [E, edge_feat_dim]``.

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size, embedding_dim]``.
        """

    @abstractmethod
    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map graph-level embedding to scalar prediction.

        Parameters
        ----------
        embedding : torch.Tensor
            Shape ``[batch_size, embedding_dim]``.

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size, 1]``.
        """

    def auxiliary_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return auxiliary loss terms accumulated during the last ``encode`` call.

        Override in subclasses that use auxiliary training objectives (e.g.
        BrainGNN's unit loss and topk loss).  The trainer calls this after
        every forward pass during training, sums the returned tensors into the
        total loss, and logs each term separately.

        Returns
        -------
        dict[str, Tensor] or None
            Keys are loss names (logged to wandb as ``fold_N/train/<name>``);
            values are scalar tensors **already scaled** by their loss weights.
            Return ``None`` (the default) when there are no auxiliary terms.
        """
        return None

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Full forward pass: ``encode`` → ``decode``.

        **Do not override.**

        Parameters
        ----------
        data : torch_geometric.data.Data

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size, 1]``.
        """
        embedding = self.encode(data)
        return self.decode(embedding)
