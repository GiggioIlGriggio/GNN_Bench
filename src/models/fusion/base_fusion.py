"""Abstract base class for modality fusion strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModalityFusion(nn.Module, ABC):
    """Abstract base for fusing SC and FC graph-level embeddings.

    Unimodal models skip fusion entirely — the :class:`BrainGNN` base handles
    this without special-casing in the backbone or head.
    """

    @abstractmethod
    def forward(
        self,
        emb_sc: torch.Tensor,
        emb_fc: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse structural and functional embeddings.

        Parameters
        ----------
        emb_sc : torch.Tensor
            Structural embedding of shape ``[B, d_sc]``.
        emb_fc : torch.Tensor
            Functional embedding of shape ``[B, d_fc]``.

        Returns
        -------
        torch.Tensor
            Fused embedding of shape ``[B, d_out]``.
        """

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the dimensionality of the fused output.

        Returns
        -------
        int
        """
