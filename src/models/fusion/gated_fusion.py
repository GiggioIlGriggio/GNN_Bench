"""Gated modality fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion.base_fusion import ModalityFusion


class GatedFusion(ModalityFusion):
    """Fuse SC and FC embeddings via a learnable gating mechanism.

    A sigmoid gate learns to weight the contribution of each modality.

    Parameters
    ----------
    d_sc : int
        Dimensionality of the structural embedding.
    d_fc : int
        Dimensionality of the functional embedding.
    d_out : int
        Desired output dimensionality.
    """

    def __init__(self, d_sc: int, d_fc: int, d_out: int) -> None:
        super().__init__()
        self.d_sc = d_sc
        self.d_fc = d_fc
        self._d_out = d_out

        self.proj_sc = nn.Linear(d_sc, d_out)
        self.proj_fc = nn.Linear(d_fc, d_out)
        self.gate = nn.Sequential(
            nn.Linear(d_sc + d_fc, d_out),
            nn.Sigmoid(),
        )

    def forward(
        self,
        emb_sc: torch.Tensor,
        emb_fc: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated fusion.

        Parameters
        ----------
        emb_sc : torch.Tensor
            ``[B, d_sc]``
        emb_fc : torch.Tensor
            ``[B, d_fc]``

        Returns
        -------
        torch.Tensor
            ``[B, d_out]``
        """
        g = self.gate(torch.cat([emb_sc, emb_fc], dim=-1))  # [B, d_out]
        return g * self.proj_sc(emb_sc) + (1 - g) * self.proj_fc(emb_fc)

    def get_output_dim(self) -> int:
        """Return ``d_out``."""
        return self._d_out
