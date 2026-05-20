"""Concatenation-based modality fusion."""

from __future__ import annotations

import torch

from src.models.fusion.base_fusion import ModalityFusion


class ConcatFusion(ModalityFusion):
    """Fuse SC and FC embeddings by concatenation.

    Output dimension: ``d_sc + d_fc``.

    Parameters
    ----------
    d_sc : int
        Dimensionality of the structural embedding.
    d_fc : int
        Dimensionality of the functional embedding.
    """

    def __init__(self, d_sc: int, d_fc: int) -> None:
        super().__init__()
        self.d_sc = d_sc
        self.d_fc = d_fc

    def forward(
        self,
        emb_sc: torch.Tensor,
        emb_fc: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate SC and FC embeddings along the feature dimension.

        Parameters
        ----------
        emb_sc : torch.Tensor
            ``[B, d_sc]``
        emb_fc : torch.Tensor
            ``[B, d_fc]``

        Returns
        -------
        torch.Tensor
            ``[B, d_sc + d_fc]``
        """
        return torch.cat([emb_sc, emb_fc], dim=-1)

    def get_output_dim(self) -> int:
        """Return ``d_sc + d_fc``."""
        return self.d_sc + self.d_fc
