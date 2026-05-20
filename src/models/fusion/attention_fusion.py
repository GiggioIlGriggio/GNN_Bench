"""Cross-attention modality fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion.base_fusion import ModalityFusion


class CrossAttentionFusion(ModalityFusion):
    """Fuse SC and FC embeddings via cross-attention.

    Each modality attends to the other, producing a joint embedding.

    Parameters
    ----------
    d_sc : int
        Dimensionality of the structural embedding.
    d_fc : int
        Dimensionality of the functional embedding.
    d_out : int
        Desired output dimensionality.
    num_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        d_sc: int,
        d_fc: int,
        d_out: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self._d_out = d_out
        self.d_sc = d_sc
        self.d_fc = d_fc
        self.num_heads = num_heads

        self.proj_sc = nn.Linear(d_sc, d_out)
        self.proj_fc = nn.Linear(d_fc, d_out)
        self.attn_sc = nn.MultiheadAttention(d_out, num_heads, batch_first=True)
        self.attn_fc = nn.MultiheadAttention(d_out, num_heads, batch_first=True)
        self.out_proj = nn.Linear(d_out * 2, d_out)

    def forward(
        self,
        emb_sc: torch.Tensor,
        emb_fc: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention fusion.

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
        q_sc = self.proj_sc(emb_sc).unsqueeze(1)  # [B, 1, d_out]
        q_fc = self.proj_fc(emb_fc).unsqueeze(1)  # [B, 1, d_out]
        ctx_sc, _ = self.attn_sc(q_sc, q_fc, q_fc)  # SC attends to FC
        ctx_fc, _ = self.attn_fc(q_fc, q_sc, q_sc)  # FC attends to SC
        fused = torch.cat([ctx_sc.squeeze(1), ctx_fc.squeeze(1)], dim=-1)  # [B, 2*d_out]
        return self.out_proj(fused)  # [B, d_out]

    def get_output_dim(self) -> int:
        """Return ``d_out``."""
        return self._d_out
