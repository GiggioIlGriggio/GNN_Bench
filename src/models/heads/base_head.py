"""Abstract base class for prediction heads."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PredictionHead(nn.Module, ABC):
    """Abstract base for mapping graph embeddings to predictions."""

    @abstractmethod
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map embedding to prediction.

        Parameters
        ----------
        embedding : torch.Tensor
            Shape ``[B, embedding_dim]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1]``.
        """
