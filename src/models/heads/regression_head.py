"""MLP-based regression prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.configs.model_config import ModelConfig
from src.models.heads.base_head import PredictionHead


class RegressionHead(PredictionHead):
    """Multi-layer perceptron regression head.

    Parameters
    ----------
    cfg : ModelConfig
        Must provide ``head_hidden_dim``, ``head_num_layers``, ``dropout``.
    embedding_dim : int
        Dimensionality of the input graph embedding.
    batchnorm : bool
        Insert ``BatchNorm1d`` after the ReLU of each hidden layer. Off by
        default; BrainGNN opts in to stay faithful to the upstream FC head
        (``bn(relu(fc(x)))``).
    """

    def __init__(
        self, cfg: ModelConfig, embedding_dim: int, batchnorm: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = embedding_dim

        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for _ in range(cfg.head_num_layers):
            layers.append(nn.Linear(in_dim, cfg.head_hidden_dim))
            layers.append(nn.ReLU())
            if batchnorm:
                layers.append(nn.BatchNorm1d(cfg.head_hidden_dim))
            layers.append(nn.Dropout(p=cfg.dropout))
            in_dim = cfg.head_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map embedding to regression prediction.

        Parameters
        ----------
        embedding : torch.Tensor
            ``[B, embedding_dim]``

        Returns
        -------
        torch.Tensor
            ``[B, 1]``
        """
        return self.layers(embedding)
