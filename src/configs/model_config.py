"""Pydantic v2 schema for the ``model`` Hydra config group."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Validated configuration for the full BrainGNN model.

    Covers backbone selection, fusion strategy, pooling, and prediction head.
    """

    name: str = Field(
        ...,
        description="Registered model name (must match a key in MODEL_REGISTRY).",
    )
    backbone: str = Field(
        default="gcn",
        description="GNN backbone type (gcn, gat, gin, transformer).",
    )
    hidden_dim: int = Field(
        default=64,
        description="Hidden dimensionality of GNN layers.",
    )
    num_layers: int = Field(
        default=3,
        description="Number of message-passing layers in the backbone.",
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout probability applied after each GNN layer.",
    )
    heads: int = Field(
        default=4,
        description="Number of attention heads (GAT / Transformer backbones).",
    )
    pooling: Literal["mean", "max", "add", "attention"] = Field(
        default="mean",
        description="Global graph pooling strategy.",
    )
    fusion: Optional[str] = Field(
        default=None,
        description="Fusion strategy name (concat, attention, gated). None for unimodal.",
    )
    jk_mode: Literal["last", "cat", "max"] = Field(
        default="last",
        description="Jumping Knowledge aggregation mode across layers.",
    )
    embedding_dim: int = Field(
        default=64,
        description="Dimensionality of the graph-level embedding (output of encode).",
    )
    head_hidden_dim: int = Field(
        default=32,
        description="Hidden dimension of the prediction head MLP.",
    )
    head_num_layers: int = Field(
        default=2,
        description="Number of layers in the prediction head MLP.",
    )

    # --- MLP-specific options ---
    mlp_input: Literal["adjacency", "node_features", "both"] = Field(
        default="adjacency",
        description="Input mode for the MLP model: flattened adjacency matrix, "
        "flattened node features, or both concatenated.",
    )
    mlp_adjacency_type: Literal["weighted", "binary"] = Field(
        default="weighted",
        description="Whether to use weighted edge values or binary (0/1) in the "
        "adjacency matrix (MLP model only).",
    )

    # --- GNN-specific options ---
    gnn_adjacency_type: Literal["weighted", "binary"] = Field(
        default="weighted",
        description="Whether to use weighted edge values or binary (0/1) when "
        "passing edge attributes to GNN backbones.",
    )

    # --- Per-model hyperparameters ---
    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form dict of model-specific hyperparameters "
        "(e.g. pool_ratio for BrainGNN). Each model reads its own keys; "
        "all other models ignore this field.",
    )
