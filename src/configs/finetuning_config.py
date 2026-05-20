"""Pydantic v2 schema for the ``finetuning`` Hydra config group."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class FinetuningConfig(BaseModel):
    """Validated configuration for fine-tuning and transfer learning.

    Controls checkpoint loading, layer freezing, and per-group learning rates.
    """

    enabled: bool = Field(
        default=False,
        description="Whether to run in fine-tuning mode.",
    )
    checkpoint_path: str = Field(
        default="",
        description="Path to the pretrained checkpoint directory (a fold_N folder).",
    )
    checkpoint_variant: Literal["best", "last"] = Field(
        default="best",
        description="Which model checkpoint to load: 'best' (model_best.pt) or 'last' (model_last.pt).",
    )
    checkpoint_fold: int = Field(
        default=0,
        description="Fold index to load the checkpoint from (e.g. 0 → fold_0/).",
    )
    frozen_layers: List[str] = Field(
        default_factory=list,
        description="List of layer name prefixes to freeze (e.g. ['backbone.convs']).",
    )
    lr_groups: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-layer-group learning rates: {name_prefix: lr}.",
    )
    epochs: int = Field(
        default=50,
        description="Number of fine-tuning epochs.",
    )
    epoch_checkpoint_dir: str = Field(
        default="",
        description=(
            "If set, run the 'epoch-sweep' finetuning mode: iterate over all "
            "epoch snapshots saved inside this directory "
            "(fold_<N>/epoch_checkpoints/*.pt) and finetune from each, "
            "logging R² as a function of pretraining epoch."
        ),
    )
