"""Pydantic v2 schema for the ``trainer`` Hydra config group."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class TrainerConfig(BaseModel):
    """Validated configuration for the training loop and cross-validation.

    The :class:`~src.training.trainer.Trainer` creates the optimizer and
    scheduler internally from these fields.
    """

    # --- Optimizer -----------------------------------------------------------
    optimizer: str = Field(
        default="adam",
        description="Optimizer class name (adam, adamw, sgd).",
    )
    lr: float = Field(
        default=1e-3,
        description="Initial learning rate.",
    )
    weight_decay: float = Field(
        default=1e-4,
        description="L2 regularisation weight.",
    )

    # --- Scheduler -----------------------------------------------------------
    scheduler: Optional[str] = Field(
        default="cosine",
        description="LR scheduler name (cosine, step, plateau, none).",
    )
    scheduler_params: Dict[str, Any] = Field(
        default_factory=lambda: {"T_max": 100},
        description="Additional scheduler keyword arguments.",
    )

    # --- Training loop -------------------------------------------------------
    epochs: int = Field(
        default=100,
        description="Maximum number of training epochs per fold.",
    )
    gradient_clip_val: Optional[float] = Field(
        default=1.0,
        description="Max gradient norm for clipping (None to disable).",
    )

    # --- Early stopping ------------------------------------------------------
    early_stopping_patience: int = Field(
        default=15,
        description="Number of epochs without improvement before stopping.",
    )
    early_stopping_metric: str = Field(
        default="val_mae",
        description="Metric to monitor for early stopping.",
    )
    early_stopping_min_delta: float = Field(
        default=1e-4,
        description="Minimum improvement to qualify as an improvement.",
    )

    # --- Label normalisation -------------------------------------------------
    label_norm_strategy: Literal["standard", "robust", "minmax", "none"] = Field(
        default="standard",
        description="Per-fold label normalisation strategy.",
    )

    # --- Cross-validation ----------------------------------------------------
    n_folds: int = Field(
        default=5,
        description="Number of cross-validation folds.",
    )
    val_ratio: float = Field(
        default=0.1,
        description="Fraction of training data held out for validation within each fold.",
    )
    test_ratio: float = Field(
        default=0.2,
        description="Fraction of data held out for testing (outer split).",
    )
    stratify_bins: int = Field(
        default=5,
        description="Number of quantile bins for stratified splitting of continuous labels.",
    )

    # --- Device --------------------------------------------------------------
    device: str = Field(
        default="auto",
        description=(
            "Device for training. 'auto' selects CUDA if available, "
            "otherwise CPU. Accepts any torch.device string: 'cpu', "
            "'cuda', 'cuda:0', 'cuda:1', etc."
        ),
    )

    # --- Reproducibility -----------------------------------------------------
    seed: int = Field(
        default=42,
        description="Global random seed.",
    )

    # --- Checkpointing -------------------------------------------------------
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory for saving fold checkpoints.",
    )
    save_every_n_epochs: int = Field(
        default=0,
        description=(
            "If > 0, save a model snapshot every this many epochs into "
            "fold_<N>/epoch_checkpoints/. 0 disables periodic saving."
        ),
    )
    save_on_best: bool = Field(
        default=False,
        description=(
            "If True, save a model snapshot whenever a new best validation "
            "metric is achieved (stored in fold_<N>/epoch_checkpoints/)."
        ),
    )
