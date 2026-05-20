"""Pydantic v2 schema for the ``trainer`` Hydra config group."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
    # ``n_folds`` is the legacy alias and still used by HydraSweep. For the
    # nested protocol (ADR-0008) the effective number of outer folds is
    # ``n_outer_folds`` if provided, otherwise it falls back to ``n_folds``.
    n_folds: int = Field(
        default=5,
        description=(
            "Legacy: number of cross-validation folds. Kept for backwards "
            "compatibility with HydraSweep. NestedCrossValidator uses "
            "``n_outer_folds`` when set."
        ),
    )
    n_outer_folds: Optional[int] = Field(
        default=None,
        description=(
            "Number of outer folds in the nested CV protocol (ADR-0008). "
            "Falls back to ``n_folds`` when ``None``."
        ),
    )
    n_repetitions: int = Field(
        default=1,
        description=(
            "Number of repetitions of the outer stratified K-fold. Total "
            "outer folds evaluated = ``n_repetitions × n_outer_folds``. "
            "Paper-grade default: 10."
        ),
    )
    inner_hpo_trials: int = Field(
        default=0,
        description=(
            "Number of Optuna trials per outer fold for inner HPO. "
            "0 disables inner HPO (legacy fast mode — fixed HPs). "
            "Paper-grade default: 20."
        ),
    )
    hpo_metric: Literal["val_mae", "val_r2"] = Field(
        default="val_mae",
        description=(
            "Validation metric optimised by Optuna and used for Trainer "
            "early stopping during inner training. Optimisation direction "
            "is inferred (minimize for val_mae, maximize for val_r2)."
        ),
    )
    outer_seeds: Optional[List[int]] = Field(
        default=None,
        description=(
            "Explicit per-repetition seeds for the outer StratifiedKFold. "
            "When ``None`` they are derived as ``seed + r`` for "
            "``r in range(n_repetitions)``."
        ),
    )
    search_space: Optional[str] = Field(
        default=None,
        description=(
            "Path to a sweeper YAML (e.g. ``configs/sweeper/bayesian.yaml``) "
            "whose ``params:`` block defines the inner HPO search space. "
            "Required when ``inner_hpo_trials > 0``."
        ),
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

    # ------------------------------------------------------------------
    # Derived / validated nested-CV accessors
    # ------------------------------------------------------------------
    @property
    def effective_n_outer_folds(self) -> int:
        """Outer-fold count used by NestedCrossValidator.

        Returns ``n_outer_folds`` when set, otherwise falls back to ``n_folds``.
        """
        return self.n_outer_folds if self.n_outer_folds is not None else self.n_folds

    @model_validator(mode="after")
    def _validate_nested_cv_fields(self) -> "TrainerConfig":
        if self.n_repetitions < 1:
            raise ValueError(
                f"n_repetitions must be >= 1, got {self.n_repetitions}"
            )
        # ``n_folds`` is allowed to be anything (legacy CrossValidator still
        # accepts 1) — we only constrain ``n_outer_folds`` when the user
        # explicitly opts into the nested protocol. NestedCrossValidator
        # enforces effective_n_outer_folds >= 2 at runtime.
        if self.n_outer_folds is not None and self.n_outer_folds < 2:
            raise ValueError(
                f"n_outer_folds must be >= 2, got {self.n_outer_folds}"
            )
        if self.inner_hpo_trials < 0:
            raise ValueError(
                f"inner_hpo_trials must be >= 0, got {self.inner_hpo_trials}"
            )
        if self.outer_seeds is not None and len(self.outer_seeds) != self.n_repetitions:
            raise ValueError(
                f"outer_seeds has length {len(self.outer_seeds)} but "
                f"n_repetitions={self.n_repetitions}"
            )
        if self.inner_hpo_trials > 0 and not self.search_space:
            raise ValueError(
                "inner_hpo_trials > 0 requires trainer.search_space to point "
                "at a sweeper YAML (e.g. configs/sweeper/bayesian.yaml)"
            )
        return self

    def resolved_outer_seeds(self) -> List[int]:
        """Return the per-repetition outer seeds, deriving them from ``seed`` if unset."""
        if self.outer_seeds is not None:
            return list(self.outer_seeds)
        return [self.seed + r for r in range(self.n_repetitions)]
