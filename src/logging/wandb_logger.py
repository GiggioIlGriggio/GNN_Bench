"""Thin, modular Weights & Biases wrapper.

Each logging method corresponds to one logical event.  All wandb key strings
are imported from :mod:`src.logging.log_schema`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from src.configs.logging_config import LoggingConfig
from src.configs.model_config import ModelConfig
from src.models.base_model import BrainGNN
from src.training.metrics import MetricDict

if TYPE_CHECKING:
    from src.training.cross_validation import CVResult


# ---------------------------------------------------------------------------
# Typed containers referenced by the logger
# ---------------------------------------------------------------------------

class DatasetStats:
    """Lightweight container for dataset-level statistics to log.

    Attributes
    ----------
    name : str
    num_subjects : int
    num_nodes : int
    modality : str
    atlas : str
    mean_edges : float, optional
        Mean number of edges per graph.
    std_edges : float, optional
        Standard deviation of edge counts per graph.
    """

    def __init__(
        self,
        name: str,
        num_subjects: int,
        num_nodes: int,
        modality: str,
        atlas: str,
        mean_edges: Optional[float] = None,
        std_edges: Optional[float] = None,
    ) -> None:
        self.name = name
        self.num_subjects = num_subjects
        self.num_nodes = num_nodes
        self.modality = modality
        self.atlas = atlas
        self.mean_edges = mean_edges
        self.std_edges = std_edges


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class WandbLogger:
    """Modular wandb logger — one method per logical event.

    Parameters
    ----------
    cfg : LoggingConfig
        Logging configuration.
    """

    def __init__(self, cfg: LoggingConfig) -> None:
        self.cfg = cfg
        self._run = None  # wandb.Run, set during init_run

    def init_run(self, full_config: dict) -> None:
        """Initialise a wandb run.
        
        Parameters
        ----------
        full_config : dict
            Flat dict of the full experiment config (for wandb.config).
        """
        if not self.cfg.enabled:
            return  # no-op if logging disabled

        import wandb
        self._run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.cfg.run_name,
            tags=self.cfg.tags,
            config=full_config,
            reinit=True,
        )

    def finish(self) -> None:
        """Finish the current wandb run."""
        if not self.cfg.enabled or self._run is None:
            return
        import wandb
        wandb.finish()

    def log_fold_metrics(
        self,
        fold_idx: int,
        metrics: MetricDict,
        split: str,
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics for a single fold/split.

        Parameters
        ----------
        fold_idx : int
        metrics : MetricDict
        split : str
            ``"train"``, ``"val"``, or ``"test"``.
        epoch : Optional[int]
            Zero-based training epoch associated with these metrics.
        """
        if not self.cfg.enabled:
            return
        import wandb
        log_data = {f"fold_{fold_idx}/{split}/{k}": v for k, v in metrics.items()}
        if epoch is not None:
            log_data[f"fold_{fold_idx}/epoch"] = epoch
            log_data[f"fold_{fold_idx}/{split}/epoch"] = epoch
        wandb.log(log_data)

    def log_fold_splits(
        self,
        fold_idx: int,
        train_ids: "List[str]",
        val_ids: "List[str]",
        test_ids: "List[str]",
    ) -> None:
        """Upload train/val/test subject ID lists for one fold as a wandb Artifact.

        Creates a JSON artifact named ``fold_{fold_idx}_splits`` of type
        ``dataset_splits`` (see :data:`src.logging.log_schema.FOLD_SPLITS_ARTIFACT_TYPE`).
        The JSON file has three keys: ``"train"``, ``"val"``, ``"test"``, each
        containing a list of subject ID strings.

        Parameters
        ----------
        fold_idx : int
            Zero-based fold index.
        train_ids : List[str]
            Subject IDs assigned to the training split.
        val_ids : List[str]
            Subject IDs assigned to the validation split.
        test_ids : List[str]
            Subject IDs assigned to the test split.
        """
        if not self.cfg.enabled:
            return

        import json
        import os
        import tempfile

        import wandb
        from src.logging.log_schema import FOLD_SPLITS_ARTIFACT_TYPE

        splits = {"train": train_ids, "val": val_ids, "test": test_ids}

        artifact = wandb.Artifact(
            name=f"{wandb.run.name}_{wandb.run.id}_fold_{fold_idx}_splits_run",
            type=FOLD_SPLITS_ARTIFACT_TYPE,
            description=(
                f"Train/val/test subject IDs for fold {fold_idx}. "
                f"Counts: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}."
            ),
            metadata={
                "fold_idx": fold_idx,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_test": len(test_ids),
                "wandb_run_id": wandb.run.id,
                "wandb_run_name": wandb.run.name,
            },
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(splits, tmp, indent=2)
            tmp_path = tmp.name

        try:
            artifact.add_file(tmp_path, name=f"{wandb.run.name}_{wandb.run.id}_fold_{fold_idx}_splits_run.json")
            wandb.log_artifact(artifact)
        finally:
            os.unlink(tmp_path)

    def log_model_summary(self, model: BrainGNN, cfg: ModelConfig) -> None:
        """Log model architecture summary (param count, name, backbone).

        Parameters
        ----------
        model : BrainGNN
        cfg : ModelConfig
        """
        if not self.cfg.enabled:
            return
        import wandb
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "model/param_count": total,
            "model/trainable_param_count": trainable,
            "model/name": type(model).__name__,
        })

    def log_dataset_stats(self, stats: DatasetStats) -> None:
        """Log dataset-level statistics.

        Parameters
        ----------
        stats : DatasetStats
        """
        if not self.cfg.enabled:
            return
        import wandb
        log_dict = {
            "dataset/name": stats.name,
            "dataset/num_subjects": stats.num_subjects,
            "dataset/num_nodes": stats.num_nodes,
            "dataset/modality": stats.modality,
            "dataset/atlas": stats.atlas,
            "dataset/num_edges_mean": stats.mean_edges,
            "dataset/num_edges_std": stats.std_edges,
        }
        wandb.log(log_dict)

    def log_sweep_trial(self, params: dict, objective: float) -> None:
        """Log a single sweep trial result.

        Parameters
        ----------
        params : dict
            Trial hyperparameters.
        objective : float
            Objective metric value.
        """
        if not self.cfg.enabled:
            return
        import wandb
        wandb.log({"sweep/trial/params": params, "sweep/trial/objective": objective})

    def log_final_summary(self, result: CVResult) -> None:
        """Log aggregated cross-validation summary.

        Parameters
        ----------
        result : CVResult
        """
        if not self.cfg.enabled:
            return
        import wandb
        log_data = {}
        for k, v in result.aggregated.items():
            log_data[f"cv/pooled/{k}"] = v
        print("Logging final CV summary to wandb: %s", log_data)
        wandb.log(log_data)

    def log_epoch_sweep_metrics(
        self,
        pretrain_epoch: int,
        metrics: MetricDict,
    ) -> None:
        """Log finetuning metrics for one pretrain-epoch checkpoint.

        Used by the epoch-sweep finetuning mode to create a curve of
        ``pretrain_epoch → finetuning R²``.  Each call creates one data point;
        the pretrain epoch is included in the logged data so you can plot
        ``epoch_sweep/pretrain_epoch`` vs ``epoch_sweep/r2`` in a custom chart.

        Parameters
        ----------
        pretrain_epoch : int
            Zero-based epoch index of the pretrained checkpoint used.
        metrics : MetricDict
            Aggregated CV metrics (mae, rmse, r2, pearson_r) after finetuning
            from this checkpoint.
        """
        if not self.cfg.enabled:
            return
        import wandb

        log_data = {f"epoch_sweep/{k}": v for k, v in metrics.items()}
        log_data["epoch_sweep/pretrain_epoch"] = pretrain_epoch
        # Do NOT set step=pretrain_epoch to avoid wandb step ordering issues.
        # Instead, let step auto-increment and use pretrain_epoch as logged data.
        wandb.log(log_data, commit=True)

    # ------------------------------------------------------------------
    # Nested cross-validation logging (ADR-0008)
    # ------------------------------------------------------------------

    def log_nested_trial_metrics(
        self,
        rep: int,
        fold: int,
        trial: int,
        split: str,
        metrics: MetricDict,
        epoch: Optional[int] = None,
    ) -> None:
        """Log per-epoch metrics for one inner-HPO trial under ``rep_<r>/fold_<k>/trial_<t>/<split>/...``.

        Parameters
        ----------
        rep : int
            Zero-based outer repetition index.
        fold : int
            Zero-based outer fold index.
        trial : int
            Zero-based Optuna trial index within this outer fold.
        split : str
            ``"train"`` or ``"val"``.
        metrics : MetricDict
            Metric → value mapping.
        epoch : Optional[int]
            Inner training epoch index.
        """
        if not self.cfg.enabled:
            return
        import wandb
        from src.logging.log_schema import nested_rep_fold_trial_split_key

        log_data = {
            nested_rep_fold_trial_split_key(rep, fold, trial, split, k): v
            for k, v in metrics.items()
        }
        if epoch is not None:
            log_data[f"rep_{rep}/fold_{fold}/trial_{trial}/{split}/epoch"] = epoch
        wandb.log(log_data)

    def log_nested_outer_test(
        self,
        rep: int,
        fold: int,
        metrics: MetricDict,
    ) -> None:
        """Log outer-test metrics for one outer fold under ``rep_<r>/fold_<k>/test/...``."""
        if not self.cfg.enabled:
            return
        import wandb
        from src.logging.log_schema import nested_rep_fold_test_key

        log_data = {
            nested_rep_fold_test_key(rep, fold, k): v for k, v in metrics.items()
        }
        wandb.log(log_data)

    def log_nested_best_hparams(
        self,
        rep: int,
        fold: int,
        best_hparams: Dict[str, object],
        best_trial: int,
        refit_epochs: int,
    ) -> None:
        """Log the chosen hyperparameters for one outer fold."""
        if not self.cfg.enabled:
            return
        import wandb
        from src.logging.log_schema import (
            NESTED_BEST_HPARAMS_KEY_FMT,
            NESTED_BEST_TRIAL_KEY_FMT,
            NESTED_REFIT_EPOCHS_KEY_FMT,
        )

        wandb.log({
            NESTED_BEST_HPARAMS_KEY_FMT.format(r=rep, k=fold): best_hparams,
            NESTED_BEST_TRIAL_KEY_FMT.format(r=rep, k=fold): best_trial,
            NESTED_REFIT_EPOCHS_KEY_FMT.format(r=rep, k=fold): refit_epochs,
        })

    def log_nested_final_summary(
        self,
        mean: MetricDict,
        std: MetricDict,
    ) -> None:
        """Log cross-fold mean and std of the outer-test metrics.

        Parameters
        ----------
        mean : MetricDict
            Mean of each metric across all ``n_repetitions × n_outer_folds`` outer folds.
        std : MetricDict
            Standard deviation across the same population.
        """
        if not self.cfg.enabled:
            return
        import wandb
        from src.logging.log_schema import (
            nested_final_mean_key,
            nested_final_std_key,
        )

        log_data: Dict[str, float] = {}
        for k, v in mean.items():
            log_data[nested_final_mean_key(k)] = v
        for k, v in std.items():
            log_data[nested_final_std_key(k)] = v
        wandb.log(log_data)

    def log_prediction_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_idx: int,
    ) -> None:
        """Log a prediction scatter plot with regression line and data to wandb.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        fold_idx : int
        """
        if not self.cfg.enabled:
            return
        import logging as _logging
        import matplotlib.pyplot as plt
        import wandb

        _log = _logging.getLogger(__name__)

        # Drop NaN / Inf values that would crash matplotlib's axis-limit setter.
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not valid_mask.any():
            _log.warning(
                "log_prediction_scatter fold %d: all predictions are NaN/Inf — "
                "skipping scatter plot.",
                fold_idx,
            )
            return
        if not valid_mask.all():
            n_dropped = int((~valid_mask).sum())
            _log.warning(
                "log_prediction_scatter fold %d: dropping %d NaN/Inf prediction(s) "
                "before plotting.",
                fold_idx, n_dropped,
            )
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        # Linear regression coefficients
        coeffs = np.polyfit(y_true, y_pred, 1)
        slope, intercept = coeffs
        x_line = np.linspace(y_true.min(), y_true.max(), 200)
        y_line = np.polyval(coeffs, x_line)

        # R² of the predictions against ground truth
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="none", label="Subjects")
        ax.plot(
            x_line, y_line, color="red", linewidth=1.5,
            label=f"Fit  slope={slope:.2f}  R²={r2:.3f}",
        )

        # Adaptive limits with a small padding margin
        pad = 0.05
        x_rng = float(y_true.max() - y_true.min()) or 1.0
        y_rng = float(y_pred.max() - y_pred.min()) or 1.0
        ax.set_xlim(float(y_true.min()) - pad * x_rng, float(y_true.max()) + pad * x_rng)
        ax.set_ylim(float(y_pred.min()) - pad * y_rng, float(y_pred.max()) + pad * y_rng)

        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.set_title(f"Predictions vs Ground Truth (Fold {fold_idx})")
        ax.legend(fontsize=8)
        fig.tight_layout()

        # Log scatter plot image
        wandb.log({f"predictions/scatter_fold_{fold_idx}": wandb.Image(fig)})
        plt.close(fig)

        # Log predictions and ground truth as table for data access
        table = wandb.Table(columns=["y_true", "y_pred"])
        for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
            table.add_data(yt, yp)
        wandb.log({f"predictions/data_fold_{fold_idx}": table})
