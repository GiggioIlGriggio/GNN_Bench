"""Pure-PyTorch training loop orchestrator."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.configs.trainer_config import TrainerConfig
from src.models.base_model import BrainGNN
from src.training.metrics import MetricDict, compute_metrics

if TYPE_CHECKING:
    from src.logging.wandb_logger import WandbLogger


# ---------------------------------------------------------------------------
# Module-level console
# ---------------------------------------------------------------------------

_console = Console(highlight=False)


def _fmt(val_loss: float, metrics: "MetricDict", star: bool = False) -> str:  # noqa: F821
    """Format val loss and metrics into a compact string for the progress bar."""
    parts = [
        f"loss [bold]{val_loss:.3f}[/]  "
        f"MAE [bold]{metrics.get('mae', 0):.3f}[/]  "
        f"R² [bold]{metrics.get('r2', 0):.3f}[/]  "
        f"r [bold]{metrics.get('pearson_r', 0):.3f}[/]"
    ]
    if star:
        parts.append(" [yellow]★[/]")
    return "".join(parts)


def _fmt_train(loss: float, metrics: "MetricDict") -> str:  # noqa: F821
    """Format train loss and metrics compactly (no r)."""
    return (
        f"loss [bold]{loss:.3f}[/]  "
        f"MAE [bold]{metrics.get('mae', 0):.3f}[/]"
    )


def _print_fold_summary(
    fold_idx: int, best_epoch: int, metrics: "MetricDict"
) -> None:  # noqa: F821
    """Print a summary panel at the end of a fold."""
    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_column(style="dim")
    table.add_column(style="bold white")
    table.add_row(
        "MAE",  f"{metrics.get('mae', 0):.4f}",
        "RMSE", f"{metrics.get('rmse', 0):.4f}",
    )
    table.add_row(
        "R²",   f"{metrics.get('r2', 0):.4f}",
        "r",    f"{metrics.get('pearson_r', 0):.4f}",
    )
    _console.print(
        Panel(
            table,
            title=f"[bold green]Fold {fold_idx}[/]  [dim]best @ epoch {best_epoch}[/]",
            border_style="green",
            padding=(0, 2),
        )
    )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Outcome of a single fold's training run.

    Attributes
    ----------
    best_epoch : int
        Epoch with the best validation metric.
    best_val_metrics : MetricDict
        Metrics at the best epoch.
    history : Dict[str, List[float]]
        Full training history keyed by metric name.
    best_model_state_dict : dict
        Model state dict from the best epoch.
    last_model_state_dict : dict
        Model state dict from the final epoch.
    last_epoch : int
        Index of the last epoch that ran.
    last_val_metrics : MetricDict
        Validation metrics at the last epoch.
    """

    best_epoch: int = 0
    best_val_metrics: MetricDict = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)
    best_model_state_dict: dict = field(default_factory=dict)
    last_model_state_dict: dict = field(default_factory=dict)
    last_epoch: int = 0
    last_val_metrics: MetricDict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Pure-PyTorch training loop.

    The trainer **owns** the optimizer and scheduler — both are constructed
    internally from :class:`TrainerConfig`.

    Parameters
    ----------
    cfg : TrainerConfig
        Training hyperparameters.
    logger : WandbLogger
        Experiment logger (may be a no-op wrapper if logging disabled).
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        logger: WandbLogger,
        param_groups: Optional[List[Dict]] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self._param_groups = param_groups

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: BrainGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        inverse_transform: Callable[[np.ndarray], np.ndarray],
        fold_idx: int,
        on_epoch_end_callback: Optional[Callable[[int, dict, bool], None]] = None,
    ) -> TrainResult:
        """Train a model for one fold.

        Runs the full training loop with early stopping.  The optimizer and
        scheduler are created fresh at the start of each call.

        Parameters
        ----------
        model : BrainGNN
            Model to train (modified in-place).
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        inverse_transform : Callable[[np.ndarray], np.ndarray]
            Maps normalised predictions/targets back to the original
            label scale. Pass ``barrier.inverse_transform_labels`` from
            the CV path (ADR-0009).
        fold_idx : int
            Current fold index (for logging).
        on_epoch_end_callback : Optional[Callable[[int, dict, bool], None]]
            If provided, called at the end of every epoch with
            ``(epoch, state_dict_copy, is_best)``.  Use this to implement
            per-epoch model snapshotting without coupling the trainer to the
            checkpoint manager.

        Returns
        -------
        TrainResult
            Best metrics, best epoch, and full training history.
        """
        device = next(model.parameters()).device
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        best_val_metric = float("inf")
        best_epoch = 0
        best_val_metrics: MetricDict = {}
        best_model_state_dict: dict = {}
        patience_counter = 0
        history: Dict[str, List[float]] = {}
        last_epoch = 0
        last_val_metrics: MetricDict = {}

        monitor_key = self.cfg.early_stopping_metric.replace("val_", "")

        _console.rule(f"[bold cyan]Fold {fold_idx}[/]", style="cyan dim")

        with Progress(
            SpinnerColumn(style="cyan"),
            MofNCompleteColumn(),
            BarColumn(bar_width=28, style="cyan", complete_style="bold cyan"),
            TextColumn("[dim]tr[/] {task.fields[train_metrics]}"),
            TextColumn("  [dim]val[/] {task.fields[val_metrics]}"),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "epoch",
                total=self.cfg.epochs,
                train_metrics="—",
                val_metrics="—",
            )

            for epoch in range(self.cfg.epochs):
                train_loss, train_preds_norm, train_targets_norm, aux_losses = (
                    self._train_one_epoch(model, train_loader, optimizer)
                )
                # Compute train metrics from the predictions already made during
                # the forward pass — no second pass over the training data.
                train_preds = inverse_transform(train_preds_norm)
                train_targets = inverse_transform(train_targets_norm)
                train_metrics = compute_metrics(train_targets, train_preds)

                val_metrics = self.evaluate(model, val_loader, inverse_transform, "val")

                # Compute validation loss for display
                model.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        pred = model(batch).squeeze(-1)
                        loss = F.mse_loss(pred, batch.y.view(-1))
                        val_loss += loss.item() * batch.num_graphs
                        val_samples += batch.num_graphs
                val_loss /= max(val_samples, 1)
                model.train()

                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics.get(monitor_key, val_metrics["mae"]))
                    else:
                        scheduler.step()

                # Record history
                history.setdefault("train_loss", []).append(train_loss)
                for k, v in train_metrics.items():
                    history.setdefault(f"train_{k}", []).append(v)
                for k, v in val_metrics.items():
                    history.setdefault(f"val_{k}", []).append(v)

                # Log metrics — train and val logged under separate split paths
                # so the wandb UI shows fold_N/train/* and fold_N/val/* cleanly.
                try:
                    self.logger.log_fold_metrics(
                        fold_idx,
                        {"loss": train_loss, **train_metrics, **aux_losses},
                        split="train",
                        epoch=epoch,
                    )
                    self.logger.log_fold_metrics(
                        fold_idx,
                        val_metrics,
                        split="val",
                        epoch=epoch,
                    )
                except (NotImplementedError, Exception):
                    raise RuntimeError("Logging failed — check your logger implementation") from None

                # Early stopping
                monitored = val_metrics.get(monitor_key, val_metrics["mae"])
                is_best = monitored < best_val_metric - self.cfg.early_stopping_min_delta
                if is_best:
                    best_val_metric = monitored
                    best_epoch = epoch
                    best_val_metrics = dict(val_metrics)
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                last_epoch = epoch
                last_val_metrics = dict(val_metrics)

                # Per-epoch snapshot callback (used by CrossValidator for epoch checkpointing).
                # Pass a copy of the current state dict so training can continue safely.
                if on_epoch_end_callback is not None:
                    on_epoch_end_callback(epoch, copy.deepcopy(model.state_dict()), is_best)

                progress.update(
                    task,
                    advance=1,
                    train_metrics=_fmt_train(train_loss, train_metrics),
                    val_metrics=_fmt(val_loss, val_metrics, star=is_best),
                )

                if patience_counter >= self.cfg.early_stopping_patience:
                    progress.update(task, val_metrics=_fmt(val_loss, val_metrics) + "  [yellow dim]early stop[/]")
                    break

        last_model_state_dict = copy.deepcopy(model.state_dict())

        _print_fold_summary(fold_idx, best_epoch, best_val_metrics)
        return TrainResult(
            best_epoch=best_epoch,
            best_val_metrics=best_val_metrics,
            history=history,
            best_model_state_dict=best_model_state_dict,
            last_model_state_dict=last_model_state_dict,
            last_epoch=last_epoch,
            last_val_metrics=last_val_metrics,
        )

    def predict(
        self,
        model: BrainGNN,
        loader: DataLoader,
        inverse_transform: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference and return ground-truth and predictions in the original label scale.

        Parameters
        ----------
        model : BrainGNN
            Trained model (set to eval mode internally).
        loader : DataLoader
            Data loader for the split.
        inverse_transform : Callable[[np.ndarray], np.ndarray]
            Maps normalised predictions/targets back to the original
            label scale. Pass ``barrier.inverse_transform_labels``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(y_true, y_pred)`` both in the original (un-normalised) scale.
        """
        model.eval()
        device = next(model.parameters()).device
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch).squeeze(-1)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch.y.view(-1).cpu().numpy())

        y_pred = inverse_transform(np.concatenate(all_preds))
        y_true = inverse_transform(np.concatenate(all_targets))
        return y_true, y_pred

    def evaluate(
        self,
        model: BrainGNN,
        loader: DataLoader,
        inverse_transform: Callable[[np.ndarray], np.ndarray],
        split: Literal["val", "test"],
    ) -> MetricDict:
        """Evaluate the model on a data split.

        Parameters
        ----------
        model : BrainGNN
            Trained model (set to eval mode internally).
        loader : DataLoader
            Data loader for the split.
        inverse_transform : Callable[[np.ndarray], np.ndarray]
            Maps normalised predictions back to the original label scale.
        split : Literal["val", "test"]
            Split name for logging.

        Returns
        -------
        MetricDict
            Computed metrics (MAE, RMSE, R², Pearson r).
        """
        y_true, y_pred = self.predict(model, loader, inverse_transform)
        return compute_metrics(y_true, y_pred)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self, model: BrainGNN) -> torch.optim.Optimizer:
        """Create the optimizer from config.

        Parameters
        ----------
        model : BrainGNN

        Returns
        -------
        torch.optim.Optimizer
        """
        opt = self.cfg.optimizer.lower()
        params = self._param_groups if self._param_groups is not None else model.parameters()
        if opt == "adam":
            return torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if opt == "adamw":
            return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if opt == "sgd":
            return torch.optim.SGD(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create the LR scheduler from config.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer

        Returns
        -------
        Optional[_LRScheduler]
            ``None`` if ``self.cfg.scheduler`` is ``None`` or ``"none"``.
        """
        sched = (self.cfg.scheduler or "none").lower()
        if sched == "none":
            return None
        params = dict(self.cfg.scheduler_params)
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        if sched == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, **params)
        if sched == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

    def _train_one_epoch(
        self,
        model: BrainGNN,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
        """Run one training epoch.

        Accumulates predictions and targets from the forward passes that are
        already performed to compute the loss, so no second pass is needed to
        derive training metrics.

        Parameters
        ----------
        model : BrainGNN
        loader : DataLoader
        optimizer : torch.optim.Optimizer

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]
            ``(avg_loss, preds_normalised, targets_normalised, aux_loss_avgs)``
            where ``aux_loss_avgs`` maps each auxiliary loss name to its
            sample-averaged value (empty dict for models without aux losses).
        """
        model.train()
        device = next(model.parameters()).device
        total_loss = 0.0
        total_samples = 0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        aux_accum: Dict[str, float] = {}

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).squeeze(-1)
            loss = F.mse_loss(pred, batch.y.view(-1))

            aux = model.auxiliary_loss()
            if aux is not None:
                for name, term in aux.items():
                    loss = loss + term
                    aux_accum[name] = (
                        aux_accum.get(name, 0.0) + term.item() * batch.num_graphs
                    )

            loss.backward()
            if self.cfg.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.gradient_clip_val)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(batch.y.view(-1).cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)
        aux_avgs = {k: v / max(total_samples, 1) for k, v in aux_accum.items()}
        return avg_loss, np.concatenate(all_preds), np.concatenate(all_targets), aux_avgs
