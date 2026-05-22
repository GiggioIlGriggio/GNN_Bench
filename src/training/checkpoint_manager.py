"""Fold checkpoint save/load manager."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from src.training.fold_barrier import FoldBarrier
from src.training.metrics import MetricDict

if TYPE_CHECKING:
    from src.models.base_model import BrainGNN

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint container
# ---------------------------------------------------------------------------

@dataclass
class FoldCheckpoint:
    """Contents of a saved fold checkpoint.

    Attributes
    ----------
    fold_idx : int
        Fold index.
    best_model_state_dict : dict
        PyTorch model state dict from the best epoch.
    last_model_state_dict : dict
        PyTorch model state dict from the final epoch.
    barrier : FoldBarrier
        Outer-train-fit leakage barrier (ADR-0009).
    best_metrics : MetricDict
        Validation metrics at the best epoch.
    best_epoch : int
        Epoch index of the best checkpoint.
    last_metrics : MetricDict
        Validation metrics at the last epoch.
    last_epoch : int
        Epoch index of the last checkpoint.
    """

    fold_idx: int = 0
    best_model_state_dict: dict = None  # type: ignore[assignment]
    last_model_state_dict: dict = None  # type: ignore[assignment]
    barrier: Optional[FoldBarrier] = None
    best_metrics: MetricDict = None  # type: ignore[assignment]
    best_epoch: int = 0
    last_metrics: MetricDict = None  # type: ignore[assignment]
    last_epoch: int = 0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save and load per-fold checkpoints.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Root directory for all checkpoints.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

    def save_fold_checkpoint(
        self,
        fold_idx: int,
        best_model_state_dict: dict,
        last_model_state_dict: dict,
        barrier: FoldBarrier,
        best_metrics: MetricDict,
        best_epoch: int,
        last_metrics: MetricDict,
        last_epoch: int,
        model_config: Optional[Dict] = None,
        feature_config: Optional[Dict] = None,
    ) -> Path:
        """Save best and last model checkpoints, fold barrier, and metrics for one fold.

        Parameters
        ----------
        fold_idx : int
        best_model_state_dict : dict
            State dict from the epoch with the best validation metric.
        last_model_state_dict : dict
            State dict from the final training epoch.
        barrier : FoldBarrier
            Outer-train-fit leakage barrier (ADR-0009). Persisted as
            ``barrier.pt``.
        best_metrics : MetricDict
        best_epoch : int
        last_metrics : MetricDict
        last_epoch : int
        model_config : Optional[Dict]
            Serialised ModelConfig (for checkpoint self-containment).
        feature_config : Optional[Dict]
            Serialised FeatureConfig (for checkpoint self-containment).

        Returns
        -------
        Path
            Directory where the checkpoint was saved.
        """
        fold_dir = self.get_fold_dir(fold_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)

        torch.save(best_model_state_dict, fold_dir / "model_best.pt")
        torch.save(last_model_state_dict, fold_dir / "model_last.pt")

        barrier.save(fold_dir / "barrier.pt")

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "best": {"metrics": best_metrics, "epoch": best_epoch},
                    "last": {"metrics": last_metrics, "epoch": last_epoch},
                },
                f,
                indent=2,
            )

        if model_config is not None:
            with open(fold_dir / "model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)

        if feature_config is not None:
            with open(fold_dir / "feature_config.json", "w") as f:
                json.dump(feature_config, f, indent=2)

        return fold_dir

    def load_fold_checkpoint(self, fold_idx: int) -> FoldCheckpoint:
        """Load a fold's checkpoint from disk.

        Parameters
        ----------
        fold_idx : int

        Returns
        -------
        FoldCheckpoint

        Raises
        ------
        FileNotFoundError
            If the checkpoint directory for the fold does not exist.
        """
        fold_dir = self.get_fold_dir(fold_idx)
        if not fold_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory for fold {fold_idx} not found: {fold_dir}"
            )

        best_state_dict = torch.load(
            fold_dir / "model_best.pt", map_location="cpu", weights_only=True
        )
        last_state_dict = torch.load(
            fold_dir / "model_last.pt", map_location="cpu", weights_only=True
        )

        barrier_path = fold_dir / "barrier.pt"
        if barrier_path.exists():
            barrier: Optional[FoldBarrier] = FoldBarrier(label_norm_strategy="standard")
            barrier.load_state_dict(
                torch.load(barrier_path, map_location="cpu", weights_only=False)
            )
        else:
            barrier = None

        with open(fold_dir / "metrics.json") as f:
            metrics_data = json.load(f)

        return FoldCheckpoint(
            fold_idx=fold_idx,
            best_model_state_dict=best_state_dict,
            last_model_state_dict=last_state_dict,
            barrier=barrier,
            best_metrics=metrics_data["best"]["metrics"],
            best_epoch=metrics_data["best"]["epoch"],
            last_metrics=metrics_data["last"]["metrics"],
            last_epoch=metrics_data["last"]["epoch"],
        )

    def load_model_for_fold(
        self,
        fold_idx: int,
        model_factory: Callable[[], BrainGNN],
        num_nodes: int = 0,
        variant: str = "best",
    ) -> Tuple[BrainGNN, Optional[FoldBarrier]]:
        """Load a model with its weights and fold barrier for a specific fold.

        Reads ``model_config.json`` and ``feature_config.json`` from the fold
        directory to reconstruct the exact architecture used during training.
        Falls back to *model_factory* when those files are absent (old
        checkpoints without config metadata).

        Parameters
        ----------
        fold_idx : int
        model_factory : Callable
            Fallback factory used when ``model_config.json`` is absent.
        num_nodes : int
            Number of nodes per graph (passed to ``get_model``).
        variant : str
            ``"best"`` or ``"last"``.

        Returns
        -------
        Tuple[BrainGNN, Optional[FoldBarrier]]
        """
        from src.configs.feature_config import FeatureConfig
        from src.configs.model_config import ModelConfig
        from src.models.registry import get_model

        fold_dir = self.get_fold_dir(fold_idx)
        model_cfg_path = fold_dir / "model_config.json"
        feature_cfg_path = fold_dir / "feature_config.json"

        if model_cfg_path.exists() and feature_cfg_path.exists():
            with open(model_cfg_path) as f:
                model_cfg = ModelConfig(**json.load(f))
            with open(feature_cfg_path) as f:
                feature_cfg = FeatureConfig(**json.load(f))
            model = get_model(
                name=model_cfg.name,
                cfg=model_cfg,
                node_feat_dim=feature_cfg.node_feat_dim,
                edge_feat_dim=feature_cfg.edge_feat_dim,
                num_nodes=num_nodes,
            )
        else:
            log.warning(
                "model_config.json / feature_config.json not found in %s — "
                "using model_factory() as fallback.",
                fold_dir,
            )
            model = model_factory()

        weights_file = fold_dir / f"model_{variant}.pt"
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        barrier_path = fold_dir / "barrier.pt"
        if barrier_path.exists():
            barrier: Optional[FoldBarrier] = FoldBarrier(label_norm_strategy="standard")
            barrier.load_state_dict(
                torch.load(barrier_path, map_location="cpu", weights_only=False)
            )
        else:
            barrier = None

        return model, barrier

    # ------------------------------------------------------------------
    # Epoch-level checkpoint helpers
    # ------------------------------------------------------------------

    def get_epoch_ckpt_dir(self, fold_idx: int) -> Path:
        """Return the directory used for per-epoch model snapshots.

        Parameters
        ----------
        fold_idx : int

        Returns
        -------
        Path
            ``<checkpoint_dir>/fold_<fold_idx>/epoch_checkpoints/``
        """
        return self.get_fold_dir(fold_idx) / "epoch_checkpoints"

    def save_epoch_checkpoint(
        self,
        fold_idx: int,
        epoch: int,
        state_dict: dict,
        is_best: bool = False,
    ) -> Path:
        """Save a model snapshot for a single training epoch.

        The file is stored as ``epoch_<EEEE>.pt`` inside
        ``fold_<fold_idx>/epoch_checkpoints/``.  A JSON metadata file
        ``best_epoch.json`` is updated whenever ``is_best`` is ``True``.

        Parameters
        ----------
        fold_idx : int
        epoch : int
            0-based epoch index.
        state_dict : dict
            Model state dict to save.
        is_best : bool
            Whether this epoch achieved the best validation metric so far.

        Returns
        -------
        Path
            Full path of the saved ``.pt`` file.
        """
        epoch_dir = self.get_epoch_ckpt_dir(fold_idx)
        epoch_dir.mkdir(parents=True, exist_ok=True)

        epoch_path = epoch_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state_dict, epoch_path)

        if is_best:
            meta_path = epoch_dir / "best_epoch.json"
            with open(meta_path, "w") as f:
                json.dump({"best_epoch": epoch}, f)

        return epoch_path

    def list_epoch_checkpoints(
        self, fold_idx: int
    ) -> List[Tuple[int, Path]]:
        """List all per-epoch snapshot files for a fold, sorted by epoch.

        Parameters
        ----------
        fold_idx : int

        Returns
        -------
        List[Tuple[int, Path]]
            ``(epoch, path)`` pairs sorted ascending by epoch.
        """
        epoch_dir = self.get_epoch_ckpt_dir(fold_idx)
        if not epoch_dir.exists():
            return []

        entries: List[Tuple[int, Path]] = []
        for p in epoch_dir.glob("epoch_*.pt"):
            try:
                epoch = int(p.stem.split("_")[1])
                entries.append((epoch, p))
            except (IndexError, ValueError):
                continue

        entries.sort(key=lambda x: x[0])
        return entries

    def get_common_epochs(self, n_folds: int) -> List[int]:
        """Return epochs present across **all** folds (intersection).

        Parameters
        ----------
        n_folds : int

        Returns
        -------
        List[int]
            Sorted list of epoch indices available in every fold.
        """
        sets: List[set] = []
        for fold_idx in range(n_folds):
            epochs = {e for e, _ in self.list_epoch_checkpoints(fold_idx)}
            sets.append(epochs)

        if not sets:
            return []

        common = sets[0]
        for s in sets[1:]:
            common = common & s

        return sorted(common)

    def get_fold_dir(self, fold_idx: int) -> Path:
        """Return the path to a fold's checkpoint directory.

        Parameters
        ----------
        fold_idx : int

        Returns
        -------
        Path
        """
        return self.checkpoint_dir / f"fold_{fold_idx}"
