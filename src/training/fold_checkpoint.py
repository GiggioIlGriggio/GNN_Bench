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
class FoldBundle:
    """Snapshot of one fold's persisted artifacts, loaded together.

    Returned by ``FoldCheckpoint.load_bundle()``. Use the typed
    accessors on :class:`FoldCheckpoint` when you only need part of
    the bundle (model weights, the barrier, metrics).

    Attributes
    ----------
    fold_idx : int
        Fold index.
    best_model_state_dict : dict
        PyTorch model state dict from the best epoch.
    last_model_state_dict : dict
        PyTorch model state dict from the final epoch.
    barrier : Optional[FoldBarrier]
        Outer-train-fit leakage barrier (ADR-0009); ``None`` when
        ``barrier.pt`` is absent.
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
# Per-fold read view (ADR-0009)
# ---------------------------------------------------------------------------

class FoldCheckpoint:
    """The single read path for one outer fold's persisted artifacts.

    Construct from a fold directory (or via :meth:`for_fold` from a root
    and a fold index), then call typed accessors to load only what you
    need:

    - :meth:`load_state_dict` — raw model weights, no model construction
    - :meth:`load_model` — reconstructs the model from saved configs and
      loads weights
    - :meth:`load_barrier` — reconstructs the :class:`FoldBarrier` from
      ``barrier.pt`` (returns ``None`` when the file is absent)
    - :meth:`load_metrics` — reads ``metrics.json``
    - :meth:`load_bundle` — loads everything into a :class:`FoldBundle`

    The on-disk layout is documented under
    :ref:`Checkpoint layout <CONTEXT.md#checkpoint-layout>`.

    Parameters
    ----------
    fold_dir : str | Path
        Path to the fold directory (e.g.
        ``checkpoints/rep_0/fold_3/``).
    """

    def __init__(self, fold_dir: str | Path) -> None:
        self.fold_dir = Path(fold_dir)

    @classmethod
    def for_fold(cls, root: str | Path, fold_idx: int) -> "FoldCheckpoint":
        """Build the view for fold *fold_idx* inside *root*.

        Parameters
        ----------
        root : str | Path
            Parent directory that holds ``fold_<K>/`` subdirectories.
        fold_idx : int

        Returns
        -------
        FoldCheckpoint
        """
        return cls(Path(root) / f"fold_{fold_idx}")

    def load_state_dict(self, *, variant: str = "best") -> dict:
        """Load raw model weights from ``model_<variant>.pt``.

        Parameters
        ----------
        variant : str
            ``"best"`` or ``"last"``.

        Returns
        -------
        dict
            Model state dict (CPU tensors).
        """
        if variant not in ("best", "last"):
            raise ValueError(
                f"variant must be 'best' or 'last', got {variant!r}"
            )
        path = self.fold_dir / f"model_{variant}.pt"
        return torch.load(path, map_location="cpu", weights_only=True)

    def load_model(
        self,
        *,
        model_factory: Optional[Callable[[], "BrainGNN"]] = None,
        num_nodes: int = 0,
        variant: str = "best",
    ) -> "BrainGNN":
        """Reconstruct the model with weights loaded.

        Reads ``model_config.json`` and ``feature_config.json`` from
        the fold directory to rebuild the exact architecture used
        during training. Falls back to *model_factory* when either
        config file is absent (old checkpoints without config
        metadata).

        Parameters
        ----------
        model_factory : Optional[Callable[[], BrainGNN]]
            Used only when the saved configs are absent. Required in
            that case; ``None`` then raises ``FileNotFoundError``.
        num_nodes : int
            Number of nodes per graph (passed to ``get_model`` when
            rebuilding from configs).
        variant : str
            ``"best"`` or ``"last"``.

        Returns
        -------
        BrainGNN
        """
        from src.configs.feature_config import FeatureConfig
        from src.configs.model_config import ModelConfig
        from src.models.registry import get_model

        model_cfg_path = self.fold_dir / "model_config.json"
        feature_cfg_path = self.fold_dir / "feature_config.json"

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
        elif model_factory is not None:
            log.warning(
                "model_config.json / feature_config.json not found in %s — "
                "using model_factory() as fallback.",
                self.fold_dir,
            )
            model = model_factory()
        else:
            raise FileNotFoundError(
                f"{self.fold_dir} has no model_config.json/feature_config.json "
                "and no model_factory was provided; cannot reconstruct model."
            )

        model.load_state_dict(self.load_state_dict(variant=variant))
        return model

    def load_barrier(self) -> Optional[FoldBarrier]:
        """Reload the per-fold leakage barrier from ``barrier.pt``.

        Returns ``None`` when ``barrier.pt`` is absent — typical for old
        checkpoints predating ADR-0009.

        The barrier reconstructs its transformers from the persisted
        state alone: the GLM substate carries ``(col_start, col_end)``
        and the label-norm substate carries its strategy. Composite-
        mode labels (``LabelBuilder``-fit state) are NOT round-tripped
        — callers that need ``transform_labels`` on composite-mode
        folds must reconstruct the ``LabelBuilder`` themselves and call
        :meth:`FoldBarrier.load` directly with ``label_builder=...``.

        Returns
        -------
        Optional[FoldBarrier]
        """
        path = self.fold_dir / "barrier.pt"
        if not path.exists():
            return None
        return FoldBarrier.load(path)

    def load_metrics(self) -> dict:
        """Return the parsed contents of ``metrics.json``.

        Returns
        -------
        dict
            Top-level keys: ``"best"`` and ``"last"``, each mapping to
            a ``{"metrics": MetricDict, "epoch": int}`` payload.
        """
        with open(self.fold_dir / "metrics.json") as f:
            return json.load(f)

    def load_bundle(self) -> "FoldBundle":
        """Load every artifact in the fold directory into a FoldBundle.

        Returns
        -------
        FoldBundle
        """
        metrics = self.load_metrics()
        return FoldBundle(
            fold_idx=self._fold_idx_from_dir(),
            best_model_state_dict=self.load_state_dict(variant="best"),
            last_model_state_dict=self.load_state_dict(variant="last"),
            barrier=self.load_barrier(),
            best_metrics=metrics["best"]["metrics"],
            best_epoch=metrics["best"]["epoch"],
            last_metrics=metrics["last"]["metrics"],
            last_epoch=metrics["last"]["epoch"],
        )

    def _fold_idx_from_dir(self) -> int:
        """Parse the fold index from the directory's basename."""
        name = self.fold_dir.name
        if not name.startswith("fold_"):
            raise ValueError(
                f"fold directory name must start with 'fold_', got "
                f"{name!r} (full path: {self.fold_dir})"
            )
        try:
            return int(name[len("fold_"):])
        except ValueError as e:
            raise ValueError(
                f"fold directory name must end with an integer index, "
                f"got {name!r}"
            ) from e


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
