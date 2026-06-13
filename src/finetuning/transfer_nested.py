"""Inject per-(rep,fold) age-pretrained backbones into a nested-CV VWM run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

log = logging.getLogger(__name__)


class SourceBackboneProvider:
    """Serve source (age) backbone weights for each outer fold, alignment-checked.

    Parameters
    ----------
    source_root : str | Path
        Root of a completed source nested run: holds ``fold_indices.json`` and
        ``rep_<R>/fold_<F>/model_<variant>.pt``.
    variant : str
        ``"best"`` or ``"last"``.
    """

    def __init__(self, source_root, variant: str = "best") -> None:
        # Validate the checkpoint variant first: the CheckpointManager only ever
        # writes model_best.pt / model_last.pt, so fail at construction rather
        # than deep in a run with a cryptic FileNotFoundError for model_*.pt.
        if variant not in ("best", "last"):
            raise ValueError(f"variant must be 'best' or 'last', got {variant!r}")
        self.source_root = Path(source_root)
        self.variant = variant
        manifest_path = self.source_root / "fold_indices.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Source run has no fold_indices.json at {manifest_path}; "
                "regenerate the source run with the manifest-writing nested CV."
            )
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("folds"):
            raise ValueError(
                f"Source manifest at {manifest_path} has no 'folds' entries; the "
                "source run did not complete or wrote a malformed manifest."
            )
        self._folds: Dict[Tuple[int, int], Dict[str, List[int]]] = {
            (rec["rep"], rec["fold"]): {
                "train_val_idx": sorted(rec["train_val_idx"]),
                "test_idx": sorted(rec["test_idx"]),
            }
            for rec in manifest["folds"]
        }

    def assert_aligned(
        self, *, rep: int, fold: int, train_val_idx, test_idx
    ) -> None:
        """Raise unless the consuming run's fold (rep,fold) matches the source."""
        key = (rep, fold)
        if key not in self._folds:
            raise ValueError(
                f"fold-index mismatch: source manifest has no (rep={rep}, fold={fold})"
            )
        src = self._folds[key]
        if (
            sorted(map(int, train_val_idx)) != src["train_val_idx"]
            or sorted(map(int, test_idx)) != src["test_idx"]
        ):
            raise ValueError(
                f"fold-index mismatch at (rep={rep}, fold={fold}): the VWM run's "
                "outer split differs from the age-source run's. Stratification is "
                "not aligned — refusing to transfer (would leak test subjects)."
            )

    def state_dict_for(self, *, rep: int, fold: int) -> dict:
        """Load the source backbone weights for one outer fold."""
        path = self.source_root / f"rep_{rep}" / f"fold_{fold}" / f"model_{self.variant}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"No source checkpoint for (rep={rep}, fold={fold}) at {path} "
                f"(variant={self.variant!r}); the source run may be incomplete."
            )
        # weights_only=True assumes source checkpoints are pure tensor state-dicts.
        return torch.load(path, map_location="cpu", weights_only=True)
