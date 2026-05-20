"""Per-fold label normalisation (prevents data leakage across folds)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np


class LabelNormalizer:
    """Fit on training labels only and transform all splits.

    Serialisable alongside fold checkpoints.

    Parameters
    ----------
    strategy : Literal["standard", "robust", "minmax", "none"]
        Normalisation strategy.
    """

    def __init__(
        self,
        strategy: Literal["standard", "robust", "minmax", "none"] = "standard",
    ) -> None:
        self.strategy = strategy

        # Fitted statistics (populated by fit)
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._median: Optional[float] = None
        self._iqr: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None

    def fit(self, y_train: np.ndarray) -> None:
        """Compute normalisation statistics from training labels.

        Parameters
        ----------
        y_train : np.ndarray
            Training labels (1-D).
        """
        if self.strategy == "standard":
            self._mean = float(np.mean(y_train))
            self._std = float(np.std(y_train))
        elif self.strategy == "robust":
            self._median = float(np.median(y_train))
            q75, q25 = np.percentile(y_train, [75, 25])
            self._iqr = float(q75 - q25)
        elif self.strategy == "minmax":
            self._min = float(np.min(y_train))
            self._max = float(np.max(y_train))
        # else: "none" — no-op

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply normalisation.

        Parameters
        ----------
        y : np.ndarray

        Returns
        -------
        np.ndarray
            Normalised labels.
        """
        if self.strategy == "standard":
            return (y - self._mean) / (self._std + 1e-8)
        if self.strategy == "robust":
            return (y - self._median) / (self._iqr + 1e-8)
        if self.strategy == "minmax":
            return (y - self._min) / (self._max - self._min + 1e-8)
        return y.copy()

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Reverse normalisation (map back to original scale).

        Parameters
        ----------
        y : np.ndarray

        Returns
        -------
        np.ndarray
        """
        if self.strategy == "standard":
            return y * (self._std + 1e-8) + self._mean
        if self.strategy == "robust":
            return y * (self._iqr + 1e-8) + self._median
        if self.strategy == "minmax":
            return y * (self._max - self._min + 1e-8) + self._min
        return y.copy()

    def fit_transform(self, y_train: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform.

        Parameters
        ----------
        y_train : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self.fit(y_train)
        return self.transform(y_train)

    def save(self, path: Path) -> None:
        """Serialise normaliser state to disk.

        Parameters
        ----------
        path : Path
            Output file path (e.g. ``checkpoint_dir/fold_0/normalizer.json``).
        """
        import json
        state = {
            "strategy": self.strategy,
            "mean": self._mean,
            "std": self._std,
            "median": self._median,
            "iqr": self._iqr,
            "min": self._min,
            "max": self._max,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "LabelNormalizer":
        """Deserialise a normaliser from disk.

        Parameters
        ----------
        path : Path
            Path to previously saved normaliser.

        Returns
        -------
        LabelNormalizer
        """
        import json
        with open(path, "r") as f:
            state = json.load(f)
        obj = cls(strategy=state["strategy"])
        obj._mean = state.get("mean")
        obj._std = state.get("std")
        obj._median = state.get("median")
        obj._iqr = state.get("iqr")
        obj._min = state.get("min")
        obj._max = state.get("max")
        return obj
