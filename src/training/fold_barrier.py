"""Per-fold leakage-protection coordinator (ADR-0009).

``FoldBarrier`` owns the three per-fold-fit transformers — the
composite ``LabelBuilder`` (when configured), the ``LabelNormalizer``,
and the ``GLMFeatureNormalizer`` (when GLM columns are present) —
as a single fitted bundle for one outer fold's train pool. It does
not know about splits, batch sizes, or DataLoader construction.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric.data

from src.datasets.label_builder import LabelBuilder
from src.training.glm_normalizer import GLMFeatureNormalizer
from src.training.label_normalizer import LabelNormalizer


class FoldBarrier:
    """The per-fold leakage barrier. Fit once on outer-train; transform anything consistent.

    Parameters
    ----------
    label_norm_strategy : Literal["standard", "robust", "minmax", "none"]
        Strategy passed to the inner :class:`LabelNormalizer`.
    glm_col_range : Optional[Tuple[int, int]]
        ``(col_start, col_end)`` of the GLM feature columns in
        ``data.x``. When ``None`` (or when ``glm_normalize`` is
        ``False`` at fit-time), the GLM step is a no-op.
    glm_normalize : bool
        Whether to apply GLM normalisation when ``glm_col_range`` is set.
    label_builder : Optional[LabelBuilder]
        Composite-label builder. When provided, ``fit`` and
        ``transform_labels`` expect a ``pd.DataFrame`` of components
        rather than a 1-D label vector.
    """

    def __init__(
        self,
        *,
        label_norm_strategy: Literal["standard", "robust", "minmax", "none"] = "standard",
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        label_builder: Optional[LabelBuilder] = None,
    ) -> None:
        self._label_norm = LabelNormalizer(strategy=label_norm_strategy)
        self._label_builder = label_builder
        self._glm: Optional[GLMFeatureNormalizer] = None
        if glm_col_range is not None and glm_normalize:
            col_start, col_end = glm_col_range
            self._glm = GLMFeatureNormalizer(col_start, col_end)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_graphs: List[torch_geometric.data.Data],
        train_labels_or_components: Union[np.ndarray, pd.DataFrame],
    ) -> "FoldBarrier":
        """Fit all configured transformers on the outer-train pool.

        Parameters
        ----------
        train_graphs : List[Data]
            Training-split graphs (used to fit the GLM normaliser; not
            mutated).
        train_labels_or_components : np.ndarray | pd.DataFrame
            A 1-D label vector in non-composite mode, a DataFrame slice
            of label components in composite mode.

        Returns
        -------
        FoldBarrier
        """
        if self._label_builder is not None:
            if not isinstance(train_labels_or_components, pd.DataFrame):
                raise TypeError(
                    "Composite mode expects a pd.DataFrame of label components."
                )
            y_train = self._label_builder.fit_transform(train_labels_or_components)
        else:
            y_train = np.asarray(train_labels_or_components, dtype=float)

        self._label_norm.fit(y_train)

        if self._glm is not None:
            self._glm.fit(train_graphs)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform_labels(
        self,
        labels_or_components: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Apply composite construction (if configured) followed by z-scoring.

        Parameters
        ----------
        labels_or_components : np.ndarray | pd.DataFrame

        Returns
        -------
        np.ndarray
        """
        self._require_fitted()
        if self._label_builder is not None:
            if not isinstance(labels_or_components, pd.DataFrame):
                raise TypeError(
                    "Composite mode expects a pd.DataFrame of label components."
                )
            y = self._label_builder.transform(labels_or_components)
        else:
            y = np.asarray(labels_or_components, dtype=float)
        return self._label_norm.transform(y)

    def inverse_transform_labels(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalise predictions back to the original target scale.

        Parameters
        ----------
        y_norm : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self._require_fitted()
        return self._label_norm.inverse_transform(y_norm)

    def transform_graphs(
        self,
        graphs: List[torch_geometric.data.Data],
    ) -> List[torch_geometric.data.Data]:
        """Return new graphs with the GLM slice z-scored on the fit pool.

        The returned graphs are shallow copies whose ``x`` tensor is
        replaced by a fresh tensor (cloned slice on the GLM range,
        original tensor elsewhere). The input graphs and the dataset
        they reference are never mutated.

        Parameters
        ----------
        graphs : List[Data]

        Returns
        -------
        List[Data]
        """
        self._require_fitted()
        if self._glm is None:
            return [copy.copy(g) for g in graphs]

        col_start, col_end = self._glm.col_start, self._glm.col_end
        out: List[torch_geometric.data.Data] = []
        for g in graphs:
            new_g = copy.copy(g)
            new_x = g.x.clone()
            new_x[:, col_start:col_end] = (
                (new_x[:, col_start:col_end] - self._glm.mean_) / self._glm.std_
            )
            new_g.x = new_x
            out.append(new_g)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Aggregate state of the three transformers.

        Returns
        -------
        dict
        """
        return {
            "fitted": self._fitted,
            "label_norm": self._label_norm.state_dict(),
            "label_builder": (
                self._label_builder.state_dict()
                if self._label_builder is not None else None
            ),
            "glm": self._glm.state_dict() if self._glm is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore aggregated state.

        Parameters
        ----------
        state : dict
        """
        self._fitted = bool(state.get("fitted", False))
        self._label_norm.load_state_dict(state["label_norm"])
        lb_state = state.get("label_builder")
        if lb_state is not None and self._label_builder is not None:
            self._label_builder.load_state_dict(lb_state)
        glm_state = state.get("glm")
        if glm_state is not None:
            if self._glm is None:
                self._glm = GLMFeatureNormalizer(
                    int(glm_state["col_start"]), int(glm_state["col_end"]),
                )
            self._glm.load_state_dict(glm_state)

    def save(self, path: Path) -> None:
        """Persist the barrier's state to ``path`` via ``torch.save``.

        Parameters
        ----------
        path : Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        label_norm_strategy: Literal["standard", "robust", "minmax", "none"] = "standard",
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        label_builder: Optional[LabelBuilder] = None,
    ) -> "FoldBarrier":
        """Reconstruct a barrier from a saved state-dict.

        The transformer instances are reconstructed from the caller-
        supplied configuration; the fitted statistics come from disk.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            label_norm_strategy=label_norm_strategy,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
            label_builder=label_builder,
        )
        obj.load_state_dict(state)
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "FoldBarrier.transform_*/inverse_transform_* called before fit()."
            )
