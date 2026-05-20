"""Per-node z-score normalizer for GLM node-feature columns.

Operates directly on the ``data.x`` tensor of PyG graphs.  Fits per-(node,
column) statistics on training subjects only, then applies the same
transform to validation and test subjects to prevent data leakage.

Usage inside cross-validation::

    normalizer = GLMFeatureNormalizer(col_start=2, col_end=3)
    normalizer.fit(train_graphs)
    normalizer.transform(train_graphs)   # in-place on copies
    normalizer.transform(val_graphs)
    normalizer.transform(test_graphs)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch_geometric.data

log = logging.getLogger(__name__)


class GLMFeatureNormalizer:
    """Node-wise z-score normalizer for GLM feature columns in ``data.x``.

    For each brain parcel (node) and each GLM feature column, the normalizer
    computes mean and standard deviation across **training subjects** and
    applies the transform ``(x - mean) / std`` to all splits.

    Parameters
    ----------
    col_start : int
        First column index (inclusive) of the GLM features in ``data.x``.
    col_end : int
        Last column index (exclusive) of the GLM features in ``data.x``.

    Attributes
    ----------
    mean_ : torch.Tensor
        Shape ``[num_nodes, glm_dim]`` — per-node, per-column training mean.
    std_ : torch.Tensor
        Shape ``[num_nodes, glm_dim]`` — per-node, per-column training std.
    """

    def __init__(self, col_start: int, col_end: int) -> None:
        self.col_start = col_start
        self.col_end = col_end
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        graphs: List[torch_geometric.data.Data],
    ) -> "GLMFeatureNormalizer":
        """Compute per-(node, column) mean and std from training graphs.

        Parameters
        ----------
        graphs : List[Data]
            Graphs belonging to the **training set** for this fold.

        Returns
        -------
        GLMFeatureNormalizer
            ``self`` for method chaining.

        Raises
        ------
        ValueError
            If no graphs are provided.
        """
        if not graphs:
            raise ValueError("Cannot fit GLMFeatureNormalizer on an empty list.")

        # Stack GLM columns: [n_subjects, num_nodes, glm_dim]
        slices = torch.stack(
            [g.x[:, self.col_start : self.col_end] for g in graphs],
            dim=0,
        )

        self.mean_ = slices.mean(dim=0)  # [num_nodes, glm_dim]
        std = slices.std(dim=0)           # [num_nodes, glm_dim]
        std[std == 0] = 1.0               # guard against zero variance
        self.std_ = std

        self._fitted = True
        log.debug(
            "GLMFeatureNormalizer fitted on %d subjects, columns [%d:%d).",
            len(graphs),
            self.col_start,
            self.col_end,
        )
        return self

    def transform(
        self,
        graphs: List[torch_geometric.data.Data],
    ) -> None:
        """Apply per-node z-scoring to GLM columns of ``data.x`` **in-place**.

        Parameters
        ----------
        graphs : List[Data]
            Graphs to normalise (should already be shallow copies so
            originals are not modified).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "GLMFeatureNormalizer.transform() called before fit()."
            )
        for g in graphs:
            g.x[:, self.col_start : self.col_end] = (
                (g.x[:, self.col_start : self.col_end] - self.mean_) / self.std_
            )

    def fit_transform(
        self,
        graphs: List[torch_geometric.data.Data],
    ) -> None:
        """Fit on the given graphs and transform them in one call.

        Parameters
        ----------
        graphs : List[Data]
            Graphs in the **training set**.
        """
        self.fit(graphs)
        self.transform(graphs)
