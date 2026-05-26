"""Label extraction and composite label construction.

Design
------
``LabelBuilder`` exposes two interfaces:

1. **One-shot** (``build``): stateless, fetches and collapses labels in a single
   call.  Safe for single-column targets where there is no data-leakage risk.
   Do *not* use this with composite targets inside cross-validation.

2. **Stateful sklearn-style** (``get_raw_components`` → ``fit`` → ``transform``):
   separates component extraction from fitting so that normalisation statistics
   and composite operations (e.g. PCA) are computed on training data only.
   ``CrossValidator`` uses this path when ``label_builder`` and
   ``label_components`` are provided.

Composite operations are delegated to registered :class:`BaseComposite`
subclasses via the composite registry.  See
``src/datasets/composite_registry.py`` for built-in composites and
instructions for adding new ones.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.configs.label_config import LabelConfig
from src.datasets.composite_registry import BaseComposite, get_composite


class LabelBuilder:
    """Extracts and constructs prediction targets from subject metadata.

    Supports single-column lookup and any registered composite method.
    For composite modes the class exposes a stateful sklearn-style interface
    (``fit`` / ``transform`` / ``fit_transform``) so that component normalisation
    and stateful composites (e.g. PCA) are fitted on training data only,
    preventing data leakage across cross-validation folds.

    Parameters
    ----------
    cfg : LabelConfig
        Label configuration (target column, composite definition, etc.).
    """

    def __init__(self, cfg: LabelConfig) -> None:
        self.cfg = cfg
        self._label_names: List[str] = []

        # Fitted state — populated by fit(), used by transform()
        self._component_means: Optional[np.ndarray] = None   # shape [K]
        self._component_stds: Optional[np.ndarray] = None    # shape [K]

        # Composite operation instance (created once, reused)
        self._composite_op: Optional[BaseComposite] = None
        if cfg.is_composite:
            self._composite_op = get_composite(
                cfg.composite_method, cfg.composite_params
            )

    # ------------------------------------------------------------------
    # Raw component extraction (no fitting, no combining)
    # ------------------------------------------------------------------

    def get_raw_components(
        self,
        subject_ids: List[str],
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the raw component columns for the given subjects.

        For a single ``target`` column returns a single-column DataFrame.
        For a composite returns all ``composite_columns`` without processing.
        The index of the returned DataFrame is the provided ``subject_ids``.

        Parameters
        ----------
        subject_ids : List[str]
            Subject identifiers in ``sub-XXX`` format.
        metadata : pd.DataFrame
            Full metadata table with an ID column mapping to subject IDs.

        Returns
        -------
        pd.DataFrame
            Shape ``[len(subject_ids), K]``, where K is 1 for a single-column
            target or the number of composite columns.

        Raises
        ------
        ValueError
            If any subject is missing from metadata, or the config specifies
            neither ``target`` nor ``composite_columns``.
        """
        if self.cfg.target is None and self.cfg.composite_columns is None:
            raise ValueError(
                "LabelConfig must specify either 'target' or 'composite_columns'."
            )

        cols = (
            list(self.cfg.composite_columns)
            if self.cfg.is_composite
            else [self.cfg.target]
        )

        # Build integer-ID indexed lookup from the metadata DataFrame
        meta_indexed = metadata.set_index(self.cfg.id_column)

        rows = []
        for sid in subject_ids:
            # Handle T0_sub-001 → sub-001 → 1, or sub-001 → 1
            if "_sub-" in sid:
                sub_part = sid.split("_sub-")[1]
            elif sid.startswith("sub-"):
                sub_part = sid[4:]
            else:
                sub_part = sid
            try:
                int_id = int(sub_part)
            except ValueError:
                int_id = sub_part  # fall back to string lookup

            if int_id not in meta_indexed.index:
                raise ValueError(f"Subject '{sid}' (ID={int_id}) not found in metadata.")
            rows.append(meta_indexed.loc[int_id][cols])

        df = pd.DataFrame(rows, index=subject_ids)
        self._label_names = list(df.columns)
        return df

    # ------------------------------------------------------------------
    # Stateful sklearn-style interface
    # ------------------------------------------------------------------

    def fit(self, components: pd.DataFrame) -> None:
        """Fit normalisation statistics and composite operation on training data.

        Must be called with **training-split data only** to prevent leakage.

        Parameters
        ----------
        components : pd.DataFrame
            Shape ``[n_train, K]`` — output of ``get_raw_components`` sliced
            to training indices.

        Notes
        -----
        * If ``cfg.normalize_components`` is ``True``, per-column mean and
          std are computed from ``components`` and stored for later use by
          ``transform``.
        * If the composite operation is stateful (e.g. PCA), its ``fit``
          method is called on the (optionally normalised) training data.
        * For a single-column target this method is effectively a no-op.
        """
        self._label_names = list(components.columns)

        if self.cfg.normalize_components:
            self._component_means = components.mean(axis=0).values.astype(float)
            self._component_stds = components.std(axis=0).values.astype(float)
            self._component_stds = np.where(
                self._component_stds == 0, 1.0, self._component_stds
            )

        if self._composite_op is not None:
            X = components.values.astype(float)
            if self._component_means is not None:
                X = (X - self._component_means) / self._component_stds
            self._composite_op.fit(X)

    def transform(self, components: pd.DataFrame) -> np.ndarray:
        """Apply fitted normalisation and composite construction.

        Parameters
        ----------
        components : pd.DataFrame
            Shape ``[N, K]`` — same columns as the DataFrame passed to ``fit``.

        Returns
        -------
        np.ndarray
            Shape ``[N]``, dtype float — one scalar label per subject.

        Raises
        ------
        RuntimeError
            If ``cfg.normalize_components`` is ``True`` but ``fit`` has not
            been called.
        """
        if self.cfg.normalize_components and self._component_means is None:
            raise RuntimeError("fit() must be called before transform()")

        X = components.values.astype(float)

        if self._component_means is not None:
            X = (X - self._component_means) / self._component_stds

        if self._composite_op is not None:
            return self._composite_op.transform(X)
        else:
            # Single-column target
            return X[:, 0].astype(float)

    def fit_transform(self, components: pd.DataFrame) -> np.ndarray:
        """Convenience: ``fit`` then ``transform`` on the same data.

        Parameters
        ----------
        components : pd.DataFrame
            Shape ``[n_train, K]``.

        Returns
        -------
        np.ndarray
            Shape ``[n_train]``.
        """
        self.fit(components)
        return self.transform(components)

    # ------------------------------------------------------------------
    # State-dict interface (composes into FoldBarrier)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return fitted state as a plain dict for ``torch.save``.

        Returns
        -------
        dict
        """
        return {
            "label_names": list(self._label_names),
            "component_means": (
                self._component_means.tolist()
                if self._component_means is not None
                else None
            ),
            "component_stds": (
                self._component_stds.tolist()
                if self._component_stds is not None
                else None
            ),
            "composite_state": (
                self._composite_op.state_dict()
                if self._composite_op is not None
                and hasattr(self._composite_op, "state_dict")
                else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore fitted state from ``state_dict``.

        Parameters
        ----------
        state : dict
        """
        self._label_names = list(state.get("label_names", []))
        cm = state.get("component_means")
        cs = state.get("component_stds")
        self._component_means = (
            np.asarray(cm, dtype=float) if cm is not None else None
        )
        self._component_stds = (
            np.asarray(cs, dtype=float) if cs is not None else None
        )
        cop_state = state.get("composite_state")
        if (
            cop_state is not None
            and self._composite_op is not None
            and hasattr(self._composite_op, "load_state_dict")
        ):
            self._composite_op.load_state_dict(cop_state)

    # ------------------------------------------------------------------
    # One-shot API (backward-compatible, safe for single-column targets)
    # ------------------------------------------------------------------

    def build(
        self,
        subject_ids: List[str],
        metadata: pd.DataFrame,
    ) -> np.ndarray:
        """Extract labels for the given subjects in a single call.

        For a single ``target`` column this is leakage-free and always safe.
        For composite targets this fits on **all provided subjects** — only
        use this during exploratory work or when ``CrossValidator`` is not
        involved.  For leakage-free composite construction pass
        ``label_builder`` and ``label_components`` to
        :meth:`~src.training.cross_validation.CrossValidator.run`.

        Parameters
        ----------
        subject_ids : List[str]
            Subject identifiers in ``sub-XXX`` format.
        metadata : pd.DataFrame
            Full metadata table with an ID column that maps to subject ids.

        Returns
        -------
        np.ndarray
            Shape ``[len(subject_ids)]``, dtype float.

        Raises
        ------
        ValueError
            If any subject is missing from metadata.
        ValueError
            If neither ``target`` nor ``composite_columns`` is specified.
        """
        components = self.get_raw_components(subject_ids, metadata)
        return self.fit_transform(components)

    def get_label_names(self) -> List[str]:
        """Human-readable names of the label components, for logging.

        Returns
        -------
        List[str]
        """
        return list(self._label_names)
