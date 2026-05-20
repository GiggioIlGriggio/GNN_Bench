"""Abstract base class for brain graph datasets and the ``RawGraphData`` container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch_geometric.data

from src.configs.dataset_config import DatasetConfig
from src.configs.feature_config import FeatureConfig
from src.configs.label_config import LabelConfig


# ---------------------------------------------------------------------------
# Raw per-subject data container (edge-list format, pre-feature-building)
# ---------------------------------------------------------------------------

@dataclass
class RawGraphData:
    """Per-subject raw graph data in sparse edge-list format.

    At least one of ``sc_edge_index`` / ``fc_edge_index`` must be populated
    depending on the requested modality.

    Attributes
    ----------
    subject_id : str
        Subject identifier in ``sub-XXX`` format.
    sc_edge_index : Optional[torch.Tensor]
        Structural connectivity edge list of shape ``[2, E_sc]``, dtype int64.
    sc_edge_attr : Optional[torch.Tensor]
        Structural edge weights/features of shape ``[E_sc, feat_dim]``, dtype float32.
    fc_edge_index : Optional[torch.Tensor]
        Functional connectivity edge list of shape ``[2, E_fc]``, dtype int64.
    fc_edge_attr : Optional[torch.Tensor]
        Functional edge weights/features of shape ``[E_fc, feat_dim]``, dtype float32.
    glm_maps : Optional[Dict[str, np.ndarray]]
        Per-contrast GLM activation maps.  Keys are contrast names
        (e.g. ``"contrast-2back_vs_0back"``), values are 1-D arrays of
        shape ``[num_nodes]`` with one scalar per parcel.
    num_nodes : int
        Number of ROI nodes in the graph.
    metadata : Dict[str, Any]
        Subject-level metadata (age, sex, behavioural scores, etc.).
    """

    subject_id: str
    sc_edge_index: Optional[torch.Tensor] = None
    sc_edge_attr: Optional[torch.Tensor] = None
    fc_edge_index: Optional[torch.Tensor] = None
    fc_edge_attr: Optional[torch.Tensor] = None
    glm_maps: Optional[Dict[str, np.ndarray]] = None
    num_nodes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph contract validator
# ---------------------------------------------------------------------------

def validate_graph_contract(data: torch_geometric.data.Data) -> None:
    """Validate that a PyG ``Data`` object satisfies the compatibility contract.

    Contract (§6):
        - ``data.x``          : ``[num_nodes, node_feat_dim]``  float32
        - ``data.edge_index``  : ``[2, num_edges]``             int64
        - ``data.edge_attr``   : ``[num_edges, edge_feat_dim]`` float32
        - ``data.y``           : ``[1]``                        float32

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph to validate.

    Raises
    ------
    ValueError
        If any attribute is missing, has wrong shape, or wrong dtype.
    """
    if data.x is None:
        raise ValueError("data.x is missing")
    if data.x.dtype != torch.float32:
        raise ValueError(f"data.x must be float32, got {data.x.dtype}")
    if data.x.dim() != 2:
        raise ValueError(f"data.x must be 2D [num_nodes, feat_dim], got shape {data.x.shape}")

    if data.edge_index is None:
        raise ValueError("data.edge_index is missing")
    if data.edge_index.dtype != torch.int64:
        raise ValueError(f"data.edge_index must be int64, got {data.edge_index.dtype}")
    if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(f"data.edge_index must be shape [2, E], got {data.edge_index.shape}")

    if data.edge_attr is None:
        raise ValueError("data.edge_attr is missing")
    if data.edge_attr.dtype != torch.float32:
        raise ValueError(f"data.edge_attr must be float32, got {data.edge_attr.dtype}")

    if data.y is None:
        raise ValueError("data.y is missing")
    if data.y.dtype != torch.float32:
        raise ValueError(f"data.y must be float32, got {data.y.dtype}")
    if data.y.shape != (1,):
        raise ValueError(f"data.y must be shape [1], got {data.y.shape}")


# ---------------------------------------------------------------------------
# Abstract base dataset
# ---------------------------------------------------------------------------

class BrainGraphDataset(ABC):
    """Abstract base for all brain graph datasets.

    Subclasses implement data loading and graph construction; feature and label
    assembly is delegated to :class:`FeatureBuilder` and :class:`LabelBuilder`.

    Parameters
    ----------
    cfg : DatasetConfig
        Dataset configuration (root path, modality, atlas, thresholding, etc.).
    feature_cfg : FeatureConfig
        Node/edge feature toggles and expected dimensions.
    label_cfg : LabelConfig
        Label column selection and composite definition.
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        feature_cfg: FeatureConfig,
        label_cfg: LabelConfig,
    ) -> None:
        self.cfg = cfg
        self.feature_cfg = feature_cfg
        self.label_cfg = label_cfg

        self._subject_ids: List[str] = []
        self._subject_allowlist: Optional[Set[str]] = self._load_subject_allowlist()

        # Lazy-initialised helpers (populated after load_raw)
        self._feature_builder: Optional["FeatureBuilder"] = None  # noqa: F821
        self._label_builder: Optional["LabelBuilder"] = None  # noqa: F821

    def _load_subject_allowlist(self) -> Optional[Set[str]]:
        """Load the subject allowlist from ``cfg.subject_list_file``, if set.

        Returns
        -------
        set[str] | None
            Set of allowed subject IDs, or ``None`` to allow all.

        Raises
        ------
        FileNotFoundError
            If the configured file does not exist.
        ValueError
            If the file is empty or contains duplicate IDs.
        """
        if not self.cfg.subject_list_file:
            return None

        path = Path(self.cfg.subject_list_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Subject list file not found: {path}"
            )

        ids = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        if not ids:
            raise ValueError(f"Subject list file is empty: {path}")

        if len(ids) != len(set(ids)):
            dupes = [x for x in ids if ids.count(x) > 1]
            raise ValueError(
                f"Subject list file contains duplicates: {set(dupes)}"
            )

        return set(ids)

    def _validate_allowlist_coverage(self, loaded_ids: List[str]) -> None:
        """Verify every ID in the allowlist was actually loaded.

        Parameters
        ----------
        loaded_ids : list[str]
            Subject IDs that were successfully loaded.

        Raises
        ------
        ValueError
            If any allowlisted ID was not found on disk.
        """
        if self._subject_allowlist is None:
            return
        loaded_set = set(loaded_ids)
        missing = self._subject_allowlist - loaded_set
        if missing:
            raise ValueError(
                f"{len(missing)} subject(s) from the allowlist were not found "
                f"on disk: {sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_raw(self) -> None:
        """Load raw data from disk into internal state.

        Must populate ``self._subject_ids: List[str]`` after execution.

        Raises
        ------
        FileNotFoundError
            If the dataset root does not exist.
        """

    @abstractmethod
    def build_graph(self, subject_id: str) -> torch_geometric.data.Data:
        """Construct a PyG ``Data`` object for a single subject.

        The resulting ``Data`` must satisfy the compatibility contract (§6).
        The method must call :func:`validate_graph_contract` before returning.

        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g. ``"sub-001"``).

        Returns
        -------
        torch_geometric.data.Data
            Graph satisfying the compatibility contract.  Implementations
            **must** also set ``data.subject_id`` (``str``) so that
            cross-validation can recover per-subject identifiers from plain
            integer indices without a separate lookup table.

        Raises
        ------
        KeyError
            If ``subject_id`` is not found in the loaded data.
        ValueError
            If the resulting graph violates the compatibility contract.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_dataset(self) -> List[torch_geometric.data.Data]:
        """Call :meth:`build_graph` for every subject and return the list.

        Returns
        -------
        List[torch_geometric.data.Data]
        """
        return [self.build_graph(sid) for sid in self._subject_ids]

    def get_labels(self) -> np.ndarray:
        """Return label array of shape ``[N]`` via :class:`LabelBuilder`.

        Uses the one-shot :meth:`~src.datasets.label_builder.LabelBuilder.build`
        path.  For composite targets inside cross-validation prefer
        :meth:`get_label_components` combined with the stateful
        ``fit`` / ``transform`` API to avoid data leakage.

        Returns
        -------
        np.ndarray
            Shape ``[num_subjects]``, dtype float.
        """
        return self._label_builder.build(self._subject_ids, self._metadata)

    def get_label_components(self) -> "pd.DataFrame":  # noqa: F821
        """Return raw label component columns for all subjects.

        Delegates to
        :meth:`~src.datasets.label_builder.LabelBuilder.get_raw_components`.
        Use this when composite labels must be built per fold inside
        :class:`~src.training.cross_validation.CrossValidator` to prevent
        data leakage from normalisation or PCA fitting.

        Returns
        -------
        pd.DataFrame
            Shape ``[num_subjects, K]``, where K is 1 for a single-column
            target or the number of composite columns.
        """
        return self._label_builder.get_raw_components(self._subject_ids, self._metadata)

    def get_subject_ids(self) -> List[str]:
        """Return the list of subject IDs populated by :meth:`load_raw`.

        Returns
        -------
        List[str]
        """
        return list(self._subject_ids)
