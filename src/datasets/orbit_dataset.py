"""ORBIT dataset loader.

Dataset root layout
-------------------
::

    <root>/
    ├── T0/
    │   ├── Structural_Mats/          # sub-XXX.mat  (scipy .mat)
    │   ├── Functional_Mats/          # sub-XXX/<task>/fc_matrix.npy
    │   ├── Tabular_Data/             # ALL tabular data.csv
    │   └── GLM_Maps/
    └── T1/
        └── (same structure)

Structural ``.mat`` keys per atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``{atlas}_region_ids``       – ``(1, N)``
- ``{atlas}_region_labels``    – ``(N,)``
- ``{atlas}_<sc_type>``        – ``(N, N)``  dense connectivity matrix

Functional ``.npy`` files
~~~~~~~~~~~~~~~~~~~~~~~~~
- ``fc_matrix.npy`` – ``(400, 400)`` correlation matrix (Schaefer-400 atlas).

Subject-ID mapping
~~~~~~~~~~~~~~~~~~
Tabular CSV uses integer ``ID`` (e.g. ``1``); filenames use zero-padded
``sub-001`` format.  Conversion is controlled by
``DatasetConfig.subject_id_pad_width``.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional
import re

import numpy as np
import pandas as pd
import torch
import torch_geometric.data

from src.configs.dataset_config import DatasetConfig
from src.configs.feature_config import FeatureConfig
from src.configs.label_config import LabelConfig
from src.datasets.base_dataset import (
    BrainGraphDataset,
    RawGraphData,
    validate_graph_contract,
)
from src.datasets.feature_builder import FeatureBuilder
from src.datasets.label_builder import LabelBuilder
from src.datasets.registry import register_dataset

log = logging.getLogger(__name__)


@register_dataset("orbit")
class ORBITDataset(BrainGraphDataset):
    """ORBIT brain-connectivity dataset.

    Parameters
    ----------
    cfg : DatasetConfig
        Must have ``cfg.name == "orbit"`` and ``cfg.root`` pointing to the ORBIT
        data directory.
    feature_cfg : FeatureConfig
        Node/edge feature toggles.
    label_cfg : LabelConfig
        Label selection & composite definition.
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        feature_cfg: FeatureConfig,
        label_cfg: LabelConfig,
    ) -> None:
        super().__init__(cfg=cfg, feature_cfg=feature_cfg, label_cfg=label_cfg)

        self._root = Path(cfg.root)
        self._feature_builder = FeatureBuilder(feature_cfg)
        self._label_builder = LabelBuilder(label_cfg)

        # Populated by load_raw
        self._raw_data: dict[str, RawGraphData] = {}
        self._metadata: Optional["pd.DataFrame"] = None  # noqa: F821

        # Atlas-level info (shared across subjects, set during load_raw)
        self._region_labels: List[str] = []
        self._region_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def load_raw(self) -> None:
        """Load ORBIT data for all configured timepoints.

        Populates ``self._subject_ids``, ``self._raw_data``,
        ``self._metadata``, and atlas-level attributes.

        iterates over configured timepoints, loads the metadata CSV, then for each subject: 
        discovers SC .mat files and/or FC .npy files, converts dense matrices to sparse edge lists, 
        and stores RawGraphData entries. Also extracts atlas-level region labels/IDs from the first
         subject's mat file

        Raises
        ------
        FileNotFoundError
            If the dataset root or any expected sub-directory is missing.
        """
        if not self._root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self._root}")

        all_metadata_frames: list[pd.DataFrame] = []
        subject_ids: list[str] = []
        multi_tp = len(self.cfg.timepoints) > 1
        skip_counts: Counter = Counter()

        for tp in self.cfg.timepoints:
            tp_dir = self._root / tp
            if not tp_dir.exists():
                raise FileNotFoundError(f"Timepoint directory not found: {tp_dir}")

            meta_path = tp_dir / self.label_cfg.metadata_file
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
            metadata = pd.read_csv(meta_path)
            all_metadata_frames.append(metadata)

            if self.cfg.modality in ("sc", "multimodal"):
                sc_dir = tp_dir / "Structural_Mats"
                for mat_path in sorted(sc_dir.glob("sub-*.mat")):
                    sub_id = mat_path.stem  # e.g. "sub-001"

                    if self._subject_allowlist is not None and sub_id not in self._subject_allowlist:
                        skip_counts["not_in_allowlist"] += 1
                        continue

                    tp_sid = f"{tp}_{sub_id}" if multi_tp else sub_id

                    try:
                        sc_matrix = self._load_sc_matrix(mat_path)
                    except Exception as e:
                        log.debug("Skipping %s (SC load error): %s", tp_sid, e)
                        skip_counts["sc_load_error"] += 1
                        continue

                    sc_ei, sc_ea = self._dense_to_sparse(sc_matrix)
                    num_nodes = sc_matrix.shape[0]

                    # Extract atlas-level info once
                    if not self._region_labels:
                        import scipy.io
                        mat = scipy.io.loadmat(str(mat_path))
                        rl_key = f"{self.cfg.atlas}_region_labels"
                        ri_key = f"{self.cfg.atlas}_region_ids"
                        if rl_key in mat:
                            raw_rl = mat[rl_key].flatten()
                            self._region_labels = [
                                str(v[0]) if hasattr(v, "__len__") else str(v)
                                for v in raw_rl
                            ]
                        if ri_key in mat:
                            self._region_ids = mat[ri_key].flatten()

                    fc_ei, fc_ea = None, None
                    if self.cfg.modality == "multimodal":
                        fc_task_dir = tp_dir / "Functional_Mats" / sub_id / self.cfg.fc_task
                        if fc_task_dir.exists():
                            try:
                                fc_matrix = self._load_fc_matrix(fc_task_dir)
                                fc_ei, fc_ea = self._dense_to_sparse(fc_matrix)
                            except Exception as e:
                                log.debug("FC load failed for %s: %s", tp_sid, e)

                    int_id = int(sub_id.split("-")[1])
                    meta_row = metadata[metadata[self.label_cfg.id_column] == int_id]
                    meta_dict = meta_row.iloc[0].to_dict() if len(meta_row) > 0 else {}

                    if self._has_nan_label(meta_dict):
                        skip_counts["label_missing"] += 1
                        continue

                    glm_maps = self._load_glm_maps(tp_dir, sub_id)

                    if self._has_missing_glm(glm_maps):
                        skip_counts["glm_missing"] += 1
                        continue

                    self._raw_data[tp_sid] = RawGraphData(
                        subject_id=tp_sid,
                        sc_edge_index=sc_ei,
                        sc_edge_attr=sc_ea,
                        fc_edge_index=fc_ei,
                        fc_edge_attr=fc_ea,
                        glm_maps=glm_maps,
                        num_nodes=num_nodes,
                        metadata=meta_dict,
                    )
                    subject_ids.append(tp_sid)

            elif self.cfg.modality == "fc":
                fc_parent = tp_dir / "Functional_Mats"
                for sub_dir in sorted(fc_parent.iterdir()):
                    if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                        continue
                    sub_id = sub_dir.name

                    if self._subject_allowlist is not None and sub_id not in self._subject_allowlist:
                        skip_counts["not_in_allowlist"] += 1
                        continue

                    fc_task_dir = sub_dir / self.cfg.fc_task
                    if not fc_task_dir.exists():
                        continue

                    try:
                        fc_matrix = self._load_fc_matrix(fc_task_dir)
                    except Exception as e:
                        log.debug("Skipping %s (FC load error): %s", sub_id, e)
                        skip_counts["fc_load_error"] += 1
                        continue

                    fc_ei, fc_ea = self._dense_to_sparse(fc_matrix)
                    num_nodes = fc_matrix.shape[0]
                    tp_sid = f"{tp}_{sub_id}" if multi_tp else sub_id

                    int_id = int(sub_id.split("-")[1])
                    meta_row = metadata[metadata[self.label_cfg.id_column] == int_id]
                    meta_dict = meta_row.iloc[0].to_dict() if len(meta_row) > 0 else {}

                    if self._has_nan_label(meta_dict):
                        skip_counts["label_missing"] += 1
                        continue

                    glm_maps = self._load_glm_maps(tp_dir, sub_id)

                    if self._has_missing_glm(glm_maps):
                        skip_counts["glm_missing"] += 1
                        continue

                    self._raw_data[tp_sid] = RawGraphData(
                        subject_id=tp_sid,
                        sc_edge_index=None,
                        sc_edge_attr=None,
                        fc_edge_index=fc_ei,
                        fc_edge_attr=fc_ea,
                        glm_maps=glm_maps,
                        num_nodes=num_nodes,
                        metadata=meta_dict,
                    )
                    subject_ids.append(tp_sid)

        self._subject_ids = subject_ids
        if all_metadata_frames:
            self._metadata = (
                pd.concat(all_metadata_frames, ignore_index=True)
                .drop_duplicates(subset=[self.label_cfg.id_column])
            )

        self._validate_allowlist_coverage(subject_ids)

        n_skipped = sum(skip_counts.values())
        skip_summary = ", ".join(f"{r}={n}" for r, n in sorted(skip_counts.items()))
        log.info(
            "load_raw complete: %d loaded, %d skipped%s (timepoints=%s, modality=%s)",
            len(self._subject_ids),
            n_skipped,
            f" ({skip_summary})" if n_skipped else "",
            self.cfg.timepoints,
            self.cfg.modality,
        )

    def build_graph(self, subject_id: str) -> torch_geometric.data.Data:
        """Construct a PyG Data object for a single ORBIT subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g. ``"T0_sub-001"`` for longitudinal).

        Returns
        -------
        torch_geometric.data.Data
            Graph satisfying the compatibility contract (§6).

        Raises
        ------
        KeyError
            If ``subject_id`` not found in ``self._raw_data``.
        ValueError
            If the resulting graph violates the compatibility contract.
        """
        if subject_id not in self._raw_data:
            raise KeyError(f"Subject '{subject_id}' not found in raw data.")

        raw = self._raw_data[subject_id]

        # Node features
        x = self._feature_builder.build_node_features(raw)

        # Pick edge_index / edge_attr by modality
        if self.cfg.modality == "fc":
            edge_index = raw.fc_edge_index
        else:  # sc or multimodal (SC primary)
            edge_index = raw.sc_edge_index

        edge_attr = self._feature_builder.build_edge_features(raw)

        # Label — for composite targets the real per-fold label is set by
        # CrossValidator; here we store a placeholder so the graph still
        # passes the compatibility contract.
        if self.label_cfg.is_composite:
            y = torch.tensor([0.0], dtype=torch.float32)
        else:
            if not self.label_cfg.target:
                raise ValueError(
                    "No target label configured "
                    "(label_cfg.target is empty and composite is not set)."
                )
            if self.label_cfg.target not in raw.metadata:
                raise KeyError(
                    f"Target label '{self.label_cfg.target}' missing "
                    f"for subject {subject_id}."
                )
            label_val = float(raw.metadata[self.label_cfg.target])
            y = torch.tensor([label_val], dtype=torch.float32)

        data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=raw.num_nodes,
        )

        # Store subject_id for downstream use (e.g. logging fold splits)
        data.subject_id = subject_id

        validate_graph_contract(data)
        return data

    # ------------------------------------------------------------------
    # ORBIT-specific helpers
    # ------------------------------------------------------------------

    def _load_sc_matrix(self, mat_path: Path) -> np.ndarray:
        """Load a structural connectivity matrix from a ``.mat`` file.

        Parameters
        ----------
        mat_path : Path
            Path to ``sub-XXX.mat``.

        Returns
        -------
        np.ndarray
            Dense connectivity matrix of shape ``[N, N]``.
        """
        import scipy.io

        mat = scipy.io.loadmat(str(mat_path))

        key = f"{self.cfg.atlas}_{self.cfg.sc_type}"

        if key not in mat:
            available = [k for k in mat.keys() if not k.startswith("_")]
            raise KeyError(
                f"Key '{key}' not found in {mat_path.name}. Available: {available}"
            )
        
        n_rois = int(re.search(r'\d+', key).group()) # e.g. scheeflere400 has in reeal 452 rois, but we only want 400
        return mat[key].astype(np.float32)[:n_rois, :n_rois]

    def _load_fc_matrix(self, fc_dir: Path) -> np.ndarray:
        """Load a functional connectivity matrix from a ``.npy`` file.

        Parameters
        ----------
        fc_dir : Path
            Path to ``sub-XXX/<fc_task>/fc_matrix.npy``.

        Returns
        -------
        np.ndarray
            Dense correlation matrix of shape ``[N, N]``.
        """
        fc_file = fc_dir / "fc_matrix.npy"
        if not fc_file.exists():
            raise FileNotFoundError(f"FC matrix not found: {fc_file}")
        return np.load(str(fc_file)).astype(np.float32)

    def _dense_to_sparse(self, matrix: np.ndarray) -> tuple:
        """Convert a dense connectivity matrix to sparse edge list.

        Applies thresholding according to ``self.cfg.edge_threshold_mode``
        and ``self.cfg.edge_threshold_value``.

        Parameters
        ----------
        matrix : np.ndarray
            Dense ``[N, N]`` connectivity matrix.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            ``(edge_index [2, E], edge_attr [E, 1])``
        """
        import torch

        mode = self.cfg.edge_threshold_mode
        val = self.cfg.edge_threshold_value

        if mode == "absolute":
            mask = matrix > val
        elif mode == "topk_percent":
            nonzero_vals = matrix[matrix > 0]
            if len(nonzero_vals) == 0:
                mask = np.zeros_like(matrix, dtype=bool)
            else:
                threshold = np.percentile(nonzero_vals, 100.0 - val)
                mask = matrix >= threshold
            # Ensure the thresholded graph is connected by augmenting with
            # maximum-spanning-tree edges when disconnected components exist.
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

            n_comp, _ = connected_components(
                csr_matrix(mask.astype(np.float32)), directed=False
            )
            if n_comp > 1:
                log.warning(
                    "topk_percent thresholding produced %d disconnected components; "
                    "augmenting with maximum-spanning-tree edges to restore connectivity.",
                    n_comp,
                )
                mst = minimum_spanning_tree(csr_matrix(-matrix))
                mst_sym = (mst + mst.T).astype(bool).toarray()
                mask = mask | mst_sym
        else:  # "none"
            mask = matrix != 0

        rows, cols = np.where(mask)
        edge_index = torch.tensor(
            np.stack([rows, cols], axis=0), dtype=torch.int64
        )
        edge_attr = torch.tensor(
            matrix[rows, cols], dtype=torch.float32
        ).unsqueeze(-1)
        return edge_index, edge_attr

    def _has_nan_label(self, meta_dict: dict) -> bool:
        """Return True if any configured label column has a NaN or missing value.

        Checks ``label_cfg.target`` for a single-column target, or every column
        in ``label_cfg.composite_columns`` for a composite target.  A missing
        key in ``meta_dict`` is treated the same as NaN.

        Parameters
        ----------
        meta_dict : dict
            Metadata dictionary for a single subject.

        Returns
        -------
        bool
            ``True`` if the subject should be skipped.
        """
        if self.label_cfg.is_composite:
            cols = list(self.label_cfg.composite_columns)
        elif self.label_cfg.target:
            cols = [self.label_cfg.target]
        else:
            return False  # no label configured; don't skip
        
        return any(pd.isna(meta_dict.get(col)) for col in cols)

    def _has_missing_glm(self, glm_maps: "dict | None") -> bool:
        """Return True if GLM contrasts are configured but any are missing.

        A subject should be skipped when GLM features are requested
        (``glm_contrasts`` is non-empty) but ``_load_glm_maps`` could not
        load all contrasts for that subject.

        Parameters
        ----------
        glm_maps : dict | None
            Return value of :meth:`_load_glm_maps`.

        Returns
        -------
        bool
            ``True`` if the subject should be skipped.
        """
        required = self.feature_cfg.glm_contrasts
        if not required:
            return False  # no GLM features requested; nothing to check
        if glm_maps is None:
            return True
        # Skip if any required contrast is absent
        return any(c not in glm_maps for c in required)

    def _load_glm_maps(
        self, tp_dir: Path, sub_id: str,
    ) -> dict[str, np.ndarray] | None:
        """Load GLM activation maps for a single subject.

        Reads one ``.npy`` file per configured contrast from::

            <tp_dir>/GLM_Maps/<atlas>_glm/<aggregation>/<map_type>/<contrast>/<sub_id>.npy

        Parameters
        ----------
        tp_dir : Path
            Timepoint directory (e.g. ``<root>/T0``).
        sub_id : str
            Subject identifier (e.g. ``"sub-001"``).

        Returns
        -------
        dict[str, np.ndarray] | None
            Mapping from contrast name to 1-D array of shape ``[num_nodes]``.
            Returns ``None`` when no GLM contrasts are configured.
        """
        contrasts = self.feature_cfg.glm_contrasts
        if not contrasts:
            return None

        glm_base = (
            tp_dir
            / "GLM_Maps"
            / f"{self.cfg.atlas}_glm"
            / self.feature_cfg.glm_aggregation
            / self.feature_cfg.glm_map_type
        )

        glm_maps: dict[str, np.ndarray] = {}
        for contrast in contrasts:
            npy_path = glm_base / contrast / f"{sub_id}.npy"
            if not npy_path.exists():
                log.debug(
                    "GLM map missing for %s / %s: %s", sub_id, contrast, npy_path
                )
                continue
            glm_maps[contrast] = np.load(str(npy_path)).astype(np.float32)

        return glm_maps if glm_maps else None

    @staticmethod
    def _int_id_to_sub_id(int_id: int, pad_width: int = 3) -> str:
        """Convert integer subject ID to ``sub-XXX`` format.

        Parameters
        ----------
        int_id : int
            Integer ID from the tabular CSV.
        pad_width : int
            Zero-padding width.

        Returns
        -------
        str
            E.g. ``"sub-001"`` for ``int_id=1, pad_width=3``.
        """
        return f"sub-{int_id:0{pad_width}d}"
