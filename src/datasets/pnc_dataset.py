"""PNC (Philadelphia Neurodevelopmental Cohort) dataset loader.

Dataset root layout
-------------------
::

    <root>/
    └── T0/
        ├── Structural_maps/           # sub-XXXXXXXXXX_run-1_space-T1w_desc-preproc_dhollanderconnectome.mat
        ├── Functional_Mats/           # sub-XXXXXXXXXX_connectivity_matrix.csv
        ├── Tabular_data/              # PNC_ALL_SCORES.csv
        └── GLM_Maps/

Structural ``.mat`` keys per atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Same format as ORBIT: ``{atlas}_region_ids``, ``{atlas}_region_labels``,
``{atlas}_<sc_type>``.

Functional ``.csv`` files
~~~~~~~~~~~~~~~~~~~~~~~~~
- ``sub-XXXXXXXXXX_connectivity_matrix.csv`` – ``(400, 400)`` correlation
  matrix (Schaefer-400 atlas), stored as a labelled CSV with region names
  as header and index.

Subject-ID mapping
~~~~~~~~~~~~~~~~~~
The metadata CSV ``SUBJID`` column uses 12-digit integer IDs with a
``6000`` prefix (e.g. ``600001103037``).  Filenames use the last 10 digits
with a ``sub-`` prefix (e.g. ``sub-0001103037``).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

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

# Regex to extract the numeric subject ID from PNC filenames.
# Matches e.g. "sub-0001103037" at the start of the filename.
_PNC_SUB_RE = re.compile(r"^(sub-\d+)")


@register_dataset("pnc")
class PNCDataset(BrainGraphDataset):
    """PNC brain-connectivity dataset.

    Parameters
    ----------
    cfg : DatasetConfig
        Must have ``cfg.name == "pnc"`` and ``cfg.root`` pointing to the PNC
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
        self._metadata: Optional["pd.DataFrame"] = None

        # Atlas-level info (shared across subjects, set during load_raw)
        self._region_labels: List[str] = []
        self._region_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def load_raw(self) -> None:  # noqa: C901
        """Load PNC data for all configured timepoints.

        Populates ``self._subject_ids``, ``self._raw_data``,
        ``self._metadata``, and atlas-level attributes.

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
            metadata = pd.read_csv(meta_path, low_memory=False)
            all_metadata_frames.append(metadata)

            # Build lookup: last 10 digits of SUBJID -> row dict
            meta_lookup = self._build_meta_lookup(metadata)

            # Build GLM ID lookup: filename numeric ID -> SUBJID_MATCHED
            # (used when GLM files are named with a different subject ID)
            glm_id_lookup = self._build_glm_id_lookup(metadata)

            if self.cfg.modality in ("sc", "multimodal"):
                sc_dir = tp_dir / "Structural_maps"
                for mat_path in sorted(sc_dir.glob("sub-*_*connectome.mat")):
                    sub_id = self._extract_sub_id(mat_path.name)
                    if sub_id is None:
                        continue

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
                        fc_path = self._find_fc_file(tp_dir, sub_id)
                        if fc_path is not None:
                            try:
                                fc_matrix = self._load_fc_matrix(fc_path)
                                fc_ei, fc_ea = self._dense_to_sparse(fc_matrix)
                            except Exception as e:
                                log.debug("FC load failed for %s: %s", tp_sid, e)

                    # PNC subject ID -> numeric part for CSV lookup
                    numeric_id = sub_id.replace("sub-", "")
                    meta_dict = meta_lookup.get(numeric_id, {})

                    if not meta_dict:
                        skip_counts["metadata_missing"] += 1
                        continue

                    if self._has_nan_label(meta_dict):
                        skip_counts["label_missing"] += 1
                        continue

                    glm_sub_id = glm_id_lookup.get(numeric_id, sub_id)
                    glm_maps = self._load_glm_maps(tp_dir, glm_sub_id)

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
                fc_dir = tp_dir / "Functional_Mats"
                for fc_path in sorted(fc_dir.glob("sub-*_connectivity_matrix.csv")):
                    sub_id = self._extract_sub_id(fc_path.name)
                    if sub_id is None:
                        continue

                    if self._subject_allowlist is not None and sub_id not in self._subject_allowlist:
                        skip_counts["not_in_allowlist"] += 1
                        continue

                    tp_sid = f"{tp}_{sub_id}" if multi_tp else sub_id

                    try:
                        fc_matrix = self._load_fc_matrix(fc_path)
                    except Exception as e:
                        log.debug("Skipping %s (FC load error): %s", tp_sid, e)
                        skip_counts["fc_load_error"] += 1
                        continue

                    fc_ei, fc_ea = self._dense_to_sparse(fc_matrix)
                    num_nodes = fc_matrix.shape[0]

                    numeric_id = sub_id.replace("sub-", "")
                    meta_dict = meta_lookup.get(numeric_id, {})

                    if not meta_dict:
                        skip_counts["metadata_missing"] += 1
                        continue

                    if self._has_nan_label(meta_dict):
                        skip_counts["label_missing"] += 1
                        continue

                    glm_sub_id = glm_id_lookup.get(numeric_id, sub_id)
                    glm_maps = self._load_glm_maps(tp_dir, glm_sub_id)

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

    def get_label_components(self) -> "pd.DataFrame":
        """Return raw label component columns for all loaded subjects.

        Overrides the base-class implementation to sidestep the integer-ID
        conversion in :class:`~src.datasets.label_builder.LabelBuilder`.
        The PNC ``SUBJID`` column uses a 12-digit format (``6000XXXXXXXXXX``)
        that cannot be recovered from the ``sub-XXXXXXXXXX`` filename ID by
        simple integer casting.  Instead we extract column values directly
        from the per-subject metadata dicts, which already hold the full CSV
        row.

        Returns
        -------
        pd.DataFrame
            Shape ``[num_subjects, K]``, indexed by subject ID.
        """
        cols = (
            list(self.label_cfg.composite_columns)
            if self.label_cfg.is_composite
            else [self.label_cfg.target]
        )
        rows = []
        for sid in self._subject_ids:
            meta = self._raw_data[sid].metadata
            rows.append({c: meta.get(c) for c in cols})
        df = pd.DataFrame(rows, index=self._subject_ids, columns=cols)
        # Keep LabelBuilder's internal state in sync
        self._label_builder._label_names = list(df.columns)
        return df

    def get_label_column(self, column: str) -> np.ndarray:
        """Per-subject vector for an arbitrary metadata column (PNC override).

        Reads the column directly from the per-subject metadata dicts (like
        :meth:`get_label_components`) to sidestep
        :class:`~src.datasets.label_builder.LabelBuilder`'s integer-ID cast,
        which cannot recover PNC's 12-digit ``SUBJID`` (``6000XXXXXXXXXX``)
        from the ``sub-XXXXXXXXXX`` filename ID.  Returns raw float values in
        ``self._subject_ids`` order — used to stratify a source run on a column
        other than its regression target (e.g. age-source runs stratified on
        VWM so their folds match the VWM runs).

        Uses ``metadata[column]`` (not ``.get(column)``) so a missing column
        raises a clear ``KeyError`` rather than silently producing ``NaN``;
        stratification on a typo'd column should fail loudly.

        Parameters
        ----------
        column : str
            Name of the metadata column to extract.

        Returns
        -------
        np.ndarray
            Shape ``[num_subjects]``, dtype float, in subject-ID order.
        """
        return np.asarray(
            [float(self._raw_data[sid].metadata[column]) for sid in self._subject_ids],
            dtype=float,
        )

    def get_labels(self) -> np.ndarray:
        """Return label array of shape ``[N]`` for all loaded subjects.

        Overrides the base-class to use :meth:`get_label_components` (which
        reads from per-subject metadata dicts) and then delegates to the
        :class:`~src.datasets.label_builder.LabelBuilder` fit/transform
        pipeline so that normalisation and composite construction work
        correctly.

        Returns
        -------
        np.ndarray
            Shape ``[num_subjects]``, dtype float.
        """
        components = self.get_label_components()
        return self._label_builder.fit_transform(components)

    def build_graph(self, subject_id: str) -> torch_geometric.data.Data:
        """Construct a PyG Data object for a single PNC subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g. ``"sub-0001103037"`` or
            ``"T0_sub-0001103037"`` for longitudinal).

        Returns
        -------
        torch_geometric.data.Data

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

        # Label
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
        data.subject_id = subject_id

        validate_graph_contract(data)
        return data

    # ------------------------------------------------------------------
    # PNC-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sub_id(filename: str) -> str | None:
        """Extract ``sub-XXXXXXXXXX`` from a PNC filename.

        Parameters
        ----------
        filename : str
            E.g. ``"sub-0001103037_run-1_space-T1w_desc-preproc_dhollanderconnectome.mat"``
            or ``"sub-0001103037_connectivity_matrix.csv"``.

        Returns
        -------
        str | None
            ``"sub-0001103037"`` or ``None`` if parsing fails.
        """
        m = _PNC_SUB_RE.match(filename)
        return m.group(1) if m else None

    def _build_meta_lookup(self, metadata: pd.DataFrame) -> dict[str, dict]:
        """Build a lookup from filename numeric ID to metadata row dict.

        The PNC metadata CSV uses 12-digit ``SUBJID`` values with a ``6000``
        prefix.  Filenames use the last 10 digits.  This method creates a
        dict keyed by those 10-digit strings for O(1) lookup.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata table with an ``id_column`` (typically ``SUBJID``).

        Returns
        -------
        dict[str, dict]
            Mapping ``"0001103037"`` → row dict.
        """
        lookup: dict[str, dict] = {}
        id_col = self.label_cfg.id_column
        for _, row in metadata.iterrows():
            raw_id = str(int(row[id_col]))
            # Take last 10 digits (strips the 6000 prefix)
            file_id = raw_id[-10:].zfill(10)
            lookup[file_id] = row.to_dict()
        return lookup

    def _build_glm_id_lookup(self, metadata: pd.DataFrame) -> dict[str, str]:
        """Build a lookup from filename numeric ID to GLM subject ID.

        When ``label_cfg.glm_id_column`` is set, the GLM ``.npy`` files are
        named using a different subject ID than the one derived from the
        connectivity filenames.  This method maps the standard 10-digit
        filename ID to the GLM filename ID (e.g. ``SUBJID_MATCHED``).

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata table containing both ``id_column`` and
            ``glm_id_column``.

        Returns
        -------
        dict[str, str]
            Mapping ``"0001103037"`` → ``"sub-2718945982"`` (the GLM file
            subject ID).  Empty dict if ``glm_id_column`` is not configured.
        """
        glm_col = self.label_cfg.glm_id_column
        if not glm_col:
            return {}
        lookup: dict[str, str] = {}
        id_col = self.label_cfg.id_column
        for _, row in metadata.iterrows():
            raw_id = str(int(row[id_col]))
            file_id = raw_id[-10:].zfill(10)
            glm_val = row.get(glm_col)
            if pd.notna(glm_val):
                lookup[file_id] = str(glm_val)
        return lookup

    def _load_sc_matrix(self, mat_path: Path) -> np.ndarray:
        """Load a structural connectivity matrix from a ``.mat`` file.

        Parameters
        ----------
        mat_path : Path
            Path to ``sub-XXX_run-1_..._connectome.mat``.

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

        n_rois = int(re.search(r"\d+", key).group())
        return mat[key].astype(np.float32)[:n_rois, :n_rois]

    def _load_fc_matrix(self, fc_path: Path) -> np.ndarray:
        """Load a functional connectivity matrix from a CSV file.

        PNC stores FC matrices as labelled CSVs (region names as both
        header and index), unlike ORBIT which uses ``.npy`` files.

        Parameters
        ----------
        fc_path : Path
            Path to ``sub-XXX_connectivity_matrix.csv``.

        Returns
        -------
        np.ndarray
            Dense correlation matrix of shape ``[N, N]``.
        """
        if not fc_path.exists():
            raise FileNotFoundError(f"FC matrix not found: {fc_path}")
        df = pd.read_csv(fc_path, header=0, index_col=0)
        return df.values.astype(np.float32)

    def _find_fc_file(self, tp_dir: Path, sub_id: str) -> Path | None:
        """Locate the FC connectivity CSV for a subject.

        Parameters
        ----------
        tp_dir : Path
            Timepoint directory (e.g. ``<root>/T0``).
        sub_id : str
            Subject identifier (e.g. ``"sub-0001103037"``).

        Returns
        -------
        Path | None
            Path to the CSV, or ``None`` if not found.
        """
        fc_path = tp_dir / "Functional_Mats" / f"{sub_id}_connectivity_matrix.csv"
        return fc_path if fc_path.exists() else None

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
            return False

        return any(pd.isna(meta_dict.get(col)) for col in cols)

    def _has_missing_glm(self, glm_maps: "dict | None") -> bool:
        """Return True if GLM contrasts are configured but any are missing.

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
            return False
        if glm_maps is None:
            return True
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
            Subject identifier (e.g. ``"sub-0001103037"``).

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
