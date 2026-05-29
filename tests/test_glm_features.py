"""Tests for GLM map node features and GLM feature normalisation.

All tests use synthetic data — no real dataset paths.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch_geometric.data

from src.configs.feature_config import FeatureConfig
from src.datasets.base_dataset import RawGraphData
from src.datasets.feature_builder import FeatureBuilder
from src.training.glm_normalizer import GLMFeatureNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_graph(
    num_nodes: int = 10,
    num_edges: int = 20,
    glm_maps: Dict[str, np.ndarray] | None = None,
    subject_id: str = "sub-001",
) -> RawGraphData:
    """Create a synthetic RawGraphData with optional GLM maps."""
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int64)
    edge_attr = torch.rand(num_edges, 1, dtype=torch.float32)
    return RawGraphData(
        subject_id=subject_id,
        sc_edge_index=edge_index,
        sc_edge_attr=edge_attr,
        glm_maps=glm_maps,
        num_nodes=num_nodes,
    )


def _make_pyg_graph(num_nodes: int, glm_cols: int, seed: int = 0) -> torch_geometric.data.Data:
    """Create a synthetic PyG Data with GLM columns appended to x."""
    rng = np.random.default_rng(seed)
    # 2 base features (degree, strength) + glm_cols
    x = torch.tensor(rng.standard_normal((num_nodes, 2 + glm_cols)), dtype=torch.float32)
    edge_index = torch.randint(0, num_nodes, (2, 20), dtype=torch.int64)
    edge_attr = torch.rand(20, 1, dtype=torch.float32)
    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([0.0]),
        num_nodes=num_nodes,
    )


# ---------------------------------------------------------------------------
# GLM scalar feature
# ---------------------------------------------------------------------------

class TestGLMScalarFeature:
    """Tests for ``_node_feat_glm_scalar``."""

    def test_single_contrast_shape(self) -> None:
        """Single contrast → [N, 1]."""
        cfg = FeatureConfig(
            node_features=["glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=1,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A"],
        )
        fb = FeatureBuilder(cfg)
        glm_vals = np.random.randn(10).astype(np.float32)
        raw = _make_raw_graph(glm_maps={"contrast-A": glm_vals})
        out = fb.build_node_features(raw)
        assert out.shape == (10, 1)
        assert torch.allclose(out.squeeze(), torch.tensor(glm_vals))

    def test_multi_contrast_shape(self) -> None:
        """Two contrasts → [N, 2]."""
        cfg = FeatureConfig(
            node_features=["glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=2,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A", "contrast-B"],
        )
        fb = FeatureBuilder(cfg)
        glm_a = np.random.randn(10).astype(np.float32)
        glm_b = np.random.randn(10).astype(np.float32)
        raw = _make_raw_graph(glm_maps={"contrast-A": glm_a, "contrast-B": glm_b})
        out = fb.build_node_features(raw)
        assert out.shape == (10, 2)

    def test_missing_glm_maps_raises(self) -> None:
        """Requesting glm_scalar without loading GLM maps should raise."""
        cfg = FeatureConfig(
            node_features=["glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=1,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A"],
        )
        fb = FeatureBuilder(cfg)
        raw = _make_raw_graph(glm_maps=None)
        with pytest.raises(ValueError, match="GLM maps not loaded"):
            fb.build_node_features(raw)

    def test_no_contrasts_configured_raises(self) -> None:
        """glm_scalar with empty glm_contrasts should raise."""
        cfg = FeatureConfig(
            node_features=["glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=1,
            edge_feat_dim=1,
            glm_contrasts=[],
        )
        fb = FeatureBuilder(cfg)
        raw = _make_raw_graph(glm_maps={"contrast-A": np.zeros(10, dtype=np.float32)})
        with pytest.raises(ValueError, match="no contrasts configured"):
            fb.build_node_features(raw)

    def test_combined_with_graph_features(self) -> None:
        """GLM scalar + degree + strength should concatenate correctly."""
        cfg = FeatureConfig(
            node_features=["degree", "strength", "glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=3,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A"],
        )
        fb = FeatureBuilder(cfg)
        glm_vals = np.random.randn(10).astype(np.float32)
        raw = _make_raw_graph(glm_maps={"contrast-A": glm_vals})
        out = fb.build_node_features(raw)
        assert out.shape == (10, 3)
        # GLM values should be in the last column
        assert torch.allclose(out[:, 2], torch.tensor(glm_vals))


# ---------------------------------------------------------------------------
# GLM diagonal feature
# ---------------------------------------------------------------------------

class TestGLMDiagonalFeature:
    """Tests for ``_node_feat_glm_diagonal``."""

    def test_single_contrast_shape(self) -> None:
        """Single contrast → [N, N]."""
        N = 10
        cfg = FeatureConfig(
            node_features=["glm_diagonal"],
            edge_features=["weight"],
            node_feat_dim=N,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A"],
        )
        fb = FeatureBuilder(cfg)
        glm_vals = np.random.randn(N).astype(np.float32)
        raw = _make_raw_graph(num_nodes=N, glm_maps={"contrast-A": glm_vals})
        out = fb.build_node_features(raw)
        assert out.shape == (N, N)

    def test_diagonal_values_correct(self) -> None:
        """Diagonal should hold GLM values, off-diagonal should be zero."""
        N = 5
        cfg = FeatureConfig(
            node_features=["glm_diagonal"],
            edge_features=["weight"],
            node_feat_dim=N,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A"],
        )
        fb = FeatureBuilder(cfg)
        glm_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        raw = _make_raw_graph(num_nodes=N, glm_maps={"contrast-A": glm_vals})
        out = fb.build_node_features(raw)

        # Check diagonal
        for i in range(N):
            assert out[i, i].item() == pytest.approx(glm_vals[i])

        # Check off-diagonal is zero
        mask = ~torch.eye(N, dtype=torch.bool)
        assert (out[mask] == 0).all()

    def test_multi_contrast_shape(self) -> None:
        """Two contrasts → [N, 2*N]."""
        N = 8
        cfg = FeatureConfig(
            node_features=["glm_diagonal"],
            edge_features=["weight"],
            node_feat_dim=2 * N,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A", "contrast-B"],
        )
        fb = FeatureBuilder(cfg)
        raw = _make_raw_graph(
            num_nodes=N,
            glm_maps={
                "contrast-A": np.random.randn(N).astype(np.float32),
                "contrast-B": np.random.randn(N).astype(np.float32),
            },
        )
        out = fb.build_node_features(raw)
        assert out.shape == (N, 2 * N)


# ---------------------------------------------------------------------------
# GLM column range
# ---------------------------------------------------------------------------

class TestGLMColumnRange:
    """Tests for ``FeatureBuilder.get_glm_column_range``."""

    def _raw(self, N: int, contrasts: List[str]):
        from src.datasets.base_dataset import RawGraphData
        edge_index = torch.randint(0, N, (2, 20), dtype=torch.int64)
        glm = {c: np.zeros(N, dtype=np.float32) for c in contrasts}
        return RawGraphData(
            subject_id="sub-001", sc_edge_index=edge_index,
            sc_edge_attr=torch.rand(20, 1), glm_maps=glm or None, num_nodes=N,
        )

    def test_no_glm_features(self) -> None:
        cfg = FeatureConfig(
            node_features=["degree", "strength"], edge_features=["weight"],
            node_feat_dim=2, edge_feat_dim=1,
        )
        fb = FeatureBuilder(cfg)
        fb.build_node_features(self._raw(10, []))
        assert fb.get_glm_column_range() is None

    def test_scalar_after_two_features(self) -> None:
        cfg = FeatureConfig(
            node_features=["degree", "strength", "glm_scalar"],
            edge_features=["weight"], node_feat_dim=3, edge_feat_dim=1,
            glm_contrasts=["contrast-A"],
        )
        fb = FeatureBuilder(cfg)
        fb.build_node_features(self._raw(10, ["contrast-A"]))
        assert fb.get_glm_column_range() == (2, 3)

    def test_scalar_multi_contrast(self) -> None:
        cfg = FeatureConfig(
            node_features=["degree", "glm_scalar"], edge_features=["weight"],
            node_feat_dim=3, edge_feat_dim=1,
            glm_contrasts=["contrast-A", "contrast-B"],
        )
        fb = FeatureBuilder(cfg)
        fb.build_node_features(self._raw(10, ["contrast-A", "contrast-B"]))
        assert fb.get_glm_column_range() == (1, 3)

    def test_empty_contrasts_returns_none(self) -> None:
        # No build needed: empty contrasts short-circuits to None.
        cfg = FeatureConfig(
            node_features=["degree", "glm_scalar"], edge_features=["weight"],
            node_feat_dim=2, edge_feat_dim=1, glm_contrasts=[],
        )
        fb = FeatureBuilder(cfg)
        assert fb.get_glm_column_range() is None


# ---------------------------------------------------------------------------
# GLM feature normalizer
# ---------------------------------------------------------------------------

class TestGLMFeatureNormalizer:
    """Tests for ``GLMFeatureNormalizer``."""

    def test_fit_computes_correct_stats(self) -> None:
        """Mean and std should match manually computed values."""
        N = 5
        # Create 3 graphs with known GLM columns (cols 2:3)
        graphs = []
        glm_values = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            np.array([3.0, 6.0, 9.0, 12.0, 15.0]),
        ]
        for i, vals in enumerate(glm_values):
            x = torch.zeros(N, 3, dtype=torch.float32)
            x[:, 2] = torch.tensor(vals, dtype=torch.float32)
            g = torch_geometric.data.Data(
                x=x,
                edge_index=torch.zeros(2, 0, dtype=torch.int64),
                edge_attr=torch.zeros(0, 1, dtype=torch.float32),
                y=torch.tensor([0.0]),
            )
            graphs.append(g)

        norm = GLMFeatureNormalizer(col_start=2, col_end=3)
        norm.fit(graphs)

        # Expected mean per node: [2.0, 4.0, 6.0, 8.0, 10.0]
        expected_mean = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
        assert torch.allclose(norm.mean_, expected_mean)

    def test_transform_produces_zero_mean(self) -> None:
        """After fit+transform, training set GLM columns should have mean≈0."""
        N = 10
        rng = np.random.default_rng(42)
        graphs = []
        for _ in range(20):
            x = torch.tensor(rng.standard_normal((N, 3)), dtype=torch.float32)
            g = torch_geometric.data.Data(
                x=x.clone(),
                edge_index=torch.zeros(2, 0, dtype=torch.int64),
                edge_attr=torch.zeros(0, 1, dtype=torch.float32),
                y=torch.tensor([0.0]),
            )
            graphs.append(g)

        norm = GLMFeatureNormalizer(col_start=2, col_end=3)
        norm.fit_transform(graphs)

        # Check mean across subjects is ≈ 0 for each node
        stacked = torch.stack([g.x[:, 2:3] for g in graphs])
        mean_per_node = stacked.mean(dim=0)
        assert torch.allclose(mean_per_node, torch.zeros_like(mean_per_node), atol=1e-5)

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform before fit should raise RuntimeError."""
        norm = GLMFeatureNormalizer(col_start=0, col_end=1)
        g = _make_pyg_graph(5, 1)
        with pytest.raises(RuntimeError, match="before fit"):
            norm.transform([g])

    def test_non_glm_columns_unchanged(self) -> None:
        """Columns outside the GLM range should not be modified."""
        N = 5
        graphs = [_make_pyg_graph(N, 1, seed=i) for i in range(10)]
        originals = [g.x[:, :2].clone() for g in graphs]

        norm = GLMFeatureNormalizer(col_start=2, col_end=3)
        norm.fit_transform(graphs)

        for g, orig in zip(graphs, originals):
            assert torch.allclose(g.x[:, :2], orig)

    def test_no_leakage_between_splits(self) -> None:
        """Val/test transforms should use train-only statistics."""
        N = 5
        rng = np.random.default_rng(123)

        # Train graphs with small values
        train_graphs = []
        for _ in range(20):
            x = torch.tensor(rng.standard_normal((N, 3)) * 0.1, dtype=torch.float32)
            g = torch_geometric.data.Data(
                x=x.clone(),
                edge_index=torch.zeros(2, 0, dtype=torch.int64),
                edge_attr=torch.zeros(0, 1, dtype=torch.float32),
                y=torch.tensor([0.0]),
            )
            train_graphs.append(g)

        # Test graphs with large values (different distribution)
        test_graphs = []
        for _ in range(5):
            x = torch.tensor(rng.standard_normal((N, 3)) * 10 + 100, dtype=torch.float32)
            g = torch_geometric.data.Data(
                x=x.clone(),
                edge_index=torch.zeros(2, 0, dtype=torch.int64),
                edge_attr=torch.zeros(0, 1, dtype=torch.float32),
                y=torch.tensor([0.0]),
            )
            test_graphs.append(g)

        norm = GLMFeatureNormalizer(col_start=2, col_end=3)
        norm.fit(train_graphs)
        norm.transform(train_graphs)
        norm.transform(test_graphs)

        # Test values should be far from zero (using train stats on different dist)
        test_means = torch.stack([g.x[:, 2:3] for g in test_graphs]).mean(dim=0)
        assert test_means.abs().mean() > 1.0  # definitely not zero-centered


# ---------------------------------------------------------------------------
# End-to-end: built dataset → feature_builder range → per-fold normalisation
# ---------------------------------------------------------------------------

class TestGLMPerFoldNormalizationIntegration:
    """Verify GLM per-fold normalisation is *active and correctly targeted*.

    This closes the seam that the unit tests leave open: each link
    (``get_glm_column_range``, the ``feature_builder`` property, the
    ``FoldBarrier``) is green in isolation, but nothing asserts that the
    range read from a **built dataset's** ``feature_builder`` actually
    indexes the GLM columns of the produced ``data.x`` and is the range that
    ``CrossValidator`` feeds into the barrier. We reproduce that exact chain
    on synthetic data (no real dataset paths) — the same one
    ``scripts/run_experiment.py`` walks.
    """

    NUM_NODES = 6
    NUM_SUBJECTS = 12
    CONTRASTS = ["contrast-A", "contrast-B"]

    def _make_dataset(self):
        """A minimal concrete ``BrainGraphDataset`` that builds graphs through
        the real ``FeatureBuilder`` path, with degree + strength + 2-contrast
        glm_scalar (GLM columns land at offset (2, 4))."""
        from src.configs.dataset_config import DatasetConfig
        from src.configs.label_config import LabelConfig
        from src.datasets.base_dataset import BrainGraphDataset, RawGraphData

        num_nodes, num_subjects, contrasts = (
            self.NUM_NODES, self.NUM_SUBJECTS, self.CONTRASTS,
        )

        class _SyntheticDataset(BrainGraphDataset):
            def load_raw(self) -> None:
                self._feature_builder = FeatureBuilder(self.feature_cfg)
                self._raw_data = {}
                self._subject_ids = []
                rng = np.random.default_rng(0)
                for i in range(num_subjects):
                    sid = f"sub-{i:03d}"
                    # A connected ring so degree/strength are non-trivial.
                    src = list(range(num_nodes))
                    dst = [(j + 1) % num_nodes for j in range(num_nodes)]
                    edge_index = torch.tensor(
                        [src + dst, dst + src], dtype=torch.int64,
                    )
                    edge_attr = torch.rand(
                        edge_index.shape[1], 1, dtype=torch.float32,
                    )
                    # Distinct GLM maps per subject so normalisation is a real
                    # transform (per-node variance across subjects > 0).
                    glm_maps = {
                        c: (rng.standard_normal(num_nodes) * (k + 1) + 5.0 * i)
                        .astype(np.float32)
                        for k, c in enumerate(contrasts)
                    }
                    self._raw_data[sid] = RawGraphData(
                        subject_id=sid,
                        sc_edge_index=edge_index,
                        sc_edge_attr=edge_attr,
                        glm_maps=glm_maps,
                        num_nodes=num_nodes,
                        metadata={"sdi_AGE": float(8 + i)},
                    )
                    self._subject_ids.append(sid)

            def build_graph(self, subject_id: str):
                # Mirror PncDataset.build_graph exactly: x from the shared
                # FeatureBuilder, sc edges, scalar label from metadata.
                raw = self._raw_data[subject_id]
                x = self._feature_builder.build_node_features(raw)
                edge_attr = self._feature_builder.build_edge_features(raw)
                data = torch_geometric.data.Data(
                    x=x,
                    edge_index=raw.sc_edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(
                        [float(raw.metadata["sdi_AGE"])], dtype=torch.float32,
                    ),
                    num_nodes=raw.num_nodes,
                )
                data.subject_id = subject_id
                return data

        feature_cfg = FeatureConfig(
            node_features=["degree", "strength", "glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=2 + len(contrasts),  # degree + strength + glm
            edge_feat_dim=1,
            glm_contrasts=contrasts,
            glm_normalize=True,
        )
        ds = _SyntheticDataset(
            cfg=DatasetConfig(name="synthetic", root="/tmp/does-not-exist"),
            feature_cfg=feature_cfg,
            label_cfg=LabelConfig(),
        )
        ds.load_raw()
        return ds, feature_cfg

    def test_built_range_indexes_real_glm_columns(self) -> None:
        """The range from the built dataset's ``feature_builder`` must point at
        the actual GLM columns of ``data.x`` — not a stale or off-by-one slice."""
        ds, _ = self._make_dataset()
        graphs = ds.get_dataset()

        glm_col_range = ds.feature_builder.get_glm_column_range()
        # degree(1) + strength(1) → GLM at columns [2, 4).
        assert glm_col_range == (2, 4)

        # The slice the range selects must equal the raw GLM maps, contrast
        # order preserved (col 2 = contrast-A, col 3 = contrast-B).
        start, end = glm_col_range
        for sid, g in zip(ds.get_subject_ids(), graphs):
            raw = ds._raw_data[sid]
            expected = torch.stack(
                [torch.tensor(raw.glm_maps[c]) for c in self.CONTRASTS], dim=1,
            )
            assert torch.allclose(g.x[:, start:end], expected, atol=1e-5)

    def test_per_fold_normalization_fires_on_built_range(self) -> None:
        """Feed the built-dataset range into a ``FoldBarrier`` exactly as
        ``CrossValidator`` does, and confirm the GLM slice is z-scored on the
        train pool while non-GLM columns are untouched."""
        from src.training.fold_barrier import FoldBarrier

        ds, _ = self._make_dataset()
        graphs = ds.get_dataset()
        glm_col_range = ds.feature_builder.get_glm_column_range()
        start, end = glm_col_range

        n = len(graphs)
        train, val, test = graphs[: n - 4], graphs[n - 4 : n - 2], graphs[n - 2 :]
        labels = np.array([float(g.y.item()) for g in graphs])
        train_labels = labels[: n - 4]

        barrier = FoldBarrier(
            label_norm_strategy="standard",
            glm_col_range=glm_col_range,
            glm_normalize=True,
        )
        barrier.fit(train, train_labels)
        train_t = barrier.transform_graphs(train)
        test_t = barrier.transform_graphs(test)

        # Train GLM slice is z-scored: per-(node, col) mean across train ≈ 0.
        train_slice = torch.stack([g.x[:, start:end] for g in train_t])
        assert torch.allclose(
            train_slice.mean(dim=0), torch.zeros(self.NUM_NODES, end - start),
            atol=1e-5,
        )
        # Sanity: it actually moved (raw GLM values were not already centred).
        raw_train_slice = torch.stack([g.x[:, start:end] for g in train])
        assert not torch.allclose(train_slice, raw_train_slice, atol=1e-3)

        # Non-GLM columns (degree, strength) are left exactly as built.
        for built, transformed in zip(train, train_t):
            assert torch.allclose(built.x[:, :start], transformed.x[:, :start])

        # Test split is transformed with TRAIN stats — its later subjects have a
        # large offset (5.0 * i), so it is NOT independently zero-centred.
        test_slice = torch.stack([g.x[:, start:end] for g in test_t])
        assert test_slice.mean().abs() > 0.5

    def test_normalize_flag_gates_the_transform(self) -> None:
        """With ``glm_normalize=False`` the same range must leave GLM columns
        untouched — proving the flag genuinely gates per-fold normalisation."""
        from src.training.fold_barrier import FoldBarrier

        ds, _ = self._make_dataset()
        graphs = ds.get_dataset()
        glm_col_range = ds.feature_builder.get_glm_column_range()

        barrier = FoldBarrier(
            label_norm_strategy="standard",
            glm_col_range=glm_col_range,
            glm_normalize=False,
        )
        barrier.fit(graphs, np.array([float(g.y.item()) for g in graphs]))
        out = barrier.transform_graphs(graphs)
        for built, transformed in zip(graphs, out):
            assert torch.allclose(built.x, transformed.x)
