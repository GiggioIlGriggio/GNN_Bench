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

    def test_no_glm_features(self) -> None:
        """No GLM features → None."""
        cfg = FeatureConfig(
            node_features=["degree", "strength"],
            edge_features=["weight"],
            node_feat_dim=2,
            edge_feat_dim=1,
        )
        fb = FeatureBuilder(cfg)
        assert fb.get_glm_column_range() is None

    def test_scalar_after_two_features(self) -> None:
        """degree(1) + strength(1) + glm_scalar(1 contrast) → (2, 3)."""
        cfg = FeatureConfig(
            node_features=["degree", "strength", "glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=3,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A"],
        )
        fb = FeatureBuilder(cfg)
        assert fb.get_glm_column_range() == (2, 3)

    def test_scalar_multi_contrast(self) -> None:
        """degree(1) + glm_scalar(2 contrasts) → (1, 3)."""
        cfg = FeatureConfig(
            node_features=["degree", "glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=3,
            edge_feat_dim=1,
            glm_contrasts=["contrast-A", "contrast-B"],
        )
        fb = FeatureBuilder(cfg)
        assert fb.get_glm_column_range() == (1, 3)

    def test_empty_contrasts_returns_none(self) -> None:
        """GLM feature in list but no contrasts → None."""
        cfg = FeatureConfig(
            node_features=["degree", "glm_scalar"],
            edge_features=["weight"],
            node_feat_dim=2,
            edge_feat_dim=1,
            glm_contrasts=[],
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
