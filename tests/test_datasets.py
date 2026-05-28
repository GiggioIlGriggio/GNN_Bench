"""Tests for the dataset module.

All tests use synthetic/mock data — no real dataset paths.
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch_geometric.data

from src.configs.dataset_config import DatasetConfig
from src.configs.feature_config import FeatureConfig
from src.configs.label_config import LabelConfig
from src.datasets.base_dataset import RawGraphData, validate_graph_contract
from src.datasets.feature_builder import FeatureBuilder
from src.datasets.label_builder import LabelBuilder
from src.datasets.registry import DATASET_REGISTRY, get_dataset, register_dataset


# ---------------------------------------------------------------------------
# Graph contract validation
# ---------------------------------------------------------------------------

class TestGraphContractValidation:
    """Tests for ``validate_graph_contract``."""

    # def test_valid_graph_passes(self) -> None:
        # """A graph satisfying all contract constraints should not raise."""
        # raise NotImplementedError(
            # "TODO: create a Data with correct shapes/dtypes, "
            # "assert validate_graph_contract does not raise"
        # )

    # def test_missing_x_raises(self) -> None:
        # """A graph without ``data.x`` should raise ValueError."""
        # raise NotImplementedError(
            # "TODO: create Data without x, assert ValueError"
        # )

    # def test_wrong_dtype_raises(self) -> None:
        # """A graph with int64 x should raise ValueError."""
        # raise NotImplementedError(
            # "TODO: create Data with wrong dtype for x, assert ValueError"
        # )

    # def test_wrong_edge_index_shape_raises(self) -> None:
        # """edge_index with shape [E, 2] instead of [2, E] should raise."""
        # raise NotImplementedError(
            # "TODO: create Data with transposed edge_index, assert ValueError"
        # )

    # def test_missing_y_raises(self) -> None:
        # """A graph without ``data.y`` should raise ValueError."""
        # raise NotImplementedError(
            # "TODO: create Data without y, assert ValueError"
        # )


# ---------------------------------------------------------------------------
# FeatureConfig new fields
# ---------------------------------------------------------------------------

class TestFeatureConfigNewFields:
    """Defaults and constraints for the positional/identity feature config."""

    def test_defaults(self) -> None:
        cfg = FeatureConfig(node_features=["degree"], node_feat_dim=1)
        assert cfg.laplacian_pe_dim == 8
        assert cfg.cycle_max_length == 4

    def test_laplacian_pe_dim_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            FeatureConfig(node_features=["degree"], node_feat_dim=1, laplacian_pe_dim=0)

    def test_cycle_max_length_at_least_two(self) -> None:
        with pytest.raises(ValueError):
            FeatureConfig(node_features=["degree"], node_feat_dim=1, cycle_max_length=1)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class TestFeatureBuilder:
    """Tests for ``FeatureBuilder``."""

    # def test_build_node_features_shape(self) -> None:
        # """Output shape should be [num_nodes, node_feat_dim]."""
        # raise NotImplementedError(
            # "TODO: create FeatureBuilder with default config, "
            # "build_node_features with synthetic RawGraphData, "
            # "assert output shape"
        # )

    # def test_build_edge_features_shape(self) -> None:
        # """Output shape should be [num_edges, edge_feat_dim]."""
        # raise NotImplementedError(
            # "TODO: create FeatureBuilder, build_edge_features, assert shape"
        # )

    # def test_unknown_feature_raises(self) -> None:
        # """Requesting a non-existent feature name should raise ValueError."""
        # raise NotImplementedError(
            # "TODO: set node_features=['nonexistent'], assert ValueError"
        # )

    # def test_feature_toggle_independence(self) -> None:
        # """Enabling/disabling features should only change dimensions."""
        # raise NotImplementedError(
            # "TODO: compare output dim with 1 feature vs 2 features"
        # )


# ---------------------------------------------------------------------------
# Label builder
# ---------------------------------------------------------------------------

class TestLabelBuilder:
    """Tests for ``LabelBuilder``."""

    # def test_single_column_extraction(self) -> None:
        # """Single target column should produce [N] array."""
        # raise NotImplementedError(
            # "TODO: create metadata DataFrame, extract single column, assert shape"
        # )

    # def test_weighted_composite(self) -> None:
        # """Weighted composite should produce correct weighted sum."""
        # raise NotImplementedError(
            # "TODO: define composite={col1: 0.5, col2: 0.5}, verify weighted sum"
        # )

    # def test_missing_subject_raises(self) -> None:
        # """Missing subject in metadata should raise ValueError."""
        # raise NotImplementedError(
            # "TODO: pass subject_id not in metadata, assert ValueError"
        # )

    # def test_label_names(self) -> None:
        # """get_label_names should return the correct column names."""
        # raise NotImplementedError("TODO: assert label names match config")


# ---------------------------------------------------------------------------
# Connectivity-profile node features (fc_row / sc_row)
# ---------------------------------------------------------------------------

class TestConnectivityProfileFeatures:
    """Tests for _node_feat_fc_row and _node_feat_sc_row in FeatureBuilder."""

    def _make_cfg(self, node_features, node_feat_dim):
        """Return a FeatureConfig with the given node features and dim."""
        return FeatureConfig(
            node_features=node_features,
            edge_features=["weight"],
            node_feat_dim=node_feat_dim,
            edge_feat_dim=1,
        )

    def _make_fc_graph(self, N: int, E: int):
        """Return a RawGraphData with synthetic FC edges (E edges, N nodes)."""
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        fc_edge_index = torch.stack([src, dst], dim=0)  # [2, E]
        fc_edge_attr = torch.randn(E, 1)                # [E, 1]
        return RawGraphData(
            subject_id="sub-001",
            fc_edge_index=fc_edge_index,
            fc_edge_attr=fc_edge_attr,
            num_nodes=N,
        )

    def _make_sc_graph(self, N: int, E: int):
        """Return a RawGraphData with synthetic SC edges (E edges, N nodes)."""
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        sc_edge_index = torch.stack([src, dst], dim=0)  # [2, E]
        sc_edge_attr = torch.randn(E, 1)                # [E, 1]
        return RawGraphData(
            subject_id="sub-001",
            sc_edge_index=sc_edge_index,
            sc_edge_attr=sc_edge_attr,
            num_nodes=N,
        )

    def test_fc_row_shape(self) -> None:
        """build_node_features with fc_row should produce [N, N] tensor."""
        N, E = 20, 50
        cfg = self._make_cfg(["fc_row"], node_feat_dim=N)
        fb = FeatureBuilder(cfg)
        graph = self._make_fc_graph(N, E)
        out = fb.build_node_features(graph)
        assert out.shape == (N, N), f"Expected ({N}, {N}), got {out.shape}"
        assert out.dtype == torch.float32

    def test_sc_row_shape(self) -> None:
        """build_node_features with sc_row should produce [N, N] tensor."""
        N, E = 20, 50
        cfg = self._make_cfg(["sc_row"], node_feat_dim=N)
        fb = FeatureBuilder(cfg)
        graph = self._make_sc_graph(N, E)
        out = fb.build_node_features(graph)
        assert out.shape == (N, N), f"Expected ({N}, {N}), got {out.shape}"
        assert out.dtype == torch.float32

    def test_fc_row_zeros_when_no_edges(self) -> None:
        """fc_row with fc_edge_index=None should return all-zeros [N, N]."""
        N = 20
        cfg = self._make_cfg(["fc_row"], node_feat_dim=N)
        fb = FeatureBuilder(cfg)
        graph = RawGraphData(subject_id="sub-001", num_nodes=N)
        out = fb.build_node_features(graph)
        assert out.shape == (N, N), f"Expected ({N}, {N}), got {out.shape}"
        assert torch.all(out == 0), "Expected all-zero tensor when no edges"

    def test_node_feat_dim_validation_error(self) -> None:
        """fc_row with node_feat_dim != N should raise ValueError."""
        N, E = 20, 50
        wrong_dim = 10  # != N
        cfg = self._make_cfg(["fc_row"], node_feat_dim=wrong_dim)
        fb = FeatureBuilder(cfg)
        graph = self._make_fc_graph(N, E)
        with pytest.raises(ValueError, match="node_feat_dim"):
            fb.build_node_features(graph)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestDatasetRegistry:
    """Tests for the dataset registry."""

    def test_orbit_is_registered(self) -> None:
        """ORBIT should be auto-registered on import."""
        assert "orbit" in DATASET_REGISTRY

    # def test_get_unknown_raises(self) -> None:
        # """get_dataset with unregistered name should raise KeyError."""
        # raise NotImplementedError(
            # "TODO: call get_dataset('nonexistent', ...), assert KeyError"
        # )

    # def test_duplicate_registration_raises(self) -> None:
        # """Registering the same name twice should raise ValueError."""
        # raise NotImplementedError(
            # "TODO: try to register a class with an existing name, assert ValueError"
        # )
