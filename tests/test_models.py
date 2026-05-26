"""Tests for the model module.

All tests use synthetic data — no real datasets.
"""

from __future__ import annotations

import pytest
import torch
import torch_geometric.data

from src.configs.model_config import ModelConfig
from src.models.base_model import BrainGNN
from src.models.backbones.base_backbone import GNNBackbone
from src.models.backbones.gcn import GCNBackbone
from src.models.backbones.gat import GATBackbone
from src.models.backbones.gin import GINBackbone
from src.models.backbones.transformer import GraphTransformerBackbone
from src.models.fusion.base_fusion import ModalityFusion
from src.models.fusion.concat_fusion import ConcatFusion
from src.models.fusion.attention_fusion import CrossAttentionFusion
from src.models.fusion.gated_fusion import GatedFusion
from src.models.heads.regression_head import RegressionHead
from src.models.registry import MODEL_REGISTRY, get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    num_nodes: int = 50,
    node_feat_dim: int = 2,
    num_edges: int = 200,
    edge_feat_dim: int = 1,
) -> torch_geometric.data.Data:
    """Create a synthetic PyG Data object for testing."""
    return torch_geometric.data.Data(
        x=torch.randn(num_nodes, node_feat_dim),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, edge_feat_dim),
        y=torch.tensor([3.14], dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Backbone tests
# ---------------------------------------------------------------------------

class TestGCNBackbone:
    """Tests for GCN backbone."""

    def test_forward_output_shape(self) -> None:
        """GCN forward should produce [num_nodes, hidden_dim]."""
        raise NotImplementedError(
            "TODO: instantiate GCNBackbone, forward synthetic data, "
            "assert output shape == [num_nodes, hidden_dim]"
        )

    def test_get_output_dim(self) -> None:
        """get_output_dim should match hidden_dim."""
        raise NotImplementedError(
            "TODO: assert backbone.get_output_dim() == cfg.hidden_dim"
        )


class TestGATBackbone:
    """Tests for GAT backbone."""

    def test_forward_output_shape(self) -> None:
        """GAT forward should produce [num_nodes, hidden_dim]."""
        raise NotImplementedError("TODO: same as GCN test but with GAT")

    def test_heads_divisibility(self) -> None:
        """hidden_dim must be divisible by heads."""
        raise NotImplementedError(
            "TODO: set hidden_dim=63, heads=4, assert error or assertion"
        )


class TestGINBackbone:
    """Tests for GIN backbone."""

    def test_forward_output_shape(self) -> None:
        """GIN forward should produce [num_nodes, hidden_dim]."""
        raise NotImplementedError("TODO: same structure as GCN test")


class TestGraphTransformerBackbone:
    """Tests for Graph Transformer backbone."""

    def test_forward_output_shape(self) -> None:
        """Transformer forward should produce [num_nodes, hidden_dim]."""
        raise NotImplementedError("TODO: same structure as GAT test")


# ---------------------------------------------------------------------------
# Fusion tests
# ---------------------------------------------------------------------------

class TestConcatFusion:
    """Tests for ConcatFusion."""

    def test_output_shape(self) -> None:
        """Output dim should be d_sc + d_fc."""
        raise NotImplementedError(
            "TODO: instantiate ConcatFusion(32, 32), forward two [B, 32] tensors, "
            "assert output shape [B, 64]"
        )

    def test_get_output_dim(self) -> None:
        """get_output_dim should return d_sc + d_fc."""
        raise NotImplementedError("TODO: assert get_output_dim() == 64")


class TestCrossAttentionFusion:
    """Tests for CrossAttentionFusion."""

    def test_output_shape(self) -> None:
        """Output should match d_out."""
        raise NotImplementedError("TODO: forward, assert shape [B, d_out]")


class TestGatedFusion:
    """Tests for GatedFusion."""

    def test_output_shape(self) -> None:
        """Output should match d_out."""
        raise NotImplementedError("TODO: forward, assert shape [B, d_out]")


# ---------------------------------------------------------------------------
# Prediction head tests
# ---------------------------------------------------------------------------

class TestRegressionHead:
    """Tests for RegressionHead."""

    def test_output_shape(self) -> None:
        """Output should be [B, 1]."""
        raise NotImplementedError(
            "TODO: instantiate RegressionHead, forward [B, embedding_dim], "
            "assert output [B, 1]"
        )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Tests for model registry."""

    def test_get_unknown_raises(self) -> None:
        """get_model with unregistered name should raise KeyError."""
        raise NotImplementedError(
            "TODO: call get_model('nonexistent', ...), assert KeyError"
        )

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice should raise ValueError."""
        raise NotImplementedError(
            "TODO: try to register a model with an existing name, assert ValueError"
        )


# ---------------------------------------------------------------------------
# BrainGNNModel with ROIAwareConv tests
# ---------------------------------------------------------------------------

class TestBrainGNNWithROIAwareConv:
    """Integration tests for BrainGNNModel using ROIAwareConv."""

    def _make_batch(
        self,
        num_graphs: int = 2,
        num_nodes: int = 10,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 1,
        num_edges_per_graph: int = 20,
    ) -> torch_geometric.data.Data:
        """Build a batched synthetic PyG Data object."""
        all_x, all_ei, all_ea, all_batch = [], [], [], []
        offset = 0
        for g in range(num_graphs):
            x = torch.randn(num_nodes, node_feat_dim)
            ei = torch.randint(0, num_nodes, (2, num_edges_per_graph)) + offset
            ea = torch.randn(num_edges_per_graph, edge_feat_dim)
            batch = torch.full((num_nodes,), g, dtype=torch.long)
            all_x.append(x)
            all_ei.append(ei)
            all_ea.append(ea)
            all_batch.append(batch)
            offset += num_nodes

        return torch_geometric.data.Data(
            x=torch.cat(all_x, dim=0),
            edge_index=torch.cat(all_ei, dim=1),
            edge_attr=torch.cat(all_ea, dim=0),
            batch=torch.cat(all_batch, dim=0),
            y=torch.randn(num_graphs, 1),
        )

    def test_braingnn_forward_with_roi_aware_conv(self) -> None:
        """BrainGNNModel forward should return [B, 1] with ROIAwareConv."""
        from src.models.braingnn_model import BrainGNNModel

        num_nodes = 10
        node_feat_dim = 10
        edge_feat_dim = 1
        B = 2

        cfg = ModelConfig(
            name="braingnn",
            hidden_dim=16,
            dropout=0.0,
            model_params={"pool_ratio": 0.5, "roi_embed_dim": 8},
        )
        model = BrainGNNModel(
            cfg, node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim, num_nodes=num_nodes,
        )
        model.eval()

        data = self._make_batch(
            num_graphs=B, num_nodes=num_nodes,
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
        )

        with torch.no_grad():
            out = model(data)

        assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"

    def test_braingnn_no_roi_embedding_attr(self) -> None:
        """BrainGNNModel with ROIAwareConv must NOT have roi_embedding attribute."""
        from src.models.braingnn_model import BrainGNNModel

        cfg = ModelConfig(
            name="braingnn",
            hidden_dim=16,
            dropout=0.0,
            model_params={"pool_ratio": 0.5, "roi_embed_dim": 8},
        )
        model = BrainGNNModel(
            cfg, node_feat_dim=10, edge_feat_dim=1, num_nodes=10,
        )
        assert not hasattr(model, "roi_embedding"), (
            "roi_embedding should be removed; ROIAwareConv handles per-ROI projection"
        )


# ---------------------------------------------------------------------------
# Hierarchical dual readout tests (Priority 3)
# ---------------------------------------------------------------------------

def _make_braingnn(num_nodes: int = 10, node_feat_dim: int = 10, hidden_dim: int = 32):
    """Instantiate a BrainGNNModel with the given parameters."""
    from src.models.braingnn_model import BrainGNNModel

    cfg = ModelConfig(
        name="braingnn",
        hidden_dim=hidden_dim,
        dropout=0.0,
        model_params={"pool_ratio": 0.5, "roi_embed_dim": 8},
    )
    return BrainGNNModel(
        cfg, node_feat_dim=node_feat_dim, edge_feat_dim=1, num_nodes=num_nodes,
    )


class TestBrainGNNHierarchicalReadout:
    """Tests for hierarchical dual readout (max+mean at both pool stages)."""

    def _make_batch(
        self,
        num_graphs: int = 2,
        num_nodes: int = 10,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 1,
        num_edges_per_graph: int = 20,
    ) -> torch_geometric.data.Data:
        """Build a batched synthetic PyG Data object."""
        all_x, all_ei, all_ea, all_batch = [], [], [], []
        offset = 0
        for g in range(num_graphs):
            x = torch.randn(num_nodes, node_feat_dim)
            ei = torch.randint(0, num_nodes, (2, num_edges_per_graph)) + offset
            ea = torch.randn(num_edges_per_graph, edge_feat_dim)
            batch_g = torch.full((num_nodes,), g, dtype=torch.long)
            all_x.append(x)
            all_ei.append(ei)
            all_ea.append(ea)
            all_batch.append(batch_g)
            offset += num_nodes

        return torch_geometric.data.Data(
            x=torch.cat(all_x, dim=0),
            edge_index=torch.cat(all_ei, dim=1),
            edge_attr=torch.cat(all_ea, dim=0),
            batch=torch.cat(all_batch, dim=0),
            y=torch.randn(num_graphs, 1),
        )

    def test_encode_output_shape(self) -> None:
        """encode() should return [B, hidden_dim * 4] with dual readout."""
        hidden_dim = 32
        num_nodes = 10
        B = 2
        model = _make_braingnn(num_nodes=num_nodes, node_feat_dim=10, hidden_dim=hidden_dim)
        model.eval()

        data = self._make_batch(num_graphs=B, num_nodes=num_nodes)
        with torch.no_grad():
            embedding = model.encode(data)

        expected_dim = hidden_dim * 4
        assert embedding.shape == (B, expected_dim), (
            f"encode() should return [{B}, {expected_dim}], got {list(embedding.shape)}"
        )

    def test_head_input_dim(self) -> None:
        """RegressionHead's first Linear layer in_features should be hidden_dim * 4."""
        hidden_dim = 32
        model = _make_braingnn(num_nodes=10, node_feat_dim=10, hidden_dim=hidden_dim)

        first_linear = model.head.layers[0]
        assert isinstance(first_linear, torch.nn.Linear), (
            "Expected first layer of head.layers to be nn.Linear"
        )
        expected_in = hidden_dim * 4
        assert first_linear.in_features == expected_in, (
            f"head first Linear in_features should be {expected_in}, "
            f"got {first_linear.in_features}"
        )


# ---------------------------------------------------------------------------
# Sigmoid TopKPooling nonlinearity tests (Priority 5)
# ---------------------------------------------------------------------------

class TestBrainGNNSigmoidPooling:
    """Tests that TopKPooling uses sigmoid (not tanh) as score nonlinearity."""

    def test_pool1_uses_sigmoid(self) -> None:
        """pool1 should use torch.sigmoid as the TopKPooling score nonlinearity."""
        model = _make_braingnn(num_nodes=10, node_feat_dim=10, hidden_dim=32)
        assert model.pool1.select.act is torch.sigmoid, (
            f"pool1 nonlinearity should be torch.sigmoid, got {model.pool1.select.act}"
        )

    def test_pool2_uses_sigmoid(self) -> None:
        """pool2 should use torch.sigmoid as the TopKPooling score nonlinearity."""
        model = _make_braingnn(num_nodes=10, node_feat_dim=10, hidden_dim=32)
        assert model.pool2.select.act is torch.sigmoid, (
            f"pool2 nonlinearity should be torch.sigmoid, got {model.pool2.select.act}"
        )


# ---------------------------------------------------------------------------
# Adjacency augmentation after pool1 tests (Priority 4)
# ---------------------------------------------------------------------------

def _augment_adj(
    edge_index: torch.Tensor,
    edge_attr: "Optional[torch.Tensor]",
    n: int,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Standalone helper: compute 2-hop adjacency A² = A@A.

    Returns (edge_index, edge_attr) with self-loops removed.
    edge_attr is a 1-D weight tensor.
    """
    from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
    from torch_sparse import spspmm

    if edge_attr is not None:
        ew = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
    else:
        ew = edge_index.new_ones(edge_index.size(1), dtype=torch.float)

    edge_index, ew = add_self_loops(edge_index, ew, num_nodes=n)
    edge_index, ew = sort_edge_index(edge_index, ew, num_nodes=n)
    edge_index, ew = spspmm(edge_index, ew, edge_index, ew, n, n, n)
    edge_index, ew = remove_self_loops(edge_index, ew)
    return edge_index, ew.unsqueeze(-1)


class TestBrainGNNAdjAugmentation:
    """Tests for 2-hop adjacency augmentation inserted after pool1."""

    def _make_batch(
        self,
        num_graphs: int = 2,
        num_nodes: int = 10,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 1,
        num_edges_per_graph: int = 20,
    ) -> "torch_geometric.data.Data":
        """Build a batched synthetic PyG Data object."""
        all_x, all_ei, all_ea, all_batch = [], [], [], []
        offset = 0
        for g in range(num_graphs):
            x = torch.randn(num_nodes, node_feat_dim)
            ei = torch.randint(0, num_nodes, (2, num_edges_per_graph)) + offset
            ea = torch.randn(num_edges_per_graph, edge_feat_dim)
            batch_g = torch.full((num_nodes,), g, dtype=torch.long)
            all_x.append(x)
            all_ei.append(ei)
            all_ea.append(ea)
            all_batch.append(batch_g)
            offset += num_nodes

        return torch_geometric.data.Data(
            x=torch.cat(all_x, dim=0),
            edge_index=torch.cat(all_ei, dim=1),
            edge_attr=torch.cat(all_ea, dim=0),
            batch=torch.cat(all_batch, dim=0),
            y=torch.randn(num_graphs, 1),
        )

    def test_encode_runs_with_adj_aug(self) -> None:
        """encode() with adjacency augmentation should return [B, hidden_dim*4]."""
        hidden_dim = 32
        num_nodes = 10
        B = 2

        model = _make_braingnn(num_nodes=num_nodes, node_feat_dim=10, hidden_dim=hidden_dim)
        model.eval()

        data = self._make_batch(num_graphs=B, num_nodes=num_nodes)
        with torch.no_grad():
            embedding = model.encode(data)

        expected_dim = hidden_dim * 4
        assert embedding.shape == (B, expected_dim), (
            f"encode() should return [{B}, {expected_dim}], got {list(embedding.shape)}"
        )

    def test_adj_aug_increases_edge_count(self) -> None:
        """2-hop adjacency expansion should produce more edges than the 1-hop input.

        Uses a simple path graph: 0-1-2-3-4.  The 2-hop A² connects
        nodes that are 1 or 2 hops apart, so more edges than the original.
        """
        # Path graph: 0→1→2→3→4 (undirected)
        n = 5
        src = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4])
        dst = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3])
        edge_index = torch.stack([src, dst])
        edge_attr = torch.ones(edge_index.size(1), 1)

        ei_aug, ea_aug = _augment_adj(edge_index, edge_attr, n)

        assert ei_aug.size(1) > edge_index.size(1), (
            f"Augmented edge count ({ei_aug.size(1)}) should exceed "
            f"original ({edge_index.size(1)}) for a path graph"
        )
