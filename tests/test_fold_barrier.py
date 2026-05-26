"""Tests for FoldBarrier and its state-dict surface."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


class TestLabelNormalizerStateDict:
    """state_dict round-trip preserves fitted statistics."""

    def test_standard_roundtrip(self) -> None:
        from src.training.label_normalizer import LabelNormalizer

        rng = np.random.default_rng(0)
        y = rng.normal(loc=3.0, scale=2.0, size=200).astype(np.float64)

        n1 = LabelNormalizer(strategy="standard")
        n1.fit(y)
        state = n1.state_dict()
        assert state["strategy"] == "standard"
        assert state["mean"] == pytest.approx(float(np.mean(y)))
        assert state["std"] == pytest.approx(float(np.std(y)))

        n2 = LabelNormalizer(strategy="standard")
        n2.load_state_dict(state)
        np.testing.assert_allclose(n2.transform(y), n1.transform(y))
        np.testing.assert_allclose(
            n2.inverse_transform(n2.transform(y)), y, atol=1e-6
        )


class TestGLMNormalizerStateDict:
    """state_dict carries (mean, std, columns) and survives a roundtrip."""

    def _make_graphs(self, n: int = 8, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        for _ in range(n):
            x = torch.tensor(
                rng.normal(size=(n_rois, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
        return graphs

    def test_roundtrip(self) -> None:
        from src.training.glm_normalizer import GLMFeatureNormalizer

        graphs = self._make_graphs(n=12, n_rois=5)
        n1 = GLMFeatureNormalizer(col_start=1, col_end=3)
        n1.fit(graphs)

        state = n1.state_dict()
        assert state["col_start"] == 1
        assert state["col_end"] == 3
        assert state["fitted"] is True

        n2 = GLMFeatureNormalizer(col_start=0, col_end=0)  # placeholders
        n2.load_state_dict(state)
        assert n2.col_start == 1 and n2.col_end == 3
        assert torch.allclose(n2.mean_, n1.mean_)
        assert torch.allclose(n2.std_, n1.std_)


class TestLabelBuilderStateDict:
    """state_dict carries component normalisation stats + composite op state."""

    def _cfg_single(self):
        from src.configs.label_config import LabelConfig
        return LabelConfig(target="score", id_column="ID")

    def test_single_column_roundtrip(self) -> None:
        import pandas as pd
        from src.datasets.label_builder import LabelBuilder

        cfg = self._cfg_single()
        lb1 = LabelBuilder(cfg)
        components = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
        lb1.fit(components)
        state = lb1.state_dict()

        lb2 = LabelBuilder(cfg)
        lb2.load_state_dict(state)
        np.testing.assert_allclose(
            lb2.transform(components), lb1.transform(components),
        )


class TestFoldBarrierLabels:
    """Round-trip on label normalisation (non-composite mode)."""

    def _make_graphs(self, n: int = 30, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(n):
            x = torch.tensor(
                rng.normal(size=(n_rois, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
            labels.append(float(rng.normal(loc=2.0, scale=1.5)))
        return graphs, np.array(labels, dtype=np.float64)

    def test_label_roundtrip_non_composite(self) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(seed=1)
        barrier = FoldBarrier(label_norm_strategy="standard")
        barrier.fit(graphs, labels)

        y_norm = barrier.transform_labels(labels)
        assert abs(float(np.mean(y_norm))) < 1e-6
        assert abs(float(np.std(y_norm)) - 1.0) < 1e-6

        y_recovered = barrier.inverse_transform_labels(y_norm)
        np.testing.assert_allclose(y_recovered, labels, atol=1e-6)


class TestFoldBarrierGraphs:
    """transform_graphs returns NEW graphs; original dataset stays untouched."""

    def _make_graphs(self, n: int = 12, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(n):
            x = torch.tensor(
                rng.normal(size=(n_rois, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
            labels.append(float(rng.normal()))
        return graphs, np.array(labels, dtype=np.float64)

    def test_inputs_not_mutated(self) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(seed=2)
        originals = [g.x.clone() for g in graphs]

        barrier = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        barrier.fit(graphs, labels)
        _ = barrier.transform_graphs(graphs)

        for orig, g in zip(originals, graphs):
            assert torch.allclose(g.x, orig), (
                "FoldBarrier.transform_graphs must not mutate input graphs."
            )

    def test_two_fold_sequence_does_not_corrupt_dataset(self) -> None:
        """Two folds in sequence — the second fold must see the original data."""
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(seed=3)
        snapshot = [g.x.clone() for g in graphs]

        for fold in range(2):
            barrier = FoldBarrier(
                label_norm_strategy="standard", glm_col_range=(1, 3),
            )
            barrier.fit(graphs, labels)
            _ = barrier.transform_graphs(graphs)

        for orig, g in zip(snapshot, graphs):
            assert torch.allclose(g.x, orig), (
                "After two folds, original graphs must remain unchanged."
            )

    def test_glm_columns_are_zscored_on_train(self) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(n=80, seed=4)
        barrier = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        barrier.fit(graphs, labels)
        transformed = barrier.transform_graphs(graphs)

        stacked = torch.stack([g.x[:, 1:3] for g in transformed], dim=0)
        # mean ≈ 0 per-(node, col) since fit was on the same set
        assert stacked.mean(dim=0).abs().max().item() < 1e-5


class TestFoldBarrierPersistence:
    """save() / load() round-trip preserves transform behaviour."""

    def _make(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(40):
            x = torch.tensor(
                rng.normal(size=(6, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=6,
            )
            graphs.append(g)
            labels.append(float(rng.normal(loc=1.0, scale=2.0)))
        return graphs, np.array(labels, dtype=np.float64)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make(seed=5)
        b1 = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        b1.fit(graphs, labels)

        b1.save(tmp_path / "barrier.pt")
        assert (tmp_path / "barrier.pt").exists()

        b2 = FoldBarrier.load(
            tmp_path / "barrier.pt",
            label_norm_strategy="standard",
            glm_col_range=(1, 3),
        )

        y_norm_a = b1.transform_labels(labels)
        y_norm_b = b2.transform_labels(labels)
        np.testing.assert_allclose(y_norm_a, y_norm_b)

        ga = b1.transform_graphs(graphs)
        gb = b2.transform_graphs(graphs)
        for x_a, x_b in zip(ga, gb):
            assert torch.allclose(x_a.x, x_b.x)
