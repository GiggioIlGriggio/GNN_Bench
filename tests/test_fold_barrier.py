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
