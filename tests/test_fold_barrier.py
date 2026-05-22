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
