"""Tests for FoldCheckpoint (the per-fold read path, ADR-0009 / PR2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


class TestFoldCheckpointConstruction:
    """``FoldCheckpoint(fold_dir)`` and ``FoldCheckpoint.for_fold(root, idx)``."""

    def test_direct_construction_records_dir(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fold_dir = tmp_path / "fold_3"
        fold_dir.mkdir()
        fc = FoldCheckpoint(fold_dir)
        assert fc.fold_dir == fold_dir

    def test_for_fold_resolves_subdir(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fc = FoldCheckpoint.for_fold(tmp_path, 7)
        assert fc.fold_dir == tmp_path / "fold_7"


class TestLoadStateDict:
    """``load_state_dict(variant=...)`` returns the saved model weights."""

    def _seed_fold_dir(self, fold_dir: Path) -> tuple[dict, dict]:
        fold_dir.mkdir(parents=True, exist_ok=True)
        best = {"layer.weight": torch.tensor([1.0, 2.0])}
        last = {"layer.weight": torch.tensor([3.0, 4.0])}
        torch.save(best, fold_dir / "model_best.pt")
        torch.save(last, fold_dir / "model_last.pt")
        return best, last

    def test_loads_best_by_default(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        best, _ = self._seed_fold_dir(tmp_path / "fold_0")
        loaded = FoldCheckpoint(tmp_path / "fold_0").load_state_dict()
        torch.testing.assert_close(loaded["layer.weight"], best["layer.weight"])

    def test_loads_last_variant(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        _, last = self._seed_fold_dir(tmp_path / "fold_0")
        loaded = FoldCheckpoint(tmp_path / "fold_0").load_state_dict(variant="last")
        torch.testing.assert_close(loaded["layer.weight"], last["layer.weight"])

    def test_rejects_unknown_variant(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        self._seed_fold_dir(tmp_path / "fold_0")
        with pytest.raises(ValueError, match="variant"):
            FoldCheckpoint(tmp_path / "fold_0").load_state_dict(variant="median")


class TestLoadBarrier:
    """``load_barrier()`` reconstructs the FoldBarrier from ``barrier.pt``."""

    def _make_graphs(self, n: int = 20, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(n):
            x = torch.tensor(rng.normal(size=(n_rois, 4)), dtype=torch.float32)
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
            labels.append(float(rng.normal()))
        return graphs, np.array(labels, dtype=np.float64)

    def test_returns_none_when_barrier_missing(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        (tmp_path / "fold_0").mkdir()
        assert FoldCheckpoint(tmp_path / "fold_0").load_barrier() is None

    def test_reconstructs_glm_from_persisted_state(self, tmp_path: Path) -> None:
        from src.training.fold_barrier import FoldBarrier
        from src.training.fold_checkpoint import FoldCheckpoint

        graphs, labels = self._make_graphs(n=40, seed=1)
        fitted = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        fitted.fit(graphs, labels)

        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        fitted.save(fold_dir / "barrier.pt")

        loaded = FoldCheckpoint(fold_dir).load_barrier()
        assert loaded is not None

        for g_a, g_b in zip(
            fitted.transform_graphs(graphs),
            loaded.transform_graphs(graphs),
        ):
            assert torch.allclose(g_a.x, g_b.x)


class TestLoadMetrics:
    """``load_metrics()`` returns the parsed ``metrics.json``."""

    def test_returns_parsed_json(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        payload = {
            "best": {"metrics": {"mae": 0.1}, "epoch": 3},
            "last": {"metrics": {"mae": 0.2}, "epoch": 9},
        }
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(payload, f)

        assert FoldCheckpoint(fold_dir).load_metrics() == payload


class TestLoadBundle:
    """``load_bundle()`` returns a populated FoldBundle from disk."""

    def _seed_fold_dir(self, fold_dir: Path) -> None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"w": torch.tensor([1.0])}, fold_dir / "model_best.pt",
        )
        torch.save(
            {"w": torch.tensor([2.0])}, fold_dir / "model_last.pt",
        )
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "best": {"metrics": {"mae": 0.1}, "epoch": 1},
                    "last": {"metrics": {"mae": 0.2}, "epoch": 9},
                },
                f,
            )

    def test_populates_all_fields_without_barrier(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldBundle, FoldCheckpoint

        fold_dir = tmp_path / "fold_5"
        self._seed_fold_dir(fold_dir)

        bundle = FoldCheckpoint(fold_dir).load_bundle()
        assert isinstance(bundle, FoldBundle)
        assert bundle.fold_idx == 5
        assert bundle.barrier is None
        assert bundle.best_epoch == 1 and bundle.last_epoch == 9
        assert bundle.best_metrics == {"mae": 0.1}
        assert bundle.last_metrics == {"mae": 0.2}
        torch.testing.assert_close(
            bundle.best_model_state_dict["w"], torch.tensor([1.0]),
        )
        torch.testing.assert_close(
            bundle.last_model_state_dict["w"], torch.tensor([2.0]),
        )

    def test_fold_idx_parsed_from_dir_name(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fold_dir = tmp_path / "fold_42"
        self._seed_fold_dir(fold_dir)
        assert FoldCheckpoint(fold_dir).load_bundle().fold_idx == 42

    def test_raises_on_malformed_dir_name(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        weird_dir = tmp_path / "not_a_fold"
        self._seed_fold_dir(weird_dir)
        with pytest.raises(ValueError, match="fold directory name"):
            FoldCheckpoint(weird_dir).load_bundle()
