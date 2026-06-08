"""TDD tests for Task A1: per-fold OOF predictions on CVResult + artifact converter.

Tests are written RED-first. They will fail until the implementation is added.

Four tests:
1. OOF surfaced: CVResult exposes per-fold y_true/y_pred arrays + split sizes.
2. Pooled equivalence: artifact fold_results concatenated matches CVResult.aggregated.
3. Per-fold metrics match: artifact fold_results[i].outer_test_metrics == CVResult.fold_test_metrics[i].
4. Readable by recompute tool: save artifact, run scripts/pooled_vs_meanfolds.py as subprocess.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_geometric.data

from src.configs.trainer_config import TrainerConfig
from src.models.base_model import BrainGNN
from src.training.cross_validation import CrossValidator, CVResult, build_flat_cv_artifact
from src.training.metrics import compute_metrics
from src.training.trainer import TrainResult


# ---------------------------------------------------------------------------
# Minimal fake model / trainer (mirrored from test_training_regressions.py)
# ---------------------------------------------------------------------------


class _ConstantModel(BrainGNN):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def encode(self, data: object) -> torch.Tensor:
        batch_size = int(data.y.view(-1).shape[0])
        return self.value.expand(batch_size, 1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding


class _FakeTrainer:
    """Fake trainer that loads best_model_state_dict and predicts a constant."""

    def __init__(self, constant: float = 2.0) -> None:
        self.constant = constant
        self.logger = MagicMock()

    def fit(self, model, train_loader, val_loader, inverse_transform, fold_idx, on_epoch_end_callback=None):
        return TrainResult(
            best_epoch=1,
            best_val_metrics={"mae": 1.0, "rmse": 1.0, "r2": 0.0, "pearson_r": 0.0},
            best_model_state_dict={"value": torch.tensor(self.constant)},
            last_model_state_dict={"value": torch.tensor(self.constant)},
            last_epoch=1,
            last_val_metrics={"mae": 1.0, "rmse": 1.0, "r2": 0.0, "pearson_r": 0.0},
        )

    def predict(self, model, loader, inverse_transform):
        y_true = np.array(
            [float(graph.y.item()) for graph in loader.dataset], dtype=np.float32
        )
        y_pred = np.full_like(y_true, float(model.value.detach().cpu().item()))
        return y_true, y_pred

    def evaluate(self, model, loader, inverse_transform, split):
        y_true, y_pred = self.predict(model, loader, inverse_transform)
        return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Shared fixture: a 2-fold CVResult over a tiny dataset
# ---------------------------------------------------------------------------


def _make_dataset(n: int):
    """Build n trivial PyG graphs with subject_id attributes."""
    return [
        torch_geometric.data.Data(
            x=torch.ones(1, 1),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            subject_id=f"subject_{i}",
        )
        for i in range(n)
    ]


@pytest.fixture()
def two_fold_cv_result(tmp_path, monkeypatch):
    """Run CrossValidator with 2 manually-specified folds and return the CVResult."""
    cfg = TrainerConfig(
        n_folds=2,
        label_norm_strategy="none",
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )
    cv = CrossValidator(cfg=cfg)
    trainer = _FakeTrainer(constant=2.0)

    # 10 subjects — fold 0: test=[0,1,2], fold 1: test=[3,4,5,6]
    # (non-equal fold sizes make aggregation non-trivial)
    dataset = _make_dataset(10)
    labels = np.arange(10, dtype=np.float32)

    monkeypatch.setattr(
        CrossValidator,
        "split",
        lambda self, dataset, labels: iter([
            ([6, 7, 8, 9], [4, 5], [0, 1, 2]),
            ([0, 1, 2, 3], [8, 9], [3, 4, 5, 6]),
        ]),
    )

    result = cv.run(
        model_factory=lambda: _ConstantModel(0.0),
        dataset=dataset,
        labels=labels,
        trainer=trainer,
    )
    return result, cfg


# ---------------------------------------------------------------------------
# Test 1: OOF predictions and split sizes are surfaced on CVResult
# ---------------------------------------------------------------------------


def test_oof_arrays_surfaced(two_fold_cv_result):
    """CVResult must expose per-fold y_true, y_pred arrays and split sizes."""
    result, _ = two_fold_cv_result

    # There must be exactly 2 folds
    assert len(result.oof_y_true) == 2
    assert len(result.oof_y_pred) == 2
    assert len(result.fold_split_sizes) == 2

    # Fold 0: test size 3
    assert len(result.oof_y_true[0]) == 3
    assert len(result.oof_y_pred[0]) == 3

    # Fold 1: test size 4
    assert len(result.oof_y_true[1]) == 4
    assert len(result.oof_y_pred[1]) == 4

    # Split sizes tuple must be (n_train, n_val, n_test)
    n_train_0, n_val_0, n_test_0 = result.fold_split_sizes[0]
    assert n_train_0 == 4  # [6,7,8,9]
    assert n_val_0 == 2    # [4,5]
    assert n_test_0 == 3   # [0,1,2]

    n_train_1, n_val_1, n_test_1 = result.fold_split_sizes[1]
    assert n_train_1 == 4  # [0,1,2,3]
    assert n_val_1 == 2    # [8,9]
    assert n_test_1 == 4   # [3,4,5,6]


# ---------------------------------------------------------------------------
# Test 2: Pooled equivalence — artifact's concatenated OOF == CVResult.aggregated
# ---------------------------------------------------------------------------


def test_pooled_equivalence(two_fold_cv_result):
    """Concatenating artifact's OOF preds must reproduce CVResult.aggregated metrics."""
    result, cfg = two_fold_cv_result

    artifact = build_flat_cv_artifact(
        result,
        cfg=cfg,
        run_name="test_run",
        model_name="ConstantModel",
    )

    # Concatenate from artifact
    art_y_true = np.concatenate([np.array(fr.y_true) for fr in artifact.fold_results])
    art_y_pred = np.concatenate([np.array(fr.y_pred) for fr in artifact.fold_results])
    art_pooled = compute_metrics(art_y_true, art_y_pred)

    # Must match CVResult.aggregated exactly
    for metric in ("r2", "mae", "rmse", "pearson_r"):
        assert art_pooled[metric] == pytest.approx(result.aggregated[metric], abs=1e-6), (
            f"Pooled {metric} from artifact ({art_pooled[metric]:.6f}) != "
            f"CVResult.aggregated ({result.aggregated[metric]:.6f})"
        )


# ---------------------------------------------------------------------------
# Test 3: Per-fold metrics in artifact match CVResult.fold_test_metrics
# ---------------------------------------------------------------------------


def test_per_fold_metrics_match(two_fold_cv_result):
    """Each artifact fold_results[i].outer_test_metrics must equal CVResult.fold_test_metrics[i]."""
    result, cfg = two_fold_cv_result

    artifact = build_flat_cv_artifact(
        result,
        cfg=cfg,
        run_name="test_run",
        model_name="ConstantModel",
    )

    assert len(artifact.fold_results) == len(result.fold_test_metrics)
    for i, (art_fr, cv_metrics) in enumerate(
        zip(artifact.fold_results, result.fold_test_metrics)
    ):
        for metric in ("r2", "mae", "rmse", "pearson_r"):
            assert art_fr.outer_test_metrics[metric] == pytest.approx(
                cv_metrics[metric], abs=1e-6
            ), (
                f"Fold {i}: artifact outer_test_metrics[{metric!r}]="
                f"{art_fr.outer_test_metrics[metric]:.6f} != "
                f"CVResult.fold_test_metrics[{i}][{metric!r}]={cv_metrics[metric]:.6f}"
            )


# ---------------------------------------------------------------------------
# Test 4: Artifact is readable by scripts/pooled_vs_meanfolds.py
# ---------------------------------------------------------------------------


def test_artifact_readable_by_recompute_tool(two_fold_cv_result, tmp_path):
    """Save artifact to disk and verify scripts/pooled_vs_meanfolds.py exits 0."""
    result, cfg = two_fold_cv_result

    artifact = build_flat_cv_artifact(
        result,
        cfg=cfg,
        run_name="test_run",
        model_name="ConstantModel",
    )

    artifact_path = tmp_path / "nested_cv_result.json"
    artifact.save(artifact_path)

    # Run the recompute tool as a subprocess
    repo_root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/pooled_vs_meanfolds.py",
            str(artifact_path),
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": repo_root},
    )

    assert proc.returncode == 0, (
        f"pooled_vs_meanfolds.py failed (rc={proc.returncode}):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )

    # Verify the printed pooled R² matches our expected value
    # CVResult.aggregated["r2"] was computed from the same pooled predictions
    pooled_r2 = result.aggregated["r2"]
    # The tool prints: "  pooled (all folds): +X.XXXX  over N preds"
    match = re.search(r"pooled \(all folds\):\s*([+-]?\d+\.\d+)", proc.stdout)
    assert match is not None, (
        f"Could not find 'pooled (all folds)' in tool output:\n{proc.stdout}"
    )
    reported_r2 = float(match.group(1))
    assert reported_r2 == pytest.approx(pooled_r2, abs=1e-3), (
        f"Tool reported pooled R²={reported_r2:.4f}, expected {pooled_r2:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: build_flat_cv_artifact raises ValueError on mismatched fold_split_sizes
# ---------------------------------------------------------------------------


def test_build_flat_cv_artifact_raises_on_mismatched_split_sizes():
    """build_flat_cv_artifact must raise ValueError when fold_split_sizes length
    does not match oof_y_true length (i.e. a legacy or hand-built CVResult)."""
    bad_result = CVResult(
        oof_y_true=[np.array([1.0, 2.0]), np.array([3.0])],
        oof_y_pred=[np.array([1.5, 2.5]), np.array([3.5])],
        fold_split_sizes=[],  # missing — mismatch with 2 folds above
        fold_results=[],
        fold_test_metrics=[],
        aggregated={},
    )
    cfg = TrainerConfig(n_folds=2, label_norm_strategy="none", checkpoint_dir="/tmp", device="cpu")
    with pytest.raises(ValueError, match="fold_split_sizes is not populated consistently"):
        build_flat_cv_artifact(bad_result, cfg=cfg, run_name="x", model_name="y")
