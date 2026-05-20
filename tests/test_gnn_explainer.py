"""Tests for the GNNExplainerRunner — covers both layouts.

Issue #4: GNNExplainerRunner.run() must walk the new ``rep_<R>/fold_<K>/``
layout produced by ``NestedCrossValidator`` while staying backwards-compatible
with the legacy ``fold_<K>/`` layout written by the old ``CrossValidator`` and
HydraSweep paths.

All tests use a tiny synthetic dataset and pre-fabricated on-disk checkpoint
trees so they avoid real training time. ``GNNExplainer`` is configured with
``epochs=1`` to keep mask-optimisation overhead negligible.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedKFold

from src.configs.explainer_config import ExplainerConfig
from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from src.gnn_explainer.explainer import GNNExplainerRunner
from src.models.registry import get_model
from src.training.label_normalizer import LabelNormalizer


# ---------------------------------------------------------------------------
# Synthetic data & checkpoint-tree builders
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n_subjects: int = 30,
    n_rois: int = 6,
    n_edges: int = 16,
    feat_dim: int = 3,
    seed: int = 0,
) -> Tuple[List, np.ndarray]:
    """Build a tiny PyG dataset compatible with the MLP model."""
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    graphs: List = []
    labels: List[float] = []
    for s in range(n_subjects):
        x = torch.tensor(rng.normal(size=(n_rois, feat_dim)), dtype=torch.float32)
        rows = torch.randint(0, n_rois, (n_edges,))
        cols = torch.randint(0, n_rois, (n_edges,))
        edge_index = torch.stack([rows, cols], dim=0)
        edge_attr = torch.rand(n_edges, 1, dtype=torch.float32)
        y = float(x[:, 0].mean().item()) + rng.normal(scale=0.05)
        g = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_rois
        )
        g.subject_id = f"S{s:03d}"
        graphs.append(g)
        labels.append(y)
    return graphs, np.asarray(labels, dtype=np.float32)


def _make_model_cfg(feat_dim: int = 3) -> ModelConfig:
    """A GCN unimodal model — its message passing exposes per-edge gradients
    that GNNExplainer needs (the MLP model flattens edge_index into a dense
    adjacency vector, which makes edge masks untrainable)."""
    return ModelConfig(
        name="unimodal",
        backbone="gcn",
        hidden_dim=8,
        num_layers=1,
        pooling="mean",
        head_hidden_dim=4,
        head_num_layers=1,
        embedding_dim=4,
    )


def _make_feature_cfg_dict(feat_dim: int = 3) -> dict:
    """Minimal FeatureConfig payload accepted by CheckpointManager.load_model_for_fold."""
    return {
        "node_features": ["random"],
        "edge_features": ["weight"],
        "node_feat_dim": feat_dim,
        "edge_feat_dim": 1,
    }


def _model_factory(feat_dim: int = 3, num_nodes: int = 6):
    cfg = _make_model_cfg(feat_dim)

    def _factory():
        return get_model(
            name=cfg.name,
            cfg=cfg,
            node_feat_dim=feat_dim,
            edge_feat_dim=1,
            num_nodes=num_nodes,
        )

    return _factory


def _write_fold_artifacts(
    fold_dir: Path,
    model: torch.nn.Module,
    normalizer: LabelNormalizer,
    model_cfg: ModelConfig,
    feature_cfg: dict,
) -> None:
    """Write the on-disk files CheckpointManager.load_model_for_fold expects."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model_best.pt")
    torch.save(model.state_dict(), fold_dir / "model_last.pt")
    with open(fold_dir / "normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    with open(fold_dir / "model_config.json", "w") as f:
        json.dump(model_cfg.model_dump(), f)
    with open(fold_dir / "feature_config.json", "w") as f:
        json.dump(feature_cfg, f)
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "best": {"metrics": {"mae": 0.1}, "epoch": 0},
                "last": {"metrics": {"mae": 0.1}, "epoch": 0},
            },
            f,
        )


def _build_nested_checkpoint_tree(
    ckpt_root: Path,
    labels: np.ndarray,
    *,
    n_repetitions: int,
    n_outer_folds: int,
    stratify_bins: int = 4,
    feat_dim: int = 3,
    num_nodes: int = 6,
) -> List[int]:
    """Fabricate ``rep_<R>/fold_<K>/`` + ``nested_cv_result.json`` on disk.

    Returns the ``outer_seeds`` used so tests can reconstruct splits.
    """
    import pandas as pd

    outer_seeds = [13 + r for r in range(n_repetitions)]
    bins = pd.qcut(labels, q=stratify_bins, labels=False, duplicates="drop")

    model_cfg = _make_model_cfg(feat_dim)
    feature_cfg = _make_feature_cfg_dict(feat_dim)
    factory = _model_factory(feat_dim, num_nodes)

    for rep, seed in enumerate(outer_seeds):
        skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
        for fold_idx, (_train_val_idx, _test_idx) in enumerate(
            skf.split(np.arange(len(labels)), bins)
        ):
            fold_dir = ckpt_root / f"rep_{rep}" / f"fold_{fold_idx}"
            normalizer = LabelNormalizer(strategy="standard")
            normalizer.fit(labels)
            _write_fold_artifacts(
                fold_dir, factory(), normalizer, model_cfg, feature_cfg
            )

    nested_payload = {
        "run_name": "synthetic",
        "model_name": "mlp",
        "n_repetitions": n_repetitions,
        "n_outer_folds": n_outer_folds,
        "inner_hpo_trials": 0,
        "hpo_metric": "val_mae",
        "outer_seeds": outer_seeds,
        "fold_results": [],
        "mean_metrics": {},
        "std_metrics": {},
    }
    with open(ckpt_root / "nested_cv_result.json", "w") as f:
        json.dump(nested_payload, f)
    return outer_seeds


def _build_legacy_checkpoint_tree(
    ckpt_root: Path,
    *,
    n_folds: int,
    feat_dim: int = 3,
    num_nodes: int = 6,
    labels: np.ndarray,
) -> None:
    """Fabricate the legacy ``fold_<K>/`` layout (no rep dirs, no nested json)."""
    model_cfg = _make_model_cfg(feat_dim)
    feature_cfg = _make_feature_cfg_dict(feat_dim)
    factory = _model_factory(feat_dim, num_nodes)

    for fold_idx in range(n_folds):
        fold_dir = ckpt_root / f"fold_{fold_idx}"
        normalizer = LabelNormalizer(strategy="standard")
        normalizer.fit(labels)
        _write_fold_artifacts(
            fold_dir, factory(), normalizer, model_cfg, feature_cfg
        )


def _make_trainer_cfg(checkpoint_dir: Path, *, n_outer_folds: int, n_repetitions: int,
                      stratify_bins: int = 4, n_folds: int = 5) -> TrainerConfig:
    return TrainerConfig(
        epochs=1,
        early_stopping_patience=1,
        n_folds=n_folds,
        n_outer_folds=n_outer_folds,
        n_repetitions=n_repetitions,
        inner_hpo_trials=0,
        hpo_metric="val_mae",
        checkpoint_dir=str(checkpoint_dir),
        stratify_bins=stratify_bins,
        device="cpu",
        seed=11,
    )


# ---------------------------------------------------------------------------
# Cycle 1 — Tracer test: nested layout produces per-(rep, fold) outputs
# ---------------------------------------------------------------------------


class TestNestedCheckpointLayout:
    """The explainer walks ``rep_<R>/fold_<K>/`` and writes outputs there."""

    def test_writes_importance_matrix_per_rep_and_fold(
        self, tmp_path: Path
    ) -> None:
        graphs, labels = _make_synthetic_dataset(n_subjects=24, n_rois=6, seed=0)
        ckpt_root = tmp_path / "ckpt"
        ckpt_root.mkdir()

        _build_nested_checkpoint_tree(
            ckpt_root,
            labels,
            n_repetitions=2,
            n_outer_folds=3,
            stratify_bins=4,
            num_nodes=6,
        )

        trainer_cfg = _make_trainer_cfg(
            ckpt_root, n_outer_folds=3, n_repetitions=2, stratify_bins=4
        )
        explainer_cfg = ExplainerConfig(
            enabled=True,
            epochs=1,
            edge_size=0.1,
            save_subject_masks=False,
        )

        runner = GNNExplainerRunner(cfg=explainer_cfg)
        output_dir = runner.run(
            dataset=graphs,
            labels=labels,
            model_factory=_model_factory(feat_dim=3, num_nodes=6),
            trainer_cfg=trainer_cfg,
        )

        assert output_dir == ckpt_root / "explanations"
        for rep in range(2):
            for fold in range(3):
                path = output_dir / f"rep_{rep}" / f"fold_{fold}" / "importance_matrix.npy"
                assert path.exists(), f"missing nested output: {path}"
                mat = np.load(path)
                assert mat.shape == (6, 6)
                assert mat.dtype == np.float32
