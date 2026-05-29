"""Tests for the training module.

All tests use synthetic data and mock objects.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.configs.trainer_config import TrainerConfig
from src.training.label_normalizer import LabelNormalizer
from src.training.metrics import (
    AggregatedMetrics,
    MetricDict,
    aggregate_fold_metrics,
    compute_metrics,
)
from src.training.fold_checkpoint import CheckpointManager, FoldBundle


# ---------------------------------------------------------------------------
# Label normalizer
# ---------------------------------------------------------------------------

class TestLabelNormalizer:
    """Tests for per-fold label normalisation."""

    # def test_standard_fit_transform(self) -> None:
        # """Standard normalisation should produce mean≈0, std≈1."""
        # raise NotImplementedError(
            # "TODO: fit on random data, transform, "
            # "assert np.allclose(mean(transformed), 0, atol=1e-6)"
        # )

    # def test_inverse_recovers_original(self) -> None:
        # """inverse_transform(transform(y)) should recover original values."""
        # raise NotImplementedError(
            # "TODO: fit, transform, inverse_transform, assert np.allclose"
        # )

    # def test_robust_strategy(self) -> None:
        # """Robust normalisation should use median and IQR."""
        # raise NotImplementedError(
            # "TODO: instantiate with strategy='robust', fit, transform, check"
        # )

    # def test_minmax_strategy(self) -> None:
        # """MinMax normalisation should scale to [0, 1]."""
        # raise NotImplementedError(
            # "TODO: instantiate with strategy='minmax', fit, transform, "
            # "assert min≈0, max≈1"
        # )

    # def test_none_strategy_identity(self) -> None:
        # """'none' strategy should be an identity transform."""
        # raise NotImplementedError(
            # "TODO: fit, transform, assert output == input"
        # )

    # def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        # """Serialisation should preserve normaliser state."""
        # raise NotImplementedError(
            # "TODO: fit, save, load, transform, assert same result"
        # )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for metric computation."""

    # def test_perfect_prediction(self) -> None:
        # """Perfect predictions should give MAE=0, R²=1, Pearson r=1."""
        # raise NotImplementedError(
            # "TODO: y_true == y_pred, assert metrics match expected values"
        # )

    # def test_compute_metrics_keys(self) -> None:
        # """compute_metrics should return all four metric keys."""
        # raise NotImplementedError(
            # "TODO: assert keys == {'mae', 'rmse', 'r2', 'pearson_r'}"
        # )

    # def test_aggregate_fold_metrics(self) -> None:
        # """Aggregation should produce mean, std, and 95% CI."""
        # raise NotImplementedError(
            # "TODO: create 3 MetricDicts, aggregate, assert all fields populated"
        # )


# ---------------------------------------------------------------------------
# Trainer (one fold cycle)
# ---------------------------------------------------------------------------

class TestTrainer:
    """Tests for the Trainer."""

    # def test_fit_returns_train_result(self) -> None:
        # """fit() should return a TrainResult with history and best metrics."""
        # raise NotImplementedError(
            # "TODO: mock model and loaders, call fit, assert TrainResult type"
        # )

    # def test_evaluate_returns_metric_dict(self) -> None:
        # """evaluate() should return a MetricDict."""
        # raise NotImplementedError(
            # "TODO: mock model and loader, call evaluate, assert MetricDict"
        # )


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    """Tests for checkpoint save/load."""

    # def test_save_creates_directory(self, tmp_path: Path) -> None:
        # """save_fold_checkpoint should create the fold directory."""
        # raise NotImplementedError(
            # "TODO: save a mock checkpoint, assert directory exists"
        # )

    # def test_load_recovers_state(self, tmp_path: Path) -> None:
        # """FoldCheckpoint.load_bundle should recover saved state."""
        # raise NotImplementedError(
            # "TODO: save then load via FoldCheckpoint.load_bundle, assert metrics match"
        # )

    # def test_load_missing_raises(self, tmp_path: Path) -> None:
        # """Loading a non-existent fold should raise FileNotFoundError."""
        # raise NotImplementedError(
            # "TODO: call FoldCheckpoint(missing_dir).load_bundle(), assert FileNotFoundError"
        # )


# ---------------------------------------------------------------------------
# Nested cross-validation (ADR-0008)
# ---------------------------------------------------------------------------

class TestSearchSpaceParser:
    """Tests for the DSL parser in src.training.search_space."""

    def test_choice_int_float_string(self) -> None:
        """choice(...) must capture int / float / unquoted-string options."""
        from src.training.search_space import parse_search_space

        params = {
            "a": "choice(32, 64, 128)",
            "b": "choice(0.0, 0.1, 0.2)",
            "c": "choice(mean, max, add)",
        }
        specs = {s.name: s for s in parse_search_space(params)}
        assert specs["a"].kind == "categorical" and specs["a"].choices == [32, 64, 128]
        assert specs["b"].choices == [0.0, 0.1, 0.2]
        assert specs["c"].choices == ["mean", "max", "add"]

    def test_range_is_python_halfopen(self) -> None:
        """range(2, 5) must yield closed [2, 4] (Python range semantics)."""
        from src.training.search_space import parse_search_space

        spec = parse_search_space({"x": "range(2, 5)"})[0]
        assert spec.kind == "int"
        assert spec.low == 2
        assert spec.high == 4
        assert spec.step == 1

    def test_interval_and_log_tag(self) -> None:
        """tag(log, interval(a,b)) must set log=True; interval alone stays linear."""
        from src.training.search_space import parse_search_space

        plain = parse_search_space({"x": "interval(0.0, 1.0)"})[0]
        logged = parse_search_space({"x": "tag(log, interval(1e-5, 1e-2))"})[0]
        assert plain.kind == "float" and plain.log is False
        assert logged.kind == "float" and logged.log is True
        assert logged.low == 1e-5 and logged.high == 1e-2

    def test_apply_overrides_does_not_mutate(self) -> None:
        """TrialOverrides.apply returns new pydantic copies, leaving originals intact."""
        from src.configs.model_config import ModelConfig
        from src.configs.trainer_config import TrainerConfig
        from src.training.search_space import TrialOverrides

        model = ModelConfig(name="mlp", hidden_dim=64, model_params={"x": 1})
        trainer = TrainerConfig(lr=1e-3)
        ov = TrialOverrides(values={
            "model.hidden_dim": 128,
            "model.model_params.x": 99,
            "trainer.lr": 5e-4,
        })
        new_model, new_trainer = ov.apply(model_cfg=model, trainer_cfg=trainer)
        assert model.hidden_dim == 64 and model.model_params["x"] == 1
        assert trainer.lr == 1e-3
        assert new_model.hidden_dim == 128
        assert new_model.model_params["x"] == 99
        assert new_trainer.lr == 5e-4


class TestStatisticalTests:
    """Bouckaert-Frank corrected t-test + Benjamini-Hochberg."""

    def test_identical_scores_yield_p_one(self) -> None:
        from src.training.statistical_tests import corrected_resampled_paired_t_test

        t, p, df, md = corrected_resampled_paired_t_test(
            [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], n_outer_folds=2,
        )
        assert t == 0.0
        assert p == 1.0
        assert md == 0.0

    def test_correction_inflates_pvalue(self) -> None:
        """Corrected p must be larger than the naive paired t-test p."""
        import numpy as np
        from scipy import stats
        from src.training.statistical_tests import corrected_resampled_paired_t_test

        rng = np.random.default_rng(1)
        a = rng.normal(0.5, 0.1, size=50)
        b = rng.normal(0.4, 0.1, size=50)
        _, p_corr, _, _ = corrected_resampled_paired_t_test(a, b, n_outer_folds=5)
        _, p_naive = stats.ttest_rel(a, b)
        assert p_corr > p_naive

    def test_benjamini_hochberg_matches_r_reference(self) -> None:
        """Matches R's p.adjust(method='BH') on a published reference vector."""
        import numpy as np
        from src.training.statistical_tests import benjamini_hochberg

        ps = [0.001, 0.01, 0.025, 0.04, 0.06]
        adj = benjamini_hochberg(ps)
        expected = np.array([0.005, 0.025, 0.04166667, 0.05, 0.06])
        np.testing.assert_allclose(adj, expected, atol=1e-6)

    def test_benjamini_hochberg_preserves_order(self) -> None:
        """Shuffled inputs produce shuffled outputs with the same set of values."""
        import numpy as np
        from src.training.statistical_tests import benjamini_hochberg

        ps_sorted = np.array([0.001, 0.01, 0.025, 0.04, 0.06])
        adj_sorted = benjamini_hochberg(ps_sorted)
        order = np.array([4, 0, 2, 3, 1])
        ps_shuffled = ps_sorted[order]
        adj_shuffled = benjamini_hochberg(ps_shuffled)
        np.testing.assert_allclose(adj_shuffled, adj_sorted[order], atol=1e-6)


class _SyntheticNestedCV:
    """Shared fixture builder — kept module-internal."""

    @staticmethod
    def make_dataset(n_subjects: int = 60, n_rois: int = 8, seed: int = 0):
        import numpy as np
        import torch
        from torch_geometric.data import Data

        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for s in range(n_subjects):
            x = torch.tensor(rng.normal(size=(n_rois, 4)), dtype=torch.float32)
            rows = torch.randint(0, n_rois, (20,))
            cols = torch.randint(0, n_rois, (20,))
            edge_index = torch.stack([rows, cols], dim=0)
            edge_attr = torch.rand(20, 1, dtype=torch.float32)
            y_label = float(x[:, 0].mean().item()) + rng.normal(scale=0.05)
            g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_rois)
            g.subject_id = f"S{s:03d}"
            graphs.append(g)
            labels.append(y_label)
        return graphs, np.array(labels)

    @staticmethod
    def make_configs(checkpoint_dir, **overrides):
        from src.configs.model_config import ModelConfig
        from src.configs.trainer_config import TrainerConfig

        model_cfg = ModelConfig(
            name="mlp",
            hidden_dim=8,
            num_layers=1,
            head_hidden_dim=4,
            head_num_layers=1,
            mlp_input="node_features",
            mlp_adjacency_type="weighted",
            embedding_dim=4,
        )
        defaults = dict(
            epochs=2,
            early_stopping_patience=2,
            n_outer_folds=3,
            n_repetitions=1,
            inner_hpo_trials=0,
            hpo_metric="val_mae",
            checkpoint_dir=str(checkpoint_dir),
            seed=11,
        )
        defaults.update(overrides)
        trainer_cfg = TrainerConfig(**defaults)
        return model_cfg, trainer_cfg

    @staticmethod
    def make_logger():
        from src.configs.logging_config import LoggingConfig
        from src.logging.wandb_logger import WandbLogger

        return WandbLogger(LoggingConfig(enabled=False))

    @staticmethod
    def make_factory(model_cfg):
        from src.models.registry import get_model

        def _factory(trial_model_cfg):
            return get_model(
                name=trial_model_cfg.name,
                cfg=trial_model_cfg,
                node_feat_dim=4,
                edge_feat_dim=1,
                num_nodes=8,
            )

        return _factory


class TestNestedCrossValidator:
    """End-to-end tests for the nested-CV orchestrator."""

    def test_split_determinism(self) -> None:
        """Same seed → identical outer and inner index lists, both reps and folds."""
        import numpy as np
        from src.training.nested_cross_validation import NestedCrossValidator

        graphs, labels = _SyntheticNestedCV.make_dataset(seed=0)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, trainer_cfg = _SyntheticNestedCV.make_configs(td, n_repetitions=2)
            ncv = NestedCrossValidator(cfg=trainer_cfg)

            outer_a = []
            outer_b = []
            for rep_runs in (outer_a, outer_b):
                bins = ncv._stratify_bins(labels)
                seeds = trainer_cfg.resolved_outer_seeds()
                from sklearn.model_selection import StratifiedKFold
                for r, seed in enumerate(seeds):
                    skf = StratifiedKFold(
                        n_splits=trainer_cfg.effective_n_outer_folds,
                        shuffle=True, random_state=seed,
                    )
                    for k, (tv, te) in enumerate(skf.split(np.arange(len(graphs)), bins)):
                        train_idx, val_idx = ncv._inner_split(tv.tolist(), bins, seed * 1000 + k)
                        rep_runs.append((r, k, tv.tolist(), te.tolist(), train_idx, val_idx))

            assert outer_a == outer_b

    def test_no_test_leakage(self) -> None:
        """Outer-test indices never appear in the inner train or val pools of the same fold."""
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        from src.training.nested_cross_validation import NestedCrossValidator

        graphs, labels = _SyntheticNestedCV.make_dataset(seed=1)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, trainer_cfg = _SyntheticNestedCV.make_configs(td, n_repetitions=2)
            ncv = NestedCrossValidator(cfg=trainer_cfg)
            bins = ncv._stratify_bins(labels)

            for r, seed in enumerate(trainer_cfg.resolved_outer_seeds()):
                skf = StratifiedKFold(
                    n_splits=trainer_cfg.effective_n_outer_folds,
                    shuffle=True, random_state=seed,
                )
                for k, (tv, te) in enumerate(skf.split(np.arange(len(graphs)), bins)):
                    inner_seed = seed * 1000 + k
                    train_idx, val_idx = ncv._inner_split(tv.tolist(), bins, inner_seed)
                    test_set = set(te.tolist())
                    assert not (set(train_idx) & test_set), (
                        f"Inner train leaks into outer test at rep={r} fold={k}"
                    )
                    assert not (set(val_idx) & test_set), (
                        f"Inner val leaks into outer test at rep={r} fold={k}"
                    )
                    # Inner train ∪ inner val == outer TrainVal
                    assert set(train_idx) | set(val_idx) == set(tv.tolist())
                    assert not (set(train_idx) & set(val_idx))

    def test_fast_preset_smoke(self) -> None:
        """Fast preset (inner_hpo_trials=0) runs end-to-end and persists a JSON."""
        import tempfile
        from pathlib import Path
        from src.training.nested_cross_validation import (
            NestedCrossValidator, NestedCVResult,
        )

        graphs, labels = _SyntheticNestedCV.make_dataset(seed=2)
        with tempfile.TemporaryDirectory() as td:
            model_cfg, trainer_cfg = _SyntheticNestedCV.make_configs(
                td, n_repetitions=2, n_outer_folds=3, inner_hpo_trials=0,
            )
            ncv = NestedCrossValidator(cfg=trainer_cfg)
            result = ncv.run(
                model_factory=_SyntheticNestedCV.make_factory(model_cfg),
                dataset=graphs,
                labels=labels,
                base_model_cfg=model_cfg,
                logger=_SyntheticNestedCV.make_logger(),
                run_name="test-fast",
            )
            assert len(result.fold_results) == 2 * 3
            assert set(result.mean_metrics) == {"mae", "rmse", "r2", "pearson_r"}
            saved = Path(td) / "nested_cv_result.json"
            assert saved.exists()
            reloaded = NestedCVResult.load(saved)
            assert reloaded.run_name == "test-fast"
            assert reloaded.mean_metrics == result.mean_metrics

    def test_hpo_preset_smoke(self) -> None:
        """Paper-grade preset (inner_hpo_trials>0) drives Optuna and writes artifacts."""
        import tempfile
        from pathlib import Path
        from src.training.nested_cross_validation import NestedCrossValidator

        graphs, labels = _SyntheticNestedCV.make_dataset(seed=3)
        with tempfile.TemporaryDirectory() as td:
            model_cfg, trainer_cfg = _SyntheticNestedCV.make_configs(
                td, n_repetitions=1, n_outer_folds=3, inner_hpo_trials=2,
                search_space="configs/sweeper/mlp.yaml",
            )
            ncv = NestedCrossValidator(
                cfg=trainer_cfg, search_space_path=trainer_cfg.search_space,
            )
            result = ncv.run(
                model_factory=_SyntheticNestedCV.make_factory(model_cfg),
                dataset=graphs,
                labels=labels,
                base_model_cfg=model_cfg,
                logger=_SyntheticNestedCV.make_logger(),
                run_name="test-hpo",
            )
            assert len(result.fold_results) == 3
            for fr in result.fold_results:
                assert fr.best_hparams  # Optuna picked something
                fold_dir = Path(td) / f"rep_{fr.rep}" / f"fold_{fr.fold}"
                assert (fold_dir / "best_hparams.json").exists()
                assert (fold_dir / "trials.csv").exists()
                assert (fold_dir / "test_predictions.npz").exists()

    def test_refit_survives_singleton_trailing_batch(self) -> None:
        """Regression: refit on a TrainVal pool of N where N % batch_size == 1.

        Without ``drop_last`` on the train loader, ``nn.BatchNorm1d`` in the
        MLP encoder raises on the final 1-sample batch. n=33 + bs=16 gives
        outer test=11, TrainVal=22 → batches of [16, 6] in folds where the
        pool size is 22, and [16, 16, 1] when the pool grows to 33 - 11 = 22.
        Use n=49, n_outer_folds=3 → TrainVal=33 → batches [16, 16, 1].
        """
        import tempfile
        from src.training.nested_cross_validation import NestedCrossValidator

        graphs, labels = _SyntheticNestedCV.make_dataset(n_subjects=49, seed=4)
        with tempfile.TemporaryDirectory() as td:
            model_cfg, trainer_cfg = _SyntheticNestedCV.make_configs(
                td, n_repetitions=1, n_outer_folds=3, inner_hpo_trials=0,
                batch_size=16,
            )
            ncv = NestedCrossValidator(cfg=trainer_cfg)
            result = ncv.run(
                model_factory=_SyntheticNestedCV.make_factory(model_cfg),
                dataset=graphs,
                labels=labels,
                base_model_cfg=model_cfg,
                logger=_SyntheticNestedCV.make_logger(),
                run_name="test-singleton",
            )
            assert len(result.fold_results) == 3


# ---------------------------------------------------------------------------
# random_sign_flip (LapPE sign augmentation)
# ---------------------------------------------------------------------------

class TestRandomSignFlip:
    """Unit tests for the LapPE eigenvector sign-flip augmentation."""

    def _xy(self):
        import torch
        x = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)  # no zeros
        batch_index = torch.tensor([0, 0, 1, 1])  # 2 graphs
        return x, batch_index

    def test_only_target_cols_change_and_abs_preserved(self) -> None:
        import torch
        from src.training.trainer import random_sign_flip
        x, bi = self._xy()
        g = torch.Generator().manual_seed(0)
        out = random_sign_flip(x, bi, 1, 3, generator=g)
        assert torch.equal(out[:, 0], x[:, 0])              # col 0 untouched
        assert torch.equal(out[:, 1:].abs(), x[:, 1:].abs())  # only signs change
        assert not torch.equal(out, x)                       # something flipped

    def test_signs_constant_within_graph(self) -> None:
        import torch
        from src.training.trainer import random_sign_flip
        x, bi = self._xy()
        g = torch.Generator().manual_seed(1)
        out = random_sign_flip(x, bi, 1, 3, generator=g)
        s0 = torch.sign(out[0, 1:] / x[0, 1:])
        s1 = torch.sign(out[1, 1:] / x[1, 1:])
        assert torch.equal(s0, s1)  # rows 0,1 are the same graph

    def test_deterministic_given_generator(self) -> None:
        import torch
        from src.training.trainer import random_sign_flip
        x, bi = self._xy()
        o1 = random_sign_flip(x, bi, 1, 3, generator=torch.Generator().manual_seed(7))
        o2 = random_sign_flip(x, bi, 1, 3, generator=torch.Generator().manual_seed(7))
        assert torch.equal(o1, o2)

    def test_does_not_mutate_input(self) -> None:
        import torch
        from src.training.trainer import random_sign_flip
        x, bi = self._xy()
        x_orig = x.clone()
        random_sign_flip(x, bi, 1, 3, generator=torch.Generator().manual_seed(0))
        assert torch.equal(x, x_orig)


# ---------------------------------------------------------------------------
# Trainer best-epoch direction (regression for the nested-CV refit_epochs=1 bug)
# ---------------------------------------------------------------------------

class TestTrainerBestEpochDirection:
    """``Trainer.fit`` must track the best epoch in the metric's natural direction.

    Regression for the bug where best-epoch selection was hardcoded to
    minimisation (``best_val_metric = +inf``; ``is_best = monitored < best``).
    With a higher-is-better monitor like ``val_r2`` that froze ``best_epoch`` at
    epoch 0 (the lowest R²), so every nested-CV refit ran for exactly one epoch
    (``refit_epochs = best_epoch + 1 = 1``) and the tested model predicted the
    mean → outer-test R² ≈ 0.
    """

    def _build(self, monitor: str, val_curve, patience=None):
        """Drive the real ``Trainer.fit`` with a scripted validation curve.

        ``evaluate`` is replaced so the per-epoch val metrics follow ``val_curve``
        deterministically, isolating the best-epoch decision logic from training
        stochasticity while still exercising the genuine ``fit`` loop.
        """
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.utils import scatter
        from src.configs.logging_config import LoggingConfig
        from src.configs.trainer_config import TrainerConfig
        from src.logging.wandb_logger import WandbLogger
        from src.training.trainer import Trainer

        class _TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 1)

            def forward(self, batch):
                pooled = scatter(batch.x, batch.batch, dim=0, reduce="mean")
                return self.lin(pooled)

            def auxiliary_loss(self):
                return None

        graphs = []
        for s in range(6):
            x = torch.arange(1, 4 * 5 + 1, dtype=torch.float32).reshape(5, 4) + s
            ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
            g = Data(x=x, edge_index=ei, edge_attr=torch.ones(4, 1),
                     y=torch.tensor([float(s)]), num_nodes=5)
            graphs.append(g)
        train_loader = DataLoader(graphs[:4], batch_size=2)
        val_loader = DataLoader(graphs[4:], batch_size=2)

        n = len(val_curve)
        cfg = TrainerConfig(
            epochs=n,
            device="cpu",
            scheduler="none",
            early_stopping_metric=monitor,
            # Never let patience trigger an early stop — we want all epochs to run
            # so the assertion is purely about the best-epoch direction.
            early_stopping_patience=patience if patience is not None else n + 1,
        )
        trainer = Trainer(cfg=cfg, logger=WandbLogger(LoggingConfig(enabled=False)))

        scripted = iter(val_curve)
        trainer.evaluate = lambda *a, **k: dict(next(scripted))

        result = trainer.fit(
            model=_TinyModel(),
            train_loader=train_loader,
            val_loader=val_loader,
            inverse_transform=lambda arr: arr,
            fold_idx=0,
        )
        return result

    def test_maximize_metric_tracks_rising_curve(self) -> None:
        """val_r2 rising 0→0.45 over 4 epochs → best_epoch must land on the last."""
        val_curve = [
            {"mae": 1.0, "rmse": 1.2, "r2": 0.00, "pearson_r": 0.10},
            {"mae": 0.7, "rmse": 0.9, "r2": 0.20, "pearson_r": 0.40},
            {"mae": 0.5, "rmse": 0.6, "r2": 0.35, "pearson_r": 0.60},
            {"mae": 0.4, "rmse": 0.5, "r2": 0.45, "pearson_r": 0.70},
        ]
        result = self._build("val_r2", val_curve)
        # The buggy minimisation-only code freezes best_epoch at 0.
        assert result.best_epoch == 3, (
            f"best_epoch={result.best_epoch}; expected the highest-R² epoch (3)"
        )
        assert result.best_epoch > 0
        assert result.best_val_metrics["r2"] == 0.45

    def test_minimize_metric_still_tracks_falling_then_rising(self) -> None:
        """val_mae minimum in the middle → best_epoch is that epoch (guards the
        existing minimise path is untouched by the fix)."""
        val_curve = [
            {"mae": 1.0, "rmse": 1.1, "r2": 0.05, "pearson_r": 0.2},
            {"mae": 0.4, "rmse": 0.5, "r2": 0.30, "pearson_r": 0.5},  # min MAE
            {"mae": 0.6, "rmse": 0.7, "r2": 0.25, "pearson_r": 0.4},
            {"mae": 0.9, "rmse": 1.0, "r2": 0.10, "pearson_r": 0.3},
        ]
        result = self._build("val_mae", val_curve)
        assert result.best_epoch == 1
        assert result.best_val_metrics["mae"] == 0.4


# ---------------------------------------------------------------------------
# Trainer sign-flip is train-only (safety-critical invariant)
# ---------------------------------------------------------------------------

class TestTrainerSignFlipTrainOnly:
    """The sign-flip must fire in training and never in eval."""

    def _make(self, sign_flip_cols):
        import numpy as np
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from src.configs.trainer_config import TrainerConfig
        from src.training.trainer import Trainer

        class _Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(1, 1)
                self.seen = []
            def forward(self, batch):
                self.seen.append(batch.x.detach().clone())
                from torch_geometric.utils import scatter
                per_node = batch.x.sum(dim=1, keepdim=True)
                pooled = scatter(per_node, batch.batch, dim=0, reduce="mean")
                return self.lin(pooled)
            def auxiliary_loss(self):
                return None

        graphs = []
        for s in range(4):
            x = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)  # 3 nodes,2 cols
            ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
            g = Data(x=x, edge_index=ei, edge_attr=torch.ones(2, 1),
                     y=torch.tensor([0.0]), num_nodes=3)
            graphs.append(g)
        loader = DataLoader(graphs, batch_size=2)
        cfg = TrainerConfig(epochs=1, device="cpu", sign_flip_cols=sign_flip_cols)
        from src.logging.wandb_logger import WandbLogger
        from src.configs.logging_config import LoggingConfig
        trainer = Trainer(cfg=cfg, logger=WandbLogger(LoggingConfig(enabled=False)))
        return trainer, _Recorder(), loader

    def test_train_flips_only_lap_cols_eval_does_not(self) -> None:
        import torch
        # Seed 0 is the first seed for which the sign-flip hook deterministically
        # produces at least one negative value in col 1 across a full epoch.
        # The trainer does NOT internally re-seed the global RNG, so we control
        # the global RNG via torch.manual_seed() immediately before each pass.
        _SEED = 0

        trainer, model, loader = self._make(sign_flip_cols=(1, 2))
        opt = torch.optim.SGD(model.parameters(), lr=0.0)

        # --- Training pass ---------------------------------------------------
        # Seed the global RNG so the flip is deterministic and we can assert
        # that it definitely fired (produced at least one negative value).
        torch.manual_seed(_SEED)
        trainer._train_one_epoch(model, loader, opt)
        train_x = torch.cat(model.seen, dim=0)
        orig_col0 = torch.tensor([1.0, 3.0, 5.0] * 4)   # col 0 across 12 nodes
        orig_col1_abs = torch.tensor([2.0, 4.0, 6.0] * 4)

        assert torch.equal(train_x[:, 0], orig_col0)           # col 0 untouched
        assert torch.equal(train_x[:, 1].abs(), orig_col1_abs) # only sign of col1 changes
        # Positive assertion: with seed 0 the hook MUST have fired and flipped
        # at least one value to negative. Without the hook col1 stays all-positive
        # and this assertion fails, making the test a genuine guard.
        assert (train_x[:, 1] < 0).any(), (
            "sign-flip hook did not fire during training — col1 is all-positive"
        )

        # --- Eval pass -------------------------------------------------------
        # Re-seed identically so that IF predict() wrongly applied the same flip
        # it would produce the identical deterministic negatives we saw above.
        # We then assert col1 exactly equals the positive originals, proving
        # the hook was correctly absent from the eval path.
        model.seen.clear()
        torch.manual_seed(_SEED)
        trainer.predict(model, loader, inverse_transform=lambda a: a)
        eval_x = torch.cat(model.seen, dim=0)
        assert torch.equal(eval_x[:, 1], orig_col1_abs)  # exact, no sign change
