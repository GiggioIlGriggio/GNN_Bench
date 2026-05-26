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
