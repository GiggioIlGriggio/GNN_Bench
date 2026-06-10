"""Epoch-free nested cross-validation for classical-ML baselines.

Reuses the GNN pipeline's leaf components — shared fold splits
(src.training.splits), FoldBarrier label/GLM normalization, the
search-space DSL + TrialOverrides, Optuna, compute_metrics, and the
NestedCVResult/FoldResult schema — so sklearn baselines are directly
comparable to the GNN runs (same folds, same label handling, same output).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from src.datasets.label_builder import LabelBuilder
from src.models.flatten import build_feature_matrix
from src.models.sklearn_baselines import build_estimator
from src.training.fold_barrier import FoldBarrier
from src.training.metrics import compute_metrics
from src.training.nested_cross_validation import (
    FoldResult, NestedCVResult, _aggregate,
)
from src.training.search_space import (
    TrialOverrides, load_sweeper_params, parse_search_space,
)
from src.training.splits import inner_split, outer_folds, stratify_bins

log = logging.getLogger(__name__)


class SklearnNestedCrossValidator:
    """Nested CV for sklearn estimators. Mirrors the GNN protocol without epochs."""

    def __init__(
        self,
        cfg: TrainerConfig,
        search_space_path: Optional[str | Path] = None,
    ) -> None:
        self.cfg = cfg
        self.search_space_path = (
            Path(search_space_path) if search_space_path is not None else None
        )

    def run(
        self,
        *,
        estimator_name: str,
        model_cfg: ModelConfig,
        dataset: List[Any],
        labels: np.ndarray,
        logger: Any,
        run_name: str,
        num_nodes: int,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        label_builder: Optional[LabelBuilder] = None,
        label_components: Optional[pd.DataFrame] = None,
        feature_config: Optional[Dict[str, Any]] = None,
    ) -> NestedCVResult:
        if (label_builder is None) != (label_components is None):
            raise ValueError(
                "'label_builder' and 'label_components' must both be provided "
                "or both be None."
            )

        n_outer = self.cfg.effective_n_outer_folds
        if n_outer < 2:
            raise ValueError(f"effective_n_outer_folds must be >= 2, got {n_outer}")

        specs = None
        if self.cfg.inner_hpo_trials > 0:
            if self.search_space_path is None:
                raise ValueError("inner_hpo_trials > 0 but no search_space_path given")
            specs = parse_search_space(load_sweeper_params(self.search_space_path))

        input_mode = model_cfg.mlp_input
        weighted = model_cfg.mlp_adjacency_type == "weighted"
        outer_seeds = self.cfg.resolved_outer_seeds()
        bins = stratify_bins(labels, self.cfg.stratify_bins)
        all_folds = outer_folds(
            n=len(dataset), bins=bins, seeds=outer_seeds, n_outer=n_outer,
        )

        ckpt_root = Path(self.cfg.checkpoint_dir) / run_name
        fold_results: List[FoldResult] = []

        for flat_idx, (train_val_idx, test_idx) in enumerate(all_folds):
            rep = flat_idx // n_outer
            fold = flat_idx % n_outer
            outer_seed = outer_seeds[rep]
            inner_seed = outer_seed * 1000 + fold
            fr = self._run_fold(
                estimator_name=estimator_name, model_cfg=model_cfg,
                dataset=dataset, labels=labels, bins=bins,
                train_val_idx=train_val_idx, test_idx=test_idx,
                rep=rep, fold=fold, inner_seed=inner_seed,
                input_mode=input_mode, weighted=weighted, num_nodes=num_nodes,
                specs=specs, glm_col_range=glm_col_range, glm_normalize=glm_normalize,
                label_builder=label_builder, label_components=label_components,
                logger=logger,
            )
            fold_results.append(fr)
            log.info("rep=%d fold=%d outer-test: %s", rep, fold, fr.outer_test_metrics)

        mean_metrics, std_metrics = _aggregate(fold_results)
        result = NestedCVResult(
            run_name=run_name, model_name=estimator_name,
            n_repetitions=self.cfg.n_repetitions, n_outer_folds=n_outer,
            inner_hpo_trials=self.cfg.inner_hpo_trials, hpo_metric=self.cfg.hpo_metric,
            outer_seeds=outer_seeds, fold_results=fold_results,
            mean_metrics=mean_metrics, std_metrics=std_metrics,
        )
        result.save(ckpt_root / "nested_cv_result.json")
        try:
            logger.log_nested_final_summary(mean_metrics, std_metrics)
        except Exception as exc:  # noqa: BLE001
            log.warning("log_nested_final_summary failed (continuing): %s", exc)
        return result

    # ------------------------------------------------------------------

    def _barrier_and_labels(
        self, *, dataset, labels, train_idx, eval_idx,
        glm_col_range, glm_normalize, label_builder, label_components,
    ):
        """Fit a FoldBarrier on train_idx; return (barrier, y_train, y_eval)."""
        barrier = FoldBarrier(
            label_norm_strategy=self.cfg.label_norm_strategy,
            glm_col_range=glm_col_range, glm_normalize=glm_normalize,
            label_builder=label_builder,
        )
        train_graphs = [dataset[i] for i in train_idx]
        if label_builder is not None and label_components is not None:
            barrier.fit(train_graphs, label_components.iloc[train_idx])
            y_train = barrier.transform_labels(label_components.iloc[train_idx])
            y_eval = barrier.transform_labels(label_components.iloc[eval_idx])
        else:
            barrier.fit(train_graphs, labels[train_idx])
            y_train = barrier.transform_labels(labels[train_idx])
            y_eval = barrier.transform_labels(labels[eval_idx])
        return barrier, y_train, y_eval

    def _matrix(self, barrier, dataset, idx, input_mode, weighted, num_nodes):
        graphs = barrier.transform_graphs([dataset[i] for i in idx])
        return build_feature_matrix(
            graphs, input_mode=input_mode, num_nodes=num_nodes, weighted=weighted,
        )

    def _score(self, name, params, X_tr, y_tr, X_ev, y_ev, barrier, seed):
        """Fit on train, predict eval, return metrics on the original scale."""
        pipe = build_estimator(name, params, seed=seed)
        pipe.fit(X_tr, y_tr)
        y_pred = barrier.inverse_transform_labels(pipe.predict(X_ev))
        y_true = barrier.inverse_transform_labels(y_ev)
        return y_true, y_pred, pipe

    def _run_fold(
        self, *, estimator_name, model_cfg, dataset, labels, bins,
        train_val_idx, test_idx, rep, fold, inner_seed, input_mode, weighted,
        num_nodes, specs, glm_col_range, glm_normalize,
        label_builder, label_components, logger,
    ) -> FoldResult:
        # ---- Inner HPO (or base params) -------------------------------
        n_val = 0
        if specs is not None and self.cfg.inner_hpo_trials > 0:
            tr_idx, va_idx = inner_split(train_val_idx, bins, inner_seed)
            n_val = len(va_idx)
            barrier_in, y_tr, y_va = self._barrier_and_labels(
                dataset=dataset, labels=labels, train_idx=tr_idx, eval_idx=va_idx,
                glm_col_range=glm_col_range, glm_normalize=glm_normalize,
                label_builder=label_builder, label_components=label_components,
            )
            X_tr = self._matrix(barrier_in, dataset, tr_idx, input_mode, weighted, num_nodes)
            X_va = self._matrix(barrier_in, dataset, va_idx, input_mode, weighted, num_nodes)
            best_hparams, best_trial = self._inner_hpo(
                estimator_name, model_cfg, specs, X_tr, y_tr, X_va, y_va,
                barrier_in, inner_seed,
            )
        else:
            best_hparams, best_trial = {}, 0

        # ---- Refit on outer-trainval, evaluate on outer-test ----------
        trial_model_cfg, _ = TrialOverrides(values=best_hparams).apply(
            model_cfg=model_cfg, trainer_cfg=self.cfg,
        )
        params = dict(trial_model_cfg.model_params)
        barrier_out, y_trv, y_te = self._barrier_and_labels(
            dataset=dataset, labels=labels, train_idx=train_val_idx, eval_idx=test_idx,
            glm_col_range=glm_col_range, glm_normalize=glm_normalize,
            label_builder=label_builder, label_components=label_components,
        )
        X_trv = self._matrix(barrier_out, dataset, train_val_idx, input_mode, weighted, num_nodes)
        X_te = self._matrix(barrier_out, dataset, test_idx, input_mode, weighted, num_nodes)
        y_true, y_pred, _ = self._score(
            estimator_name, params, X_trv, y_trv, X_te, y_te, barrier_out, inner_seed,
        )
        outer_metrics = compute_metrics(y_true, y_pred)

        try:
            logger.log_nested_outer_test(rep, fold, outer_metrics)
            logger.log_nested_best_hparams(rep, fold, best_hparams, best_trial, 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb log rep=%d fold=%d failed: %s", rep, fold, exc)

        return FoldResult(
            rep=rep, fold=fold, outer_test_metrics=outer_metrics,
            best_hparams=best_hparams, best_trial=best_trial, refit_epochs=0,
            n_train=len(train_val_idx), n_val=n_val, n_test=len(test_idx),
            y_true=y_true.tolist(), y_pred=y_pred.tolist(),
        )

    def _inner_hpo(
        self, name, model_cfg, specs, X_tr, y_tr, X_va, y_va, barrier, inner_seed,
    ) -> Tuple[Dict[str, Any], int]:
        import optuna

        maximize = self.cfg.hpo_metric != "val_mae"
        metric_key = self.cfg.hpo_metric.replace("val_", "")
        study = optuna.create_study(
            direction="maximize" if maximize else "minimize",
            sampler=optuna.samplers.TPESampler(seed=inner_seed),
        )

        def objective(trial: "optuna.Trial") -> float:
            overrides = TrialOverrides(values={s.name: s.suggest(trial) for s in specs})
            trial_cfg, _ = overrides.apply(model_cfg=model_cfg, trainer_cfg=self.cfg)
            params = dict(trial_cfg.model_params)
            y_true, y_pred, _ = self._score(
                name, params, X_tr, y_tr, X_va, y_va, barrier, inner_seed,
            )
            return compute_metrics(y_true, y_pred)[metric_key]

        study.optimize(objective, n_trials=self.cfg.inner_hpo_trials)
        # Reconstruct chosen overrides from the best trial's sampled params.
        best = study.best_trial
        best_hparams = {s.name: best.params[s.name] for s in specs}
        return best_hparams, int(best.number)
