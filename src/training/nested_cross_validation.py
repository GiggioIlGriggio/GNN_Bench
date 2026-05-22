"""Repeated stratified nested cross-validation (ADR-0008).

Single entry point for benchmark evaluation. Two knobs subsume both the
legacy fast CV and the paper-grade protocol:

============================  =================  =====================
Knob                          Fast preset        Paper preset
============================  =================  =====================
``n_repetitions``             1                  10
``n_outer_folds``             5                  5
``inner_hpo_trials``          0                  20
============================  =================  =====================

When ``inner_hpo_trials == 0`` the validator behaves like the legacy
CrossValidator — train on inner-train, early-stop on inner-val, evaluate
the best-val checkpoint on outer-test. When ``inner_hpo_trials > 0`` an
in-process Optuna study picks HPs for that fold, then the model is
**refit on outer-TrainVal** for the winning trial's ``best_epoch + 1``
epochs and that refit checkpoint is the only one evaluated on outer-test.

All per-fold leakage protections from the legacy CrossValidator are
preserved behind a single ``FoldBarrier`` (ADR-0009): one barrier is
fit on each outer-Train pool and applied uniformly to inner-train,
inner-val, and outer-test — never the other way round.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.data
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from src.datasets.label_builder import LabelBuilder
from src.training.checkpoint_manager import CheckpointManager
from src.training.fold_barrier import FoldBarrier
from src.training.metrics import MetricDict, compute_metrics
from src.training.search_space import (
    SearchSpec,
    TrialOverrides,
    load_sweeper_params,
    parse_search_space,
)
from src.training.trainer import Trainer

if TYPE_CHECKING:
    from src.logging.wandb_logger import WandbLogger
    from src.models.base_model import BrainGNN

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Outcome of one outer fold inside one repetition."""

    rep: int
    fold: int
    outer_test_metrics: MetricDict
    best_hparams: Dict[str, Any]
    best_trial: int
    refit_epochs: int
    n_train: int
    n_val: int
    n_test: int
    y_true: List[float] = field(default_factory=list)
    y_pred: List[float] = field(default_factory=list)


@dataclass
class NestedCVResult:
    """Aggregated nested-CV result for one (model, dataset, label) run.

    Persists to JSON via :meth:`save` and is the unit of input consumed by
    ``scripts/compare_models.py`` for cross-model statistical tests.
    """

    run_name: str
    model_name: str
    n_repetitions: int
    n_outer_folds: int
    inner_hpo_trials: int
    hpo_metric: str
    outer_seeds: List[int]
    fold_results: List[FoldResult]
    mean_metrics: MetricDict
    std_metrics: MetricDict

    def per_fold_scores(self, metric: str) -> List[float]:
        """Return the per-outer-fold scores for one metric, in run order."""
        return [fr.outer_test_metrics[metric] for fr in self.fold_results]

    def save(self, path: str | Path) -> None:
        """Persist to a JSON file readable by :meth:`load`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NestedCVResult":
        """Load a previously saved :meth:`save` payload."""
        with open(path) as f:
            data = json.load(f)
        data["fold_results"] = [FoldResult(**fr) for fr in data["fold_results"]]
        return cls(**data)


# ---------------------------------------------------------------------------
# Logger adapter — redirects Trainer.fit's per-epoch logs into nested keys.
# ---------------------------------------------------------------------------


class _NestedTrialLogger:
    """Duck-typed WandbLogger that retargets ``log_fold_metrics`` to nested keys.

    The Trainer calls ``logger.log_fold_metrics(fold_idx, metrics, split, epoch)``
    twice per epoch (train + val). When that logger is this adapter, those
    calls become ``log_nested_trial_metrics(rep, fold, trial, split, metrics,
    epoch)`` against the wrapped real logger — and ``fold_idx`` is discarded
    (the trainer doesn't know it's running inside a trial).

    All other attributes pass through unchanged so Trainer can keep using
    ``logger.cfg`` and the rest of the API.
    """

    def __init__(
        self, base: "WandbLogger", rep: int, fold: int, trial: int,
    ) -> None:
        self._base = base
        self._rep = rep
        self._fold = fold
        self._trial = trial

    def log_fold_metrics(
        self,
        fold_idx: int,  # noqa: ARG002 — discarded; (rep, fold, trial) come from adapter
        metrics: MetricDict,
        split: str,
        epoch: Optional[int] = None,
    ) -> None:
        self._base.log_nested_trial_metrics(
            self._rep, self._fold, self._trial, split, metrics, epoch=epoch,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


# ---------------------------------------------------------------------------
# Nested cross-validator
# ---------------------------------------------------------------------------


class NestedCrossValidator:
    """Outer repeated stratified K-fold + inner stratified 4-way + Optuna HPO.

    Construct once per (dataset, model) run. Call :meth:`run` to evaluate
    the model under the protocol configured in the supplied ``TrainerConfig``.

    Parameters
    ----------
    cfg : TrainerConfig
        Must carry the nested-CV knobs (``n_repetitions``, ``n_outer_folds``,
        ``inner_hpo_trials``, ``hpo_metric``, ``outer_seeds``) — see
        :class:`~src.configs.trainer_config.TrainerConfig`.
    search_space_path : Optional[Path]
        Path to the sweeper YAML to drive inner HPO. Required when
        ``cfg.inner_hpo_trials > 0``; ignored otherwise.
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        search_space_path: Optional[str | Path] = None,
    ) -> None:
        self.cfg = cfg
        self.search_space_path = (
            Path(search_space_path) if search_space_path is not None else None
        )
        self._search_specs: Optional[List[SearchSpec]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        model_factory: Callable[[ModelConfig], "BrainGNN"],
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        base_model_cfg: ModelConfig,
        logger: "WandbLogger",
        run_name: str = "nested_cv",
        label_builder: Optional[LabelBuilder] = None,
        label_components: Optional[pd.DataFrame] = None,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        feature_config: Optional[Dict[str, Any]] = None,
    ) -> NestedCVResult:
        """Run the full nested cross-validation protocol.

        Parameters
        ----------
        model_factory : Callable[[ModelConfig], BrainGNN]
            Builds a fresh, untrained model from a (possibly HP-overridden)
            ModelConfig.
        dataset : list of PyG Data
            Full dataset.
        labels : np.ndarray
            Per-subject scalar labels (used for stratification binning and
            as default fold labels when ``label_builder`` is None).
        base_model_cfg : ModelConfig
            Baseline model config; per-trial HP overrides are applied via
            ``model_copy(update=...)``.
        logger : WandbLogger
        run_name : str
            Top-level identifier persisted into ``NestedCVResult.run_name``.
        label_builder, label_components, glm_col_range, glm_normalize
            Per-fold leakage protections inherited from the legacy CV.
        feature_config : dict, optional
            Serialised FeatureConfig stored alongside each refit checkpoint
            for self-contained reload.

        Returns
        -------
        NestedCVResult
        """
        if (label_builder is None) != (label_components is None):
            raise ValueError(
                "'label_builder' and 'label_components' must both be provided "
                "or both be None."
            )

        n_outer = self.cfg.effective_n_outer_folds
        if n_outer < 2:
            raise ValueError(
                f"NestedCrossValidator requires effective_n_outer_folds >= 2, "
                f"got {n_outer} (n_outer_folds={self.cfg.n_outer_folds}, "
                f"n_folds={self.cfg.n_folds})"
            )
        if self.cfg.inner_hpo_trials > 0:
            if self.search_space_path is None:
                raise ValueError(
                    "inner_hpo_trials > 0 but no search_space_path was given"
                )
            params = load_sweeper_params(self.search_space_path)
            self._search_specs = parse_search_space(params)
            log.info(
                "Loaded %d search-space specs from %s",
                len(self._search_specs), self.search_space_path,
            )

        device = self._resolve_device()
        log.info("Training device: %s", device)
        outer_seeds = self.cfg.resolved_outer_seeds()
        log.info(
            "Nested CV — reps=%d  outer_folds=%d  inner_trials=%d  hpo_metric=%s",
            self.cfg.n_repetitions, n_outer, self.cfg.inner_hpo_trials,
            self.cfg.hpo_metric,
        )

        ckpt_root = Path(self.cfg.checkpoint_dir)
        fold_results: List[FoldResult] = []

        # Bin labels once — same bins across all reps (deterministic given the
        # label vector).
        bins = self._stratify_bins(labels)

        for rep, outer_seed in enumerate(outer_seeds):
            log.info("=== Repetition %d / %d (seed=%d) ===",
                     rep + 1, self.cfg.n_repetitions, outer_seed)
            skf = StratifiedKFold(
                n_splits=n_outer, shuffle=True, random_state=outer_seed,
            )
            for fold_idx, (train_val_idx, test_idx) in enumerate(
                skf.split(np.arange(len(dataset)), bins)
            ):
                fr = self._run_outer_fold(
                    rep=rep,
                    fold=fold_idx,
                    outer_seed=outer_seed,
                    train_val_idx=train_val_idx.tolist(),
                    test_idx=test_idx.tolist(),
                    dataset=dataset,
                    labels=labels,
                    bins=bins,
                    base_model_cfg=base_model_cfg,
                    model_factory=model_factory,
                    logger=logger,
                    device=device,
                    label_builder=label_builder,
                    label_components=label_components,
                    glm_col_range=glm_col_range,
                    glm_normalize=glm_normalize,
                    feature_config=feature_config,
                    ckpt_root=ckpt_root,
                )
                fold_results.append(fr)
                log.info(
                    "rep=%d fold=%d outer-test: %s",
                    rep, fold_idx, fr.outer_test_metrics,
                )

        mean_metrics, std_metrics = _aggregate(fold_results)
        result = NestedCVResult(
            run_name=run_name,
            model_name=base_model_cfg.name,
            n_repetitions=self.cfg.n_repetitions,
            n_outer_folds=n_outer,
            inner_hpo_trials=self.cfg.inner_hpo_trials,
            hpo_metric=self.cfg.hpo_metric,
            outer_seeds=outer_seeds,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
        )
        result.save(ckpt_root / "nested_cv_result.json")
        try:
            logger.log_nested_final_summary(mean_metrics, std_metrics)
        except Exception as exc:  # noqa: BLE001 — logger errors must not kill the run
            log.warning("log_nested_final_summary failed (continuing): %s", exc)
        return result

    # ------------------------------------------------------------------
    # Outer-fold orchestration
    # ------------------------------------------------------------------

    def _run_outer_fold(
        self,
        *,
        rep: int,
        fold: int,
        outer_seed: int,
        train_val_idx: List[int],
        test_idx: List[int],
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        bins: np.ndarray,
        base_model_cfg: ModelConfig,
        model_factory: Callable[[ModelConfig], "BrainGNN"],
        logger: "WandbLogger",
        device: torch.device,
        label_builder: Optional[LabelBuilder],
        label_components: Optional[pd.DataFrame],
        glm_col_range: Optional[Tuple[int, int]],
        glm_normalize: bool,
        feature_config: Optional[Dict[str, Any]],
        ckpt_root: Path,
    ) -> FoldResult:
        inner_seed = outer_seed * 1000 + fold
        train_idx, val_idx = self._inner_split(
            train_val_idx, bins, inner_seed,
        )
        log.info(
            "rep=%d fold=%d — inner train=%d val=%d  outer test=%d",
            rep, fold, len(train_idx), len(val_idx), len(test_idx),
        )

        # ------------------------------------------------------------------
        # Build inner data (with leakage protections fit on inner-train only).
        # ------------------------------------------------------------------
        inner_loaders, inner_barrier = self._make_split_loaders(
            dataset=dataset,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            label_builder=label_builder,
            label_components=label_components,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
        )

        # ------------------------------------------------------------------
        # Inner phase: HPO with Optuna, or single training when n_trials == 0.
        # ------------------------------------------------------------------
        fold_ckpt_dir = ckpt_root / f"rep_{rep}" / f"fold_{fold}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.inner_hpo_trials > 0:
            best_hparams, best_trial_idx, best_epoch, trials_rows = self._run_inner_hpo(
                rep=rep, fold=fold, inner_seed=inner_seed,
                base_model_cfg=base_model_cfg,
                model_factory=model_factory,
                inner_loaders=inner_loaders,
                inner_barrier=inner_barrier,
                logger=logger,
                device=device,
            )
            _write_trials_csv(fold_ckpt_dir / "trials.csv", trials_rows)
        else:
            best_hparams = {}
            best_trial_idx = 0
            # No HPO — train once with base HPs to determine best_epoch (for
            # logging / refit_epochs persistence) and reuse that model.
            best_epoch = self._train_inner_trial(
                rep=rep, fold=fold, trial=0,
                trial_model_cfg=base_model_cfg,
                trial_trainer_cfg=self._trainer_cfg_for_inner(),
                model_factory=model_factory,
                inner_loaders=inner_loaders,
                inner_barrier=inner_barrier,
                logger=logger,
                device=device,
            )[0]

        # ------------------------------------------------------------------
        # Refit-on-TrainVal (when HPO ran) or reuse the best inner checkpoint
        # (fast preset). Either way, evaluate on outer test.
        # ------------------------------------------------------------------
        outer_loaders, outer_barrier = self._make_split_loaders(
            dataset=dataset,
            labels=labels,
            train_idx=train_val_idx,  # full outer-TrainVal pool
            val_idx=[],
            test_idx=test_idx,
            label_builder=label_builder,
            label_components=label_components,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
        )

        trial_model_cfg, trial_trainer_cfg = TrialOverrides(values=best_hparams).apply(
            model_cfg=base_model_cfg, trainer_cfg=self._trainer_cfg_for_inner(),
        )

        if self.cfg.inner_hpo_trials > 0:
            refit_epochs = best_epoch + 1
            model = self._refit_on_trainval(
                trial_model_cfg=trial_model_cfg,
                trial_trainer_cfg=trial_trainer_cfg,
                model_factory=model_factory,
                train_loader=outer_loaders["train"],
                refit_epochs=refit_epochs,
                device=device,
            )
        else:
            # Fast preset: retrain on outer TrainVal for the same best_epoch + 1
            # epochs so the test eval uses a model that has seen no part of the
            # outer-test fold. (The legacy CV used the inner-val best-checkpoint
            # weights directly — this is the modest improvement we keep even
            # without HPO.)
            refit_epochs = best_epoch + 1
            model = self._refit_on_trainval(
                trial_model_cfg=trial_model_cfg,
                trial_trainer_cfg=trial_trainer_cfg,
                model_factory=model_factory,
                train_loader=outer_loaders["train"],
                refit_epochs=refit_epochs,
                device=device,
            )

        # Outer-test evaluation
        eval_trainer = Trainer(cfg=trial_trainer_cfg, logger=logger)
        y_true, y_pred = eval_trainer.predict(
            model, outer_loaders["test"], outer_barrier.inverse_transform_labels,
        )
        outer_metrics = compute_metrics(y_true, y_pred)

        # ------------------------------------------------------------------
        # Persist fold artifacts
        # ------------------------------------------------------------------
        np.savez(
            fold_ckpt_dir / "test_predictions.npz",
            y_true=y_true, y_pred=y_pred,
        )
        with open(fold_ckpt_dir / "best_hparams.json", "w") as f:
            json.dump(
                {
                    "best_hparams": best_hparams,
                    "best_trial": best_trial_idx,
                    "refit_epochs": refit_epochs,
                },
                f, indent=2,
            )

        ckpt_mgr = CheckpointManager(fold_ckpt_dir.parent)  # rep_<R> directory
        ckpt_mgr.save_fold_checkpoint(
            fold_idx=fold,
            best_model_state_dict=model.state_dict(),
            last_model_state_dict=model.state_dict(),
            barrier=outer_barrier,
            best_metrics=outer_metrics,
            best_epoch=refit_epochs - 1,
            last_metrics=outer_metrics,
            last_epoch=refit_epochs - 1,
            model_config=trial_model_cfg.model_dump(),
            feature_config=feature_config,
        )

        try:
            logger.log_nested_outer_test(rep, fold, outer_metrics)
            logger.log_nested_best_hparams(
                rep, fold, best_hparams, best_trial_idx, refit_epochs,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "wandb log for rep=%d fold=%d failed (continuing): %s",
                rep, fold, exc,
            )

        return FoldResult(
            rep=rep,
            fold=fold,
            outer_test_metrics=outer_metrics,
            best_hparams=best_hparams,
            best_trial=best_trial_idx,
            refit_epochs=refit_epochs,
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_test=len(test_idx),
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
        )

    # ------------------------------------------------------------------
    # Inner HPO
    # ------------------------------------------------------------------

    def _run_inner_hpo(
        self,
        *,
        rep: int,
        fold: int,
        inner_seed: int,
        base_model_cfg: ModelConfig,
        model_factory: Callable[[ModelConfig], "BrainGNN"],
        inner_loaders: Dict[str, DataLoader],
        inner_barrier: FoldBarrier,
        logger: "WandbLogger",
        device: torch.device,
    ) -> Tuple[Dict[str, Any], int, int, List[Dict[str, Any]]]:
        """Run an in-process Optuna study over the inner train / val split.

        Returns
        -------
        Tuple[dict, int, int, list]
            ``(best_hparams, best_trial_idx, best_epoch, trials_rows)``
            where ``trials_rows`` is the per-trial CSV body.
        """
        import optuna

        direction = "minimize" if self.cfg.hpo_metric == "val_mae" else "maximize"
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=inner_seed),
        )

        # Per-trial bookkeeping (Optuna's user_attrs are not iteration-stable
        # across the same trial.number when failures happen; we keep our own).
        trial_records: Dict[int, Dict[str, Any]] = {}

        def objective(trial: "optuna.Trial") -> float:
            overrides = TrialOverrides(
                values={s.name: s.suggest(trial) for s in self._search_specs}
            )
            trial_model_cfg, trial_trainer_cfg = overrides.apply(
                model_cfg=base_model_cfg,
                trainer_cfg=self._trainer_cfg_for_inner(),
            )

            best_epoch, best_val_metric = self._train_inner_trial(
                rep=rep, fold=fold, trial=trial.number,
                trial_model_cfg=trial_model_cfg,
                trial_trainer_cfg=trial_trainer_cfg,
                model_factory=model_factory,
                inner_loaders=inner_loaders,
                inner_barrier=inner_barrier,
                logger=logger,
                device=device,
            )
            trial_records[trial.number] = {
                "params": dict(overrides.values),
                "best_epoch": best_epoch,
                "best_val_metric": best_val_metric,
                "objective": best_val_metric,
            }
            return best_val_metric

        study.optimize(objective, n_trials=self.cfg.inner_hpo_trials)

        best_trial = study.best_trial
        record = trial_records[best_trial.number]
        best_hparams = record["params"]
        best_epoch = record["best_epoch"]

        # Build CSV rows from all completed trials in trial-number order.
        trials_rows: List[Dict[str, Any]] = []
        for tn in sorted(trial_records):
            r = trial_records[tn]
            row = {
                "trial": tn,
                "objective": r["objective"],
                "best_epoch": r["best_epoch"],
            }
            for k, v in r["params"].items():
                row[k] = v
            trials_rows.append(row)

        return best_hparams, int(best_trial.number), int(best_epoch), trials_rows

    def _train_inner_trial(
        self,
        *,
        rep: int,
        fold: int,
        trial: int,
        trial_model_cfg: ModelConfig,
        trial_trainer_cfg: TrainerConfig,
        model_factory: Callable[[ModelConfig], "BrainGNN"],
        inner_loaders: Dict[str, DataLoader],
        inner_barrier: FoldBarrier,
        logger: "WandbLogger",
        device: torch.device,
    ) -> Tuple[int, float]:
        """Train one inner-loop model and return ``(best_epoch, best_val_metric)``.

        ``best_val_metric`` is the value of ``cfg.hpo_metric`` at the best
        epoch — already in the optimiser's preferred sign (minimize for MAE,
        maximize for R²; Optuna handles the direction).
        """
        adapter = _NestedTrialLogger(logger, rep, fold, trial)
        trainer = Trainer(cfg=trial_trainer_cfg, logger=adapter)
        model = model_factory(trial_model_cfg).to(device)
        result = trainer.fit(
            model,
            inner_loaders["train"],
            inner_loaders["val"],
            inner_barrier.inverse_transform_labels,
            fold_idx=fold,
            on_epoch_end_callback=None,
        )
        metric_key = self.cfg.hpo_metric.replace("val_", "")
        return result.best_epoch, float(result.best_val_metrics[metric_key])

    def _refit_on_trainval(
        self,
        *,
        trial_model_cfg: ModelConfig,
        trial_trainer_cfg: TrainerConfig,
        model_factory: Callable[[ModelConfig], "BrainGNN"],
        train_loader: DataLoader,
        refit_epochs: int,
        device: torch.device,
    ) -> "BrainGNN":
        """Refit on outer TrainVal for a fixed epoch count — no val, no early stop."""
        model = model_factory(trial_model_cfg).to(device)
        # Reuse Trainer's optimizer/scheduler builders without entering fit().
        helper = Trainer(cfg=trial_trainer_cfg, logger=_NullLogger())
        optimizer = helper._build_optimizer(model)
        scheduler = helper._build_scheduler(optimizer)
        for _ in range(max(1, refit_epochs)):
            helper._train_one_epoch(model, train_loader, optimizer)
            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()
        return model

    # ------------------------------------------------------------------
    # Splits + loaders
    # ------------------------------------------------------------------

    def _stratify_bins(self, labels: np.ndarray) -> np.ndarray:
        return pd.qcut(
            labels, q=self.cfg.stratify_bins, labels=False, duplicates="drop",
        )

    def _inner_split(
        self, train_val_idx: List[int], bins: np.ndarray, inner_seed: int,
    ) -> Tuple[List[int], List[int]]:
        """Stratified 4-way split → first fold's split is 3:1 train:val.

        Bins are taken from the global label binning (consistent with
        ADR-0003) and indexed into via ``train_val_idx``.
        """
        idx_arr = np.asarray(train_val_idx)
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=inner_seed)
        first_train, first_val = next(skf.split(idx_arr, bins[idx_arr]))
        return idx_arr[first_train].tolist(), idx_arr[first_val].tolist()

    def _make_split_loaders(
        self,
        *,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
        label_builder: Optional[LabelBuilder],
        label_components: Optional[pd.DataFrame],
        glm_col_range: Optional[Tuple[int, int]],
        glm_normalize: bool,
    ) -> Tuple[Dict[str, DataLoader], FoldBarrier]:
        """Build train/val/test DataLoaders behind a FoldBarrier (ADR-0009).

        Set ``val_idx=[]`` for the outer refit phase — the returned dict
        omits the ``"val"`` key.
        """
        barrier = FoldBarrier(
            label_norm_strategy=self.cfg.label_norm_strategy,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
            label_builder=label_builder,
        )

        train_graphs_raw = [dataset[i] for i in train_idx]
        if label_builder is not None and label_components is not None:
            barrier.fit(train_graphs_raw, label_components.iloc[train_idx])
            y_train_norm = barrier.transform_labels(label_components.iloc[train_idx])
            y_test_norm = barrier.transform_labels(label_components.iloc[test_idx])
            y_val_norm = (
                barrier.transform_labels(label_components.iloc[val_idx])
                if val_idx else np.zeros(0)
            )
        else:
            barrier.fit(train_graphs_raw, labels[train_idx])
            y_train_norm = barrier.transform_labels(labels[train_idx])
            y_test_norm = barrier.transform_labels(labels[test_idx])
            y_val_norm = (
                barrier.transform_labels(labels[val_idx])
                if val_idx else np.zeros(0)
            )

        train_graphs = barrier.transform_graphs(train_graphs_raw)
        test_graphs = barrier.transform_graphs([dataset[i] for i in test_idx])
        val_graphs = (
            barrier.transform_graphs([dataset[i] for i in val_idx])
            if val_idx else []
        )

        bs = self.cfg.batch_size
        loaders: Dict[str, DataLoader] = {
            "train": _make_loader_from_graphs(
                train_graphs, y_train_norm, shuffle=True,
                batch_size=bs, drop_last=len(train_idx) > bs,
            ),
            "test": _make_loader_from_graphs(
                test_graphs, y_test_norm, shuffle=False,
                batch_size=bs, drop_last=False,
            ),
        }
        if val_idx:
            loaders["val"] = _make_loader_from_graphs(
                val_graphs, y_val_norm, shuffle=False,
                batch_size=bs, drop_last=False,
            )

        return loaders, barrier

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _trainer_cfg_for_inner(self) -> TrainerConfig:
        """Return a TrainerConfig with early stopping pinned to the HPO metric.

        Inner training must use the same signal Optuna optimises so the
        ``best_epoch`` Optuna picks matches the val curve we recorded.
        """
        return self.cfg.model_copy(
            update={"early_stopping_metric": self.cfg.hpo_metric}
        )

    def _resolve_device(self) -> torch.device:
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loader_from_graphs(
    graphs: List[torch_geometric.data.Data],
    y_values: np.ndarray,
    *,
    shuffle: bool,
    batch_size: int,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap pre-transformed graphs in a PyG DataLoader, attaching labels.

    The graphs must have been produced by ``FoldBarrier.transform_graphs``
    (or otherwise be safe to mutate in ``.y``).
    """
    for i, g in enumerate(graphs):
        g.y = torch.tensor([y_values[i]], dtype=torch.float32)
    return DataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
    )


def _aggregate(
    fold_results: List[FoldResult],
) -> Tuple[MetricDict, MetricDict]:
    """Mean ± std of each metric across all outer folds."""
    if not fold_results:
        return {}, {}
    keys = list(fold_results[0].outer_test_metrics.keys())
    mean: MetricDict = {}
    std: MetricDict = {}
    for k in keys:
        vals = np.array([fr.outer_test_metrics[k] for fr in fold_results])
        mean[k] = float(vals.mean())
        std[k] = float(vals.std(ddof=1) if vals.size > 1 else 0.0)
    return mean, std


def _write_trials_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write one CSV row per Optuna trial, with the union of all keys."""
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class _NullLogger:
    """Minimal logger stand-in for the refit phase (no per-epoch logging)."""

    class _Cfg:
        enabled = False

    def __init__(self) -> None:
        self.cfg = self._Cfg()

    def log_fold_metrics(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            return None
        return _noop
