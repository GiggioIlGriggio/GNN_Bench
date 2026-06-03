#!/usr/bin/env python
"""Entry point — Hydra-based experiment runner.

Usage::
    PYTHONPATH=$(pwd) python scripts/run_experiment.py \\
        dataset=orbit \\
        model=gcn \\
        trainer=default \\
        features=default \\
        labels=default

Sweep (multirun) example::

    PYTHONPATH=$(pwd) python scripts/run_experiment.py \\
        --multirun \\
        model=gcn \\
        sweeper=gcn_embedding_dim \\
        labels.target=mri_VWM_p \\
        features=glm_diagonal

Optional overrides::

    finetuning=default
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import hydra
import numpy as np
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

_VALID_RUNNERS = ("flat_cv", "nested")


def select_runner(*, is_sweep: bool, finetuning_enabled: bool, runner: str | None) -> str:
    """Decide which training runner to dispatch.

    Precedence: sweep (--multirun) > finetuning > flat_cv > nested.

    Parameters
    ----------
    is_sweep : bool
        ``True`` when Hydra is in ``MULTIRUN`` mode (``--multirun`` flag).
    finetuning_enabled : bool
        ``True`` when ``finetuning.enabled`` is set in the config.
    runner : str or None
        Value of the top-level ``runner`` config key.  ``"flat_cv"`` routes
        to the legacy :class:`~src.training.cross_validation.CrossValidator`;
        ``"nested"`` (or ``None``) routes to
        :class:`~src.training.nested_cross_validation.NestedCrossValidator`.
        An unrecognised value raises ``ValueError``.

    Returns
    -------
    str
        One of ``"sweep"``, ``"finetune"``, ``"flat_cv"``, or ``"nested"``.

    Raises
    ------
    ValueError
        If *runner* is not ``None`` and not one of ``_VALID_RUNNERS``.
    """
    if runner is not None and runner not in _VALID_RUNNERS:
        raise ValueError(
            f"Unknown runner={runner!r}. Valid values: {_VALID_RUNNERS} (or unset → 'nested')."
        )
    if is_sweep:
        return "sweep"
    if finetuning_enabled:
        return "finetune"
    if runner == "flat_cv":
        return "flat_cv"
    return "nested"


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiment",
)
def main(cfg: DictConfig) -> None:
    """Compose the full config, validate with pydantic, and run the experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra-composed configuration from all config groups.
    """

    # ------------------------------------------------------------------
    # 1) Validate each config group with the corresponding pydantic schema
    # ------------------------------------------------------------------
    from src.configs.dataset_config import DatasetConfig
    from src.configs.explainer_config import ExplainerConfig
    from src.configs.feature_config import FeatureConfig
    from src.configs.finetuning_config import FinetuningConfig
    from src.configs.label_config import LabelConfig
    from src.configs.logging_config import LoggingConfig
    from src.configs.model_config import ModelConfig
    from src.configs.trainer_config import TrainerConfig

    dataset_cfg = DatasetConfig(**OmegaConf.to_container(cfg.dataset, resolve=True))
    feature_cfg = FeatureConfig(**OmegaConf.to_container(cfg.features, resolve=True))
    label_cfg = LabelConfig(**OmegaConf.to_container(cfg.labels, resolve=True))
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    trainer_cfg = TrainerConfig(**OmegaConf.to_container(cfg.trainer, resolve=True))
    logging_cfg = LoggingConfig(**OmegaConf.to_container(cfg.logging, resolve=True))
    ft_cfg = FinetuningConfig(**OmegaConf.to_container(cfg.finetuning, resolve=True))
    explainer_cfg = ExplainerConfig(**OmegaConf.to_container(cfg.explainer, resolve=True))

    # Detect sweep mode: Hydra sets mode to MULTIRUN when --multirun is used.
    from hydra.core.hydra_config import HydraConfig

    is_sweep = False
    try:
        is_sweep = HydraConfig.get().mode == RunMode.MULTIRUN
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 1c) If finetuning, resolve the feature config from the pretrained
    #     checkpoint before the dataset is loaded.
    #
    #     • If the user did NOT pass "features=..." on the CLI the
    #       pretrained checkpoint's feature config is adopted silently.
    #     • If the user DID pass "features=..." explicitly and it is
    #       incompatible (different node_feat_dim) an error is raised.
    # ------------------------------------------------------------------
    if ft_cfg.enabled:
        import json

        task_overrides = HydraConfig.get().overrides.task
        features_explicitly_set = any(
            str(o).startswith("features=") for o in task_overrides
        )

        ckpt_feat_path = (
            Path(ft_cfg.checkpoint_path)
            / f"fold_{ft_cfg.checkpoint_fold}"
            / "feature_config.json"
        )
        if ckpt_feat_path.exists():
            with open(ckpt_feat_path) as _f:
                pretrained_feature_cfg = FeatureConfig(**json.load(_f))

            if features_explicitly_set:
                # User explicitly chose a feature config — validate it.
                if pretrained_feature_cfg.node_feat_dim != feature_cfg.node_feat_dim:
                    raise ValueError(
                        f"Feature dimension mismatch: pretrained model expects "
                        f"node_feat_dim={pretrained_feature_cfg.node_feat_dim}, "
                        f"but you explicitly requested "
                        f"node_feat_dim={feature_cfg.node_feat_dim}. "
                        f"Either omit the 'features=' override to auto-adopt the "
                        f"pretrained config, or use a compatible feature set."
                    )
            else:
                # No explicit override — silently adopt the pretrained feature config.
                log.info(
                    "Finetuning: adopting pretrained feature config from checkpoint "
                    "(node_feat_dim %d → %d).",
                    feature_cfg.node_feat_dim,
                    pretrained_feature_cfg.node_feat_dim,
                )
                feature_cfg = pretrained_feature_cfg
        else:
            log.warning(
                "Finetuning: checkpoint feature_config.json not found at %s; "
                "proceeding with current feature config.",
                ckpt_feat_path,
            )

    # ------------------------------------------------------------------
    # 1b) Print all experiment configurations in organized tables
    # ------------------------------------------------------------------
    from src.utils.model_printer import print_all_configs

    print_all_configs(
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        logging_cfg=logging_cfg,
        ft_cfg=ft_cfg,
        explainer_cfg=explainer_cfg,
    )

    # ------------------------------------------------------------------
    # 2) Seed everything for reproducibility
    # ------------------------------------------------------------------
    from src.utils.seed import seed_everything

    seed_everything(trainer_cfg.seed)
    log.info("Global seed set to %d", trainer_cfg.seed)

    # ------------------------------------------------------------------
    # 3) Initialise WandbLogger
    # ------------------------------------------------------------------
    from src.logging.wandb_logger import WandbLogger

    logger = WandbLogger(logging_cfg)
    logger.init_run(OmegaConf.to_container(cfg, resolve=True))

    # ------------------------------------------------------------------
    # 4) Instantiate dataset via registry
    # ------------------------------------------------------------------
    from src.datasets.registry import get_dataset

    dataset = get_dataset(
        name=dataset_cfg.name,
        cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        label_cfg=label_cfg,
    )

    # ------------------------------------------------------------------
    # 5) Load raw data and build the list of PyG graph objects
    # ------------------------------------------------------------------
    log.info("Loading dataset '%s' from '%s'...", dataset_cfg.name, dataset_cfg.root)
    dataset.load_raw()
    graphs: List = dataset.get_dataset()
    labels: np.ndarray = dataset.get_labels()
    log.info("Dataset ready — %d subjects loaded", len(graphs))

    # When composite labels are configured, prepare the stateful
    # label_builder / label_components pair so that CrossValidator
    # can build per-fold labels without data leakage.
    from src.datasets.label_builder import LabelBuilder

    label_builder = None
    label_components = None

    if label_cfg.is_composite:
        label_builder = LabelBuilder(label_cfg)
        label_components = dataset.get_label_components()
        log.info(
            "Composite label '%s' configured — columns: %s",
            label_cfg.composite_method,
            label_cfg.composite_columns,
        )

    # Feature column ranges, read from the dataset's FeatureBuilder which has
    # already built every graph (get_dataset above). Single source of truth.
    fb = dataset.feature_builder

    glm_col_range = fb.get_glm_column_range()
    if glm_col_range is not None:
        log.info(
            "GLM features at columns [%d:%d) (normalisation %s).",
            glm_col_range[0], glm_col_range[1],
            "ENABLED, per-fold" if feature_cfg.glm_normalize else "disabled",
        )

    lap_pe_col_range = fb.get_laplacian_pe_column_range()
    if lap_pe_col_range is not None:
        log.info(
            "laplacian_pe at columns [%d:%d) — train-time sign-flip enabled.",
            lap_pe_col_range[0], lap_pe_col_range[1],
        )
        trainer_cfg = trainer_cfg.model_copy(
            update={"sign_flip_cols": lap_pe_col_range}
        )

    # Log dataset statistics to wandb.
    from src.logging.wandb_logger import DatasetStats

    # Calculate edge statistics
    edge_counts = [g.edge_index.shape[1] for g in graphs] if graphs else []
    mean_edges = float(np.mean(edge_counts)) if edge_counts else 0.0
    std_edges = float(np.std(edge_counts)) if edge_counts else 0.0

    logger.log_dataset_stats(
        DatasetStats(
            name=dataset_cfg.name,
            num_subjects=len(graphs),
            num_nodes=graphs[0].num_nodes if graphs else 0,
            modality=dataset_cfg.modality,
            atlas=dataset_cfg.atlas,
            mean_edges=mean_edges,
            std_edges=std_edges,
        )
    )

    # ------------------------------------------------------------------
    # 6) Define model factory (called fresh per fold / per trial)
    # ------------------------------------------------------------------
    from src.models.registry import get_model

    def model_factory():
        return get_model(
            name=model_cfg.name,
            cfg=model_cfg,
            node_feat_dim=feature_cfg.node_feat_dim,
            edge_feat_dim=feature_cfg.edge_feat_dim,
            num_nodes=graphs[0].num_nodes if graphs else 0,
        )

    log.info("Model '%s' (backbone=%s) ready", model_cfg.name, model_cfg.backbone)

    # ------------------------------------------------------------------
    # 6b) Print professional model summary (console + wandb logging)
    # ------------------------------------------------------------------
    from src.utils.model_printer import print_model_summary

    sample_model = model_factory()
    print_model_summary(sample_model, model_cfg)
    logger.log_model_summary(sample_model, model_cfg)

    # ------------------------------------------------------------------
    # 7) Dispatch to the appropriate runner
    # ------------------------------------------------------------------
    runner = select_runner(
        is_sweep=is_sweep,
        finetuning_enabled=ft_cfg.enabled,
        runner=cfg.get("runner"),
    )
    log.info("Runner selected: %s", runner)

    if runner == "sweep":
        from src.sweeps.hydra_sweep import HydraSweep
        from src.training.cross_validation import CrossValidator
        from src.training.trainer import Trainer

        # Read objective_metric from the top-level config (set in experiment.yaml,
        # overridden on the CLI per sweeper, e.g. objective_metric=r2).
        objective_metric = cfg.get("objective_metric", "val_mae")

        log.info("Sweep trial — objective_metric=%s", objective_metric)
        trainer = Trainer(cfg=trainer_cfg, logger=logger)
        cross_validator = CrossValidator(cfg=trainer_cfg)
        sweep = HydraSweep(cfg)
        objective, results, is_best_trial = sweep.run(
            model_factory=model_factory,
            dataset=graphs,
            labels=labels,
            trainer=trainer,
            cross_validator=cross_validator,
            objective_metric=objective_metric,
            logger=logger,
            label_builder=label_builder,
            label_components=label_components,
            glm_col_range=glm_col_range,
            glm_normalize=feature_cfg.glm_normalize,
            model_config=model_cfg.model_dump(),
            feature_config=feature_cfg.model_dump(),
        )
        log.info("Cross-validation complete (sweep trial)")
        _log_cv_results(results, logger)
        if is_best_trial:
            log.info(
                "This trial is the current best — running GNNExplainer on "
                "the promoted checkpoints."
            )
            _maybe_run_explainer(explainer_cfg, graphs, labels, model_factory, trainer_cfg,
                                 glm_col_range, feature_cfg.glm_normalize)
        else:
            log.info(
                "This trial is not the best so far — skipping GNNExplainer."
            )

    # ------------------------------------------------------------------
    # 8) Finetuning runner → load checkpoint and fine-tune
    # ------------------------------------------------------------------
    elif runner == "finetune":
        from src.finetuning.finetuner import Finetuner

        finetuner = Finetuner(ft_cfg)

        if ft_cfg.epoch_checkpoint_dir:
            # -------------------------------------------------------
            # 8a) Epoch-sweep mode: finetune from every saved epoch
            #     snapshot and log R² as a function of pretrain epoch.
            # -------------------------------------------------------
            log.info(
                "Epoch-sweep finetuning from epoch checkpoints in: %s",
                ft_cfg.epoch_checkpoint_dir,
            )
            epoch_results = finetuner.run_epoch_sweep(
                epoch_checkpoint_dir=ft_cfg.epoch_checkpoint_dir,
                dataset=graphs,
                labels=labels,
                trainer_cfg=trainer_cfg,
                logger=logger,
                label_builder=label_builder,
                label_components=label_components,
                glm_col_range=glm_col_range,
                glm_normalize=feature_cfg.glm_normalize,
            )
            log.info(
                "Epoch-sweep complete — finetuned from %d checkpoints",
                len(epoch_results),
            )
            # Log summary for each epoch and pick the best R² for the
            # final wandb summary.
            best_epoch, best_result = max(
                epoch_results.items(),
                key=lambda kv: kv[1].aggregated.get("r2", float("-inf")),
            )
            log.info(
                "Best R² after epoch-sweep finetuning: R²=%.4f at pretrain epoch %d",
                best_result.aggregated.get("r2", float("nan")),
                best_epoch,
            )
            for pretrain_epoch, cv_res in sorted(epoch_results.items()):
                agg = cv_res.aggregated
                log.info(
                    "  Pretrain epoch %d — MAE=%.4f  RMSE=%.4f  R²=%.4f  r=%.4f",
                    pretrain_epoch,
                    agg["mae"], agg["rmse"], agg["r2"], agg["pearson_r"],
                )
            _log_cv_results(best_result, logger)
            _maybe_run_explainer(explainer_cfg, graphs, labels, model_factory, trainer_cfg,
                                 glm_col_range, feature_cfg.glm_normalize)
        else:
            # -------------------------------------------------------
            # 8b) Standard finetuning from a single checkpoint
            # -------------------------------------------------------
            log.info("Fine-tuning from checkpoint: %s", ft_cfg.checkpoint_path)

            # Load pretrained checkpoint (model config + weights)
            finetuner.load_pretrained(Path(ft_cfg.checkpoint_path))

            # Run finetuning with cross-validation (same seed → same splits)
            results = finetuner.run(
                dataset=graphs,
                labels=labels,
                trainer_cfg=trainer_cfg,
                logger=logger,
                label_builder=label_builder,
                label_components=label_components,
                glm_col_range=glm_col_range,
                glm_normalize=feature_cfg.glm_normalize,
            )
            log.info("Fine-tuning cross-validation complete")
            _log_cv_results(results, logger)
            _maybe_run_explainer(explainer_cfg, graphs, labels, model_factory, trainer_cfg,
                                 glm_col_range, feature_cfg.glm_normalize)

    # ------------------------------------------------------------------
    # 9a) Flat CV (legacy CrossValidator — no inner HPO, no explainer)
    # ------------------------------------------------------------------
    elif runner == "flat_cv":
        from src.training.cross_validation import CrossValidator, build_flat_cv_artifact
        from src.training.trainer import Trainer
        from src.training.run_identity import build_run_name

        log.info("Flat CV (legacy CrossValidator) — no HPO, no sweep, no explainer.")
        trainer = Trainer(cfg=trainer_cfg, logger=logger)
        cross_validator = CrossValidator(cfg=trainer_cfg)
        results = cross_validator.run(
            model_factory=model_factory,
            dataset=graphs,
            labels=labels,
            trainer=trainer,
            label_builder=label_builder,
            label_components=label_components,
            glm_col_range=glm_col_range,
            glm_normalize=feature_cfg.glm_normalize,
            model_config=model_cfg.model_dump(),
            feature_config=feature_cfg.model_dump(),
        )
        log.info("Flat cross-validation complete")
        _log_cv_results(results, logger)

        # Persist a nested_cv_result.json-compatible artifact under the per-run
        # checkpoint dir (ADR-0012), so scripts/pooled_vs_meanfolds.py works unchanged.
        run_name = build_run_name(cfg.experiment_name)
        ckpt_root = Path(trainer_cfg.checkpoint_dir) / run_name
        artifact = build_flat_cv_artifact(
            results, cfg=trainer_cfg, run_name=run_name, model_name=model_cfg.name,
        )
        artifact.save(ckpt_root / "nested_cv_result.json")
        log.info("Saved flat-CV artifact to %s", ckpt_root / "nested_cv_result.json")

    # ------------------------------------------------------------------
    # 9b) Nested cross-validation (ADR-0008) — default runner
    # ------------------------------------------------------------------
    else:
        from src.training.nested_cross_validation import NestedCrossValidator
        from src.training.run_identity import build_run_name

        def nested_model_factory(trial_model_cfg):
            return get_model(
                name=trial_model_cfg.name,
                cfg=trial_model_cfg,
                node_feat_dim=feature_cfg.node_feat_dim,
                edge_feat_dim=feature_cfg.edge_feat_dim,
                num_nodes=graphs[0].num_nodes if graphs else 0,
            )

        ncv = NestedCrossValidator(
            cfg=trainer_cfg, search_space_path=trainer_cfg.search_space,
        )
        log.info(
            "Starting nested CV — reps=%d  outer_folds=%d  inner_trials=%d  hpo_metric=%s",
            trainer_cfg.n_repetitions,
            trainer_cfg.effective_n_outer_folds,
            trainer_cfg.inner_hpo_trials,
            trainer_cfg.hpo_metric,
        )
        nested_result = ncv.run(
            model_factory=nested_model_factory,
            dataset=graphs,
            labels=labels,
            base_model_cfg=model_cfg,
            logger=logger,
            run_name=build_run_name(cfg.experiment_name),
            label_builder=label_builder,
            label_components=label_components,
            glm_col_range=glm_col_range,
            glm_normalize=feature_cfg.glm_normalize,
            feature_config=feature_cfg.model_dump(),
        )
        log.info(
            "Nested CV complete — mean=%s  std=%s",
            nested_result.mean_metrics, nested_result.std_metrics,
        )
        _maybe_run_explainer(
            explainer_cfg, graphs, labels, model_factory, trainer_cfg,
            glm_col_range, feature_cfg.glm_normalize,
        )

    # ------------------------------------------------------------------
    # 10) Finish wandb run and (for sweep trials) return the objective
    # ------------------------------------------------------------------
    log.info("Experiment '%s' finished.", cfg.experiment_name)
    logger.finish()

    # Return the objective for Hydra/Optuna sweepers (no-op for standard CV).
    if is_sweep:
        return objective

def _log_cv_results(results, logger) -> None:
    """Log per-fold and pooled cross-validation metrics.

    Shared by the standard CV branch and the sweep branch so no logic is
    duplicated.

    Parameters
    ----------
    results : CVResult
    logger : WandbLogger
    """
    for fold_idx, fm in enumerate(results.fold_test_metrics):
        log.info(
            "  Fold %d — MAE=%.4f  RMSE=%.4f  R\u00b2=%.4f  Pearson r=%.4f",
            fold_idx + 1,
            fm["mae"], fm["rmse"], fm["r2"], fm["pearson_r"],
        )
    agg = results.aggregated
    log.info(
        "Pooled — MAE=%.4f  RMSE=%.4f  R\u00b2=%.4f  Pearson r=%.4f",
        agg["mae"], agg["rmse"], agg["r2"], agg["pearson_r"],
    )
    logger.log_final_summary(results)


def _maybe_run_explainer(
    explainer_cfg,
    graphs,
    labels,
    model_factory,
    trainer_cfg,
    glm_col_range,
    glm_normalize: bool,
) -> None:
    """Launch GNNExplainer post-training analysis when enabled.

    This helper is called at the end of every training branch (standard CV,
    finetuning, sweep).  It is a no-op when ``explainer_cfg.enabled`` is
    ``False``.

    Parameters
    ----------
    explainer_cfg : ExplainerConfig
    graphs : list of PyG Data
    labels : np.ndarray
    model_factory : Callable
        Fallback factory; the explainer prefers loading architecture from
        saved checkpoint JSON files.
    trainer_cfg : TrainerConfig
    glm_col_range : Optional[Tuple[int, int]]
    glm_normalize : bool
    """
    if not explainer_cfg.enabled:
        return

    from src.gnn_explainer.explainer import GNNExplainerRunner

    log.info("Running GNNExplainer post-training analysis...")
    runner = GNNExplainerRunner(cfg=explainer_cfg)
    output_dir = runner.run(
        dataset=graphs,
        labels=labels,
        model_factory=model_factory,
        trainer_cfg=trainer_cfg,
        glm_col_range=glm_col_range,
        glm_normalize=glm_normalize,
    )
    log.info("GNNExplainer analysis complete — outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()