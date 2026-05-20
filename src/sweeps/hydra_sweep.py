"""Hydra Sweeper integration for hyperparameter search.

The sweep search space is defined entirely in ``configs/sweeper/*.yaml``
— no hard-coded ranges appear here.

Checkpoint behaviour
--------------------
Each sweep trial saves its per-fold checkpoints to a temporary directory
``<checkpoint_dir>/_trial_current/``.  After computing the objective, the
trial's result is compared against the current best (tracked in
``<checkpoint_dir>/_best_objective.json``).  If the trial is an
improvement, its checkpoints are promoted to ``<checkpoint_dir>/fold_*/``,
replacing the previous best.  The temporary directory is always cleaned up
at the end of the trial.

The checkpoint directory is set by ``trainer.checkpoint_dir`` which is
injected from the SLURM template as ``checkpoints/<sweep_name>``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch_geometric.data
from omegaconf import DictConfig, OmegaConf

from src.datasets.label_builder import LabelBuilder
from src.logging.wandb_logger import WandbLogger
from src.sweeps.base_sweep import SweepRunner
from src.training.cross_validation import CrossValidator
from src.training.trainer import Trainer

log = logging.getLogger(__name__)


class HydraSweep(SweepRunner):
    """Wraps the Hydra Sweeper plugin (supports Optuna, random, grid).

    Hydra's multirun calls ``main(cfg)`` once per trial with the trial's
    hyperparameter overrides already injected into ``cfg``.  All dataset /
    model / trainer objects are built in ``main()`` and passed here — nothing
    is duplicated.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra-composed configuration for this trial.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        model_factory: Callable[[], "BrainGNN"],  # noqa: F821
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        trainer: Trainer,
        cross_validator: CrossValidator,
        objective_metric: str,
        logger: WandbLogger,
        label_builder: Optional[LabelBuilder] = None,
        label_components: Optional[pd.DataFrame] = None,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        model_config: Optional[Dict] = None,
        feature_config: Optional[Dict] = None,
    ) -> tuple:
        """Execute one sweep trial using pre-built objects from ``main()``.

        Checkpoints for this trial are saved to a temporary directory.
        After the trial finishes, the objective is compared against the
        running best stored in ``_best_objective.json``.  If this trial
        is better (lower for "minimize" metrics, higher for "maximize"),
        its fold checkpoints are promoted to the main checkpoint directory.

        Returns
        -------
        tuple[float, CVResult]
            ``(objective, results)`` — the objective metric value for Optuna
            to optimise, and the full cross-validation result so the caller
            can log fold-level metrics without duplicating logic here.
        """
        from src.training.cross_validation import CVResult  # local to avoid circular

        # ------------------------------------------------------------------
        # Determine the sweep's checkpoint directory and set up a temp dir
        # for this trial's checkpoints.
        # ------------------------------------------------------------------
        sweep_ckpt_dir = Path(cross_validator.cfg.checkpoint_dir)
        sweep_ckpt_dir.mkdir(parents=True, exist_ok=True)

        trial_dir = sweep_ckpt_dir / "_trial_current"

        # Create a modified CrossValidator that writes to the temp dir so
        # the main checkpoint dir only ever holds the best trial's weights.
        temp_cfg = cross_validator.cfg.model_copy(
            update={"checkpoint_dir": str(trial_dir)}
        )
        temp_cv = CrossValidator(temp_cfg)

        # ------------------------------------------------------------------
        # Run cross-validation for this trial.
        # ------------------------------------------------------------------
        results: CVResult = temp_cv.run(
            model_factory=model_factory,
            dataset=dataset,
            labels=labels,
            trainer=trainer,
            label_builder=label_builder,
            label_components=label_components,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
            model_config=model_config,
            feature_config=feature_config,
        )
        print("Cross-validation complete. Results: %s", results.aggregated)
        objective = float(results.aggregated.get(objective_metric, float("inf")))
        log.info("Trial objective (%s) = %.4f", objective_metric, objective)

        # ------------------------------------------------------------------
        # Determine the optimisation direction (minimize / maximize) from
        # the Hydra sweeper config.  Defaults to "minimize" if not found.
        # ------------------------------------------------------------------
        direction = "minimize"
        try:
            from hydra.core.hydra_config import HydraConfig
            hc_direction = HydraConfig.get().sweeper.direction
            if hc_direction is not None:
                direction = str(hc_direction)
        except Exception:
            pass

        def _is_better(new_obj: float, best_obj: float) -> bool:
            if direction == "maximize":
                return new_obj > best_obj
            return new_obj < best_obj  # "minimize" (default)

        # ------------------------------------------------------------------
        # Compare with the stored best objective and promote if better.
        # ------------------------------------------------------------------
        best_file = sweep_ckpt_dir / "_best_objective.json"
        best_objective: Optional[float] = None
        if best_file.exists():
            try:
                with open(best_file) as f:
                    best_objective = json.load(f).get("objective")
            except Exception:
                best_objective = None

        is_best = best_objective is None or _is_better(objective, best_objective)

        if is_best:
            # Promote trial checkpoints: replace fold_* dirs in sweep_ckpt_dir.
            for fold_dir in sorted(trial_dir.glob("fold_*")):
                dest = sweep_ckpt_dir / fold_dir.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(fold_dir, dest)

            with open(best_file, "w") as f:
                json.dump(
                    {"objective": objective, "metric": objective_metric, "direction": direction},
                    f,
                    indent=2,
                )
            log.info(
                "New best trial! %s=%.4f (direction=%s) — checkpoints saved to %s",
                objective_metric, objective, direction, sweep_ckpt_dir,
            )
        else:
            log.info(
                "Trial %s=%.4f did not improve best=%.4f — checkpoints discarded.",
                objective_metric, objective, best_objective,
            )

        # Always clean up the temp dir.
        shutil.rmtree(trial_dir, ignore_errors=True)

        # ------------------------------------------------------------------
        # Log trial to wandb.
        # ------------------------------------------------------------------
        logger.log_sweep_trial(
            params=OmegaConf.to_container(self.cfg, resolve=True),
            objective=objective,
        )

        return objective, results, is_best
