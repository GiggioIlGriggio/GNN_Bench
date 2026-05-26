"""Pretrained model loading, layer freezing, and fine-tuning loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric.data

from src.configs.finetuning_config import FinetuningConfig
from src.configs.model_config import ModelConfig
from src.configs.feature_config import FeatureConfig
from src.configs.trainer_config import TrainerConfig
from src.datasets.label_builder import LabelBuilder
from src.interfaces.adapters import load_partial_state_dict
from src.models.base_model import BrainGNN
from src.models.registry import get_model

log = logging.getLogger(__name__)


class Finetuner:
    """Orchestrates fine-tuning of a pretrained :class:`BrainGNN`.

    Parameters
    ----------
    cfg : FinetuningConfig
        Fine-tuning configuration (checkpoint path, frozen layers, LR groups).
    """

    def __init__(self, cfg: FinetuningConfig) -> None:
        self.cfg = cfg
        self._pretrained_state_dict: Optional[dict] = None
        self._source_model_config: Optional[ModelConfig] = None
        self._source_feature_config: Optional[FeatureConfig] = None
        self._checkpoint_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_pretrained(self, checkpoint_path: Path) -> None:
        """Load a pretrained checkpoint from a fold directory.

        Reads ``model_config.json``, ``feature_config.json``, and the
        selected model weights (``model_best.pt`` or ``model_last.pt``).

        Parameters
        ----------
        checkpoint_path : Path
            Path to a fold checkpoint directory (e.g. ``checkpoints/fold_0``).

        Raises
        ------
        FileNotFoundError
            If required checkpoint files are missing.
        """
        checkpoint_path = Path(checkpoint_path)
        fold_dir = checkpoint_path / f"fold_{self.cfg.checkpoint_fold}"

        # Load architecture configs
        model_cfg_path = fold_dir / "model_config.json"
        feature_cfg_path = fold_dir / "feature_config.json"
        if not model_cfg_path.exists():
            raise FileNotFoundError(
                f"model_config.json not found in {fold_dir}. "
                "Was the checkpoint saved with config metadata?"
            )
        if not feature_cfg_path.exists():
            raise FileNotFoundError(
                f"feature_config.json not found in {fold_dir}. "
                "Was the checkpoint saved with config metadata?"
            )

        with open(model_cfg_path) as f:
            self._source_model_config = ModelConfig(**json.load(f))
        with open(feature_cfg_path) as f:
            self._source_feature_config = FeatureConfig(**json.load(f))

        # Load weights via CheckpointManager
        self._pretrained_state_dict = self._load_weights_for_fold(
            checkpoint_path, self.cfg.checkpoint_fold
        )
        self._checkpoint_path = checkpoint_path
        log.info(
            "Loaded pretrained checkpoint: %s (%s variant, %d keys)",
            fold_dir,
            self.cfg.checkpoint_variant,
            len(self._pretrained_state_dict),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_weights_for_fold(self, checkpoint_path: Path, fold_idx: int) -> dict:
        """Load pretrained weights from a specific fold checkpoint.

        Used to match pretrained weights to the current CV fold so that
        the backbone never sees the fold's test subjects during pretraining.
        """
        from src.training.checkpoint_manager import CheckpointManager
        ckpt = CheckpointManager(checkpoint_path).load_fold_checkpoint(fold_idx)
        variant = self.cfg.checkpoint_variant
        state_dict = (
            ckpt.best_model_state_dict if variant == "best" else ckpt.last_model_state_dict
        )
        log.info(
            "Fold %d: loaded pretrained weights from %s (%d keys)",
            fold_idx,
            Path(checkpoint_path) / f"fold_{fold_idx}",
            len(state_dict),
        )
        return state_dict

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def make_model_factory(
        self,
        num_nodes: int = 0,
    ) -> Callable[[], BrainGNN]:
        """Return a model factory that creates models with pretrained backbone + fresh head.

        Each call to the returned factory:
        1. Instantiates a fresh model with the pretrained architecture.
        2. Loads the pretrained state dict (backbone weights).
        3. Reinitialises the prediction head.
        4. Freezes layers specified in ``cfg.frozen_layers``.

        Parameters
        ----------
        num_nodes : int
            Number of nodes per graph (needed for MLP models).

        Returns
        -------
        Callable[[], BrainGNN]
        """
        if self._pretrained_state_dict is None:
            raise RuntimeError("Call load_pretrained() before make_model_factory().")

        model_cfg = self._source_model_config
        feature_cfg = self._source_feature_config
        state_dict = self._pretrained_state_dict
        frozen_layers = self.cfg.frozen_layers

        def factory() -> BrainGNN:
            model = get_model(
                name=model_cfg.name,
                cfg=model_cfg,
                node_feat_dim=feature_cfg.node_feat_dim,
                edge_feat_dim=feature_cfg.edge_feat_dim,
                num_nodes=num_nodes,
            )

            # Load pretrained weights (matching keys only)
            loaded, skipped = load_partial_state_dict(model, state_dict)
            log.info(
                "Loaded %d/%d keys from pretrained checkpoint (%d skipped)",
                len(loaded),
                len(loaded) + len(skipped),
                len(skipped),
            )

            # Reinitialise the prediction head
            _reinit_head(model)

            # Freeze specified layers
            _freeze_layers(model, frozen_layers)

            return model

        return factory

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def build_param_groups(self, model: BrainGNN) -> List[Dict]:
        """Build optimizer parameter groups with per-layer learning rates.

        Unfrozen parameters not matching any ``lr_groups`` prefix use the
        default learning rate from the trainer config.

        Parameters
        ----------
        model : BrainGNN

        Returns
        -------
        List[Dict]
            Parameter groups suitable for ``torch.optim`` constructors.
        """
        lr_groups = self.cfg.lr_groups
        grouped_params: Dict[str, List[torch.nn.Parameter]] = {}
        default_params: List[torch.nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            matched = False
            for prefix, lr in lr_groups.items():
                if name.startswith(prefix):
                    grouped_params.setdefault(prefix, []).append(param)
                    matched = True
                    break

            if not matched:
                default_params.append(param)

        param_groups: List[Dict] = []
        for prefix, params in grouped_params.items():
            param_groups.append({"params": params, "lr": lr_groups[prefix]})

        if default_params:
            param_groups.append({"params": default_params})

        return param_groups

    # ------------------------------------------------------------------
    # Full finetuning run
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        trainer_cfg: TrainerConfig,
        logger: "WandbLogger",  # noqa: F821
        label_builder: Optional[LabelBuilder] = None,
        label_components: Optional[pd.DataFrame] = None,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
    ):
        """Run finetuning with cross-validation (same seed as pretraining).

        Parameters
        ----------
        dataset : List[Data]
            Full dataset of PyG graphs.
        labels : np.ndarray
            Target labels for the finetuning task.
        trainer_cfg : TrainerConfig
            Trainer config (seed controls CV split reproducibility).
        logger : WandbLogger
            Experiment logger.
        label_builder : Optional[LabelBuilder]
            For composite labels.
        label_components : Optional[pd.DataFrame]
            Raw component columns for composite labels.
        glm_col_range : Optional[Tuple[int, int]]
            GLM feature column range for per-fold normalisation.
        glm_normalize : bool
            Whether to z-score GLM features.

        Returns
        -------
        CVResult
        """
        from src.training.cross_validation import CrossValidator
        from src.training.trainer import Trainer

        # Override epochs from finetuning config
        ft_trainer_cfg = trainer_cfg.model_copy(
            update={"epochs": self.cfg.epochs}
        )

        num_nodes = dataset[0].num_nodes if dataset else 0
        finetuner = self
        ckpt_path = self._checkpoint_path

        # Build a sample model (from cfg.checkpoint_fold) to initialise param
        # groups for the Trainer.  Architecture is identical across all folds.
        _sample_state = self._load_weights_for_fold(ckpt_path, self.cfg.checkpoint_fold)
        _sample_model = get_model(
            name=self._source_model_config.name,
            cfg=self._source_model_config,
            node_feat_dim=self._source_feature_config.node_feat_dim,
            edge_feat_dim=self._source_feature_config.edge_feat_dim,
            num_nodes=num_nodes,
        )
        load_partial_state_dict(_sample_model, _sample_state)
        param_groups = self.build_param_groups(_sample_model)
        del _sample_model, _sample_state

        # Fold-aware factory: finetuning fold k loads pretrained weights from
        # pretraining fold k, so the backbone never saw fold k's test subjects.
        fold_counter = [0]

        def _ft_model_factory_with_groups():
            fold_idx = fold_counter[0]
            fold_counter[0] += 1

            state_dict = finetuner._load_weights_for_fold(ckpt_path, fold_idx)
            model = get_model(
                name=finetuner._source_model_config.name,
                cfg=finetuner._source_model_config,
                node_feat_dim=finetuner._source_feature_config.node_feat_dim,
                edge_feat_dim=finetuner._source_feature_config.edge_feat_dim,
                num_nodes=num_nodes,
            )
            load_partial_state_dict(model, state_dict)
            _reinit_head(model)
            _freeze_layers(model, finetuner.cfg.frozen_layers)
            ft_trainer._param_groups = finetuner.build_param_groups(model)
            return model

        ft_trainer = Trainer(
            cfg=ft_trainer_cfg,
            logger=logger,
            param_groups=param_groups,
        )

        cross_validator = CrossValidator(cfg=ft_trainer_cfg)

        log.info(
            "Starting finetuning %d-fold cross-validation (%d epochs)...",
            ft_trainer_cfg.n_folds,
            ft_trainer_cfg.epochs,
        )

        results = cross_validator.run(
            model_factory=_ft_model_factory_with_groups,
            dataset=dataset,
            labels=labels,
            trainer=ft_trainer,
            label_builder=label_builder,
            label_components=label_components,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
            model_config=self._source_model_config.model_dump(),
            feature_config=self._source_feature_config.model_dump(),
        )

        return results

    # ------------------------------------------------------------------
    # Epoch-sweep finetuning
    # ------------------------------------------------------------------

    def run_epoch_sweep(
        self,
        epoch_checkpoint_dir: "str | Path",
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        trainer_cfg: TrainerConfig,
        logger: "WandbLogger",  # noqa: F821
        label_builder: Optional[LabelBuilder] = None,
        label_components: Optional[pd.DataFrame] = None,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
    ) -> Dict[int, "CVResult"]:
        """Finetune from every epoch snapshot in *epoch_checkpoint_dir*.

        For each pretrain-epoch checkpoint available **across all folds**
        (intersection), this method:

        1. Builds a fold-aware model factory that loads
           ``fold_<F>/epoch_checkpoints/epoch_<E>.pt`` for fold *F*.
        2. Runs a full finetuning cross-validation.
        3. Logs the aggregated R² (and other metrics) to wandb with
           ``step = pretrain_epoch``.

        Parameters
        ----------
        epoch_checkpoint_dir : str | Path
            Root directory that contains ``fold_<N>/epoch_checkpoints/``
            sub-directories.  Typically the same as ``trainer.checkpoint_dir``
            used during normal training.
        dataset : List[Data]
        labels : np.ndarray
        trainer_cfg : TrainerConfig
        logger : WandbLogger
        label_builder : Optional[LabelBuilder]
        label_components : Optional[pd.DataFrame]
        glm_col_range : Optional[Tuple[int, int]]
        glm_normalize : bool

        Returns
        -------
        Dict[int, CVResult]
            Mapping from pretrain-epoch index to the full ``CVResult``.
        """
        from src.training.checkpoint_manager import CheckpointManager
        from src.training.cross_validation import CrossValidator
        from src.training.trainer import Trainer

        epoch_checkpoint_dir = Path(epoch_checkpoint_dir)
        ckpt_manager = CheckpointManager(epoch_checkpoint_dir)
        n_folds = trainer_cfg.n_folds

        # ----------------------------------------------------------------
        # Load architecture configs from fold 0 (identical for all folds)
        # ----------------------------------------------------------------
        fold0_dir = epoch_checkpoint_dir / "fold_0"
        model_cfg_path = fold0_dir / "model_config.json"
        feature_cfg_path = fold0_dir / "feature_config.json"

        if not model_cfg_path.exists():
            raise FileNotFoundError(
                f"model_config.json not found in {fold0_dir}. "
                "Ensure checkpoints were saved with config metadata."
            )
        if not feature_cfg_path.exists():
            raise FileNotFoundError(
                f"feature_config.json not found in {fold0_dir}. "
                "Ensure checkpoints were saved with config metadata."
            )

        with open(model_cfg_path) as f:
            self._source_model_config = ModelConfig(**json.load(f))
        with open(feature_cfg_path) as f:
            self._source_feature_config = FeatureConfig(**json.load(f))

        # ----------------------------------------------------------------
        # Discover common epochs across all folds
        # ----------------------------------------------------------------
        common_epochs = ckpt_manager.get_common_epochs(n_folds)
        if not common_epochs:
            raise ValueError(
                f"No epoch checkpoints found in common across all {n_folds} "
                f"folds under '{epoch_checkpoint_dir}'. "
                "Run normal training with save_every_n_epochs > 0 or "
                "save_on_best = true first."
            )
        log.info(
            "Epoch sweep: %d pretrain-epoch checkpoints to finetune from: %s",
            len(common_epochs),
            common_epochs,
        )

        # ----------------------------------------------------------------
        # Training config (override epochs from finetuning config)
        # ----------------------------------------------------------------
        ft_trainer_cfg = trainer_cfg.model_copy(
            update={"epochs": self.cfg.epochs}
        )
        num_nodes = dataset[0].num_nodes if dataset else 0
        finetuner = self

        # Build param groups using the first available epoch checkpoint of
        # the fold used for sampling (cfg.checkpoint_fold).
        _sample_path = (
            epoch_checkpoint_dir
            / f"fold_{self.cfg.checkpoint_fold}"
            / "epoch_checkpoints"
            / f"epoch_{common_epochs[0]:04d}.pt"
        )
        _sample_state = torch.load(
            _sample_path, map_location="cpu", weights_only=True
        )
        _sample_model = get_model(
            name=self._source_model_config.name,
            cfg=self._source_model_config,
            node_feat_dim=self._source_feature_config.node_feat_dim,
            edge_feat_dim=self._source_feature_config.edge_feat_dim,
            num_nodes=num_nodes,
        )
        load_partial_state_dict(_sample_model, _sample_state)
        param_groups = self.build_param_groups(_sample_model)
        del _sample_model, _sample_state

        ft_trainer = Trainer(
            cfg=ft_trainer_cfg,
            logger=logger,
            param_groups=param_groups,
        )
        cross_validator = CrossValidator(cfg=ft_trainer_cfg)

        # ----------------------------------------------------------------
        # Iterate over epochs
        # ----------------------------------------------------------------
        results_per_epoch: Dict[int, object] = {}

        for epoch in common_epochs:
            log.info(
                "Epoch sweep: finetuning from pretrain epoch %d ...", epoch
            )

            # Fresh counter per epoch so each CV fold gets the right fold index.
            fold_counter = [0]
            _epoch_dir = epoch_checkpoint_dir  # capture for closure

            def _make_factory(_epoch: int, _counter: list) -> Callable[[], BrainGNN]:
                def _factory() -> BrainGNN:
                    fi = _counter[0]
                    _counter[0] += 1
                    weights_path = (
                        _epoch_dir
                        / f"fold_{fi}"
                        / "epoch_checkpoints"
                        / f"epoch_{_epoch:04d}.pt"
                    )
                    state_dict = torch.load(
                        weights_path, map_location="cpu", weights_only=True
                    )
                    model = get_model(
                        name=finetuner._source_model_config.name,
                        cfg=finetuner._source_model_config,
                        node_feat_dim=finetuner._source_feature_config.node_feat_dim,
                        edge_feat_dim=finetuner._source_feature_config.edge_feat_dim,
                        num_nodes=num_nodes,
                    )
                    load_partial_state_dict(model, state_dict)
                    _reinit_head(model)
                    _freeze_layers(model, finetuner.cfg.frozen_layers)
                    ft_trainer._param_groups = finetuner.build_param_groups(model)
                    return model
                return _factory

            cv_result = cross_validator.run(
                model_factory=_make_factory(epoch, fold_counter),
                dataset=dataset,
                labels=labels,
                trainer=ft_trainer,
                label_builder=label_builder,
                label_components=label_components,
                glm_col_range=glm_col_range,
                glm_normalize=glm_normalize,
                model_config=self._source_model_config.model_dump(),
                feature_config=self._source_feature_config.model_dump(),
            )
            results_per_epoch[epoch] = cv_result

            log.info(
                "Epoch sweep — pretrain epoch %d: R²=%.4f  MAE=%.4f",
                epoch,
                cv_result.aggregated.get("r2", float("nan")),
                cv_result.aggregated.get("mae", float("nan")),
            )

            # Log to wandb with step = pretrain_epoch for correct x-axis
            try:
                logger.log_epoch_sweep_metrics(epoch, cv_result.aggregated)
            except Exception as exc:
                log.warning(
                    "log_epoch_sweep_metrics failed for epoch %d: %s", epoch, exc
                )

        return results_per_epoch

    @property
    def source_model_config(self) -> Optional[ModelConfig]:
        """Return the pretrained model's ModelConfig (after loading)."""
        return self._source_model_config

    @property
    def source_feature_config(self) -> Optional[FeatureConfig]:
        """Return the pretrained model's FeatureConfig (after loading)."""
        return self._source_feature_config


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _reinit_head(model: BrainGNN) -> None:
    """Reinitialise the prediction head of a BrainGNN model."""
    if hasattr(model, "head"):
        for m in model.head.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        log.info("Reinitialised prediction head")


def _freeze_layers(model: BrainGNN, frozen_prefixes: List[str]) -> None:
    """Freeze parameters whose names start with any of the given prefixes."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_prefixes):
            param.requires_grad = False
            frozen_count += 1
    if frozen_count:
        log.info("Froze %d parameters matching prefixes: %s", frozen_count, frozen_prefixes)
