"""Stratified K-fold cross-validation for continuous labels."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric.data
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from src.configs.trainer_config import TrainerConfig
from src.datasets.label_builder import LabelBuilder
from src.models.base_model import BrainGNN
from src.training.checkpoint_manager import CheckpointManager
from src.training.glm_normalizer import GLMFeatureNormalizer
from src.training.label_normalizer import LabelNormalizer
from src.training.metrics import MetricDict, compute_metrics
from src.training.trainer import Trainer, TrainResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Aggregated cross-validation result.

    Attributes
    ----------
    fold_results : List[TrainResult]
        Per-fold training results.
    fold_test_metrics : List[MetricDict]
        Per-fold test metrics.
    aggregated : MetricDict
        Metrics computed on all fold test predictions pooled together.
    """

    fold_results: List[TrainResult] = field(default_factory=list)
    fold_test_metrics: List[MetricDict] = field(default_factory=list)
    aggregated: MetricDict = field(default_factory=dict)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Cross-validator
# ---------------------------------------------------------------------------

class CrossValidator:
    """Stratified K-fold cross-validation orchestrator.

    Continuous labels are binned into quantile buckets for stratification.

    Parameters
    ----------
    cfg : TrainerConfig
        Must provide ``n_folds``, ``val_ratio``, ``test_ratio``,
        ``stratify_bins``, ``seed``.
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg

    def split(
        self,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
    ) -> Iterator[Tuple[List[int], List[int], List[int]]]:
        """Yield (train, val, test) index tuples for each fold.

        Stratification bins continuous labels into ``cfg.stratify_bins``
        quantile buckets before splitting.

        Parameters
        ----------
        dataset : List[Data]
            Full list of PyG Data objects.
        labels : np.ndarray
            Shape ``[len(dataset)]``.

        Yields
        ------
        Tuple[List[int], List[int], List[int]]
            ``(train_indices, val_indices, test_indices)``
        """
        indices = np.arange(len(dataset))

        # Bin continuous labels into quantile buckets for stratification
        bins = pd.qcut(
            labels, q=self.cfg.stratify_bins, labels=False, duplicates="drop"
        )

        skf = StratifiedKFold(
            n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed
        )

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, bins)):
            # Split train_val into train / val
            rng = np.random.default_rng(self.cfg.seed + fold_idx)
            shuffled = rng.permutation(train_val_idx)
            val_size = max(1, int(len(train_val_idx) * self.cfg.val_ratio))
            val_idx = shuffled[:val_size]
            train_idx = shuffled[val_size:]

            yield train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def run(
        self,
        model_factory: Callable[[], BrainGNN],
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        trainer: Trainer,
        label_builder: Optional[LabelBuilder] = None,
        label_components: Optional[pd.DataFrame] = None,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        model_config: Optional[Dict] = None,
        feature_config: Optional[Dict] = None,
    ) -> CVResult:
        """Execute full cross-validation.

        For each fold:
        1. Split data into train / val / test indices.
        2. Build per-fold labels:
           - **Simple mode** (``label_builder`` is ``None``): slice the
             pre-computed ``labels`` array directly.
           - **Composite mode** (both ``label_builder`` and
             ``label_components`` are provided): call
             ``label_builder.fit_transform(label_components.iloc[train_idx])``
             to fit normalisation / PCA on training subjects only, then
             ``label_builder.transform(...)`` for val and test.  This
             prevents leakage from component normalisation or PCA fitting.
        3. If ``glm_col_range`` is provided, z-score the GLM feature
           columns of ``data.x`` per node — fit on train only.
        4. Fit :class:`~src.training.label_normalizer.LabelNormalizer` on
           the (per-fold) training labels to map them to a normalised scale.
        5. Create fresh model via ``model_factory``.
        6. Build DataLoaders and attach per-fold labels to each graph's
           ``data.y``.
        7. Train via ``trainer.fit``.
        8. Reload the best validation weights and evaluate them on the test split.
        9. Save checkpoint via ``CheckpointManager``.

        Parameters
        ----------
        model_factory : Callable[[], BrainGNN]
            Factory that returns a fresh, untrained model instance.
        dataset : List[Data]
            Full dataset.
        labels : np.ndarray
            Pre-computed scalar labels of shape ``[N]``.  Always required for
            stratification binning in :meth:`split`, and used directly as fold
            labels when ``label_builder`` is ``None``.
        trainer : Trainer
            Configured trainer instance.
        label_builder : Optional[LabelBuilder]
            When provided together with ``label_components``, composite labels
            are constructed per fold (leakage-free).  Must be ``None`` for
            single-column targets.
        label_components : Optional[pd.DataFrame]
            Raw component columns of shape ``[N, K]`` as returned by
            :meth:`~src.datasets.base_dataset.BrainGraphDataset.get_label_components`.
            Required when ``label_builder`` is not ``None``.
        glm_col_range : Optional[Tuple[int, int]]
            ``(col_start, col_end)`` column range of GLM features in
            ``data.x``.  When provided, per-node z-scoring is applied
            per fold (fit on train only) to prevent data leakage.

        Returns
        -------
        CVResult
            Per-fold and aggregated results.

        Raises
        ------
        ValueError
            If exactly one of ``label_builder`` / ``label_components`` is
            provided (both or neither must be supplied).
        """
        if (label_builder is None) != (label_components is None):
            raise ValueError(
                "'label_builder' and 'label_components' must both be provided "
                "or both be None."
            )

        checkpoint_manager = CheckpointManager(self.cfg.checkpoint_dir)

        # Resolve the training device once for all folds.
        if self.cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.cfg.device)
        log.info("Training device: %s", device)

        fold_results: List[TrainResult] = []
        fold_test_metrics: List[MetricDict] = []
        all_y_true: List[np.ndarray] = []
        all_y_pred: List[np.ndarray] = []

        # Helper: map integer indices → subject ID strings.
        # Raises an error if graphs lack a ``subject_id`` attribute.
        def _get_subject_ids(indices: List[int]) -> List[str]:
            if not indices:
                return []
            if not hasattr(dataset[indices[0]], "subject_id"):
                raise ValueError(
                    "Dataset graphs must have a 'subject_id' attribute. "
                    f"Found none on dataset[{indices[0]}]."
                )
            return [dataset[i].subject_id for i in indices]

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(
            self.split(dataset, labels)
        ):
            log.info(
                "Fold %d/%d — train=%d  val=%d  test=%d",
                fold_idx + 1,
                self.cfg.n_folds,
                len(train_idx),
                len(val_idx),
                len(test_idx),
            )

            # Resolve subject IDs for this fold (used for split logging)
            train_ids = _get_subject_ids(train_idx)
            val_ids   = _get_subject_ids(val_idx)
            test_ids  = _get_subject_ids(test_idx)

            # ------------------------------------------------------------------
            # Build per-fold labels
            # ------------------------------------------------------------------
            if label_builder is not None:
                y_train = label_builder.fit_transform(
                    label_components.iloc[train_idx]
                )
                y_val = label_builder.transform(label_components.iloc[val_idx])
                y_test = label_builder.transform(label_components.iloc[test_idx])
            else:
                y_train = labels[train_idx]
                y_val = labels[val_idx]
                y_test = labels[test_idx]

            # ------------------------------------------------------------------
            # Normalise labels (fit on train only)
            # ------------------------------------------------------------------
            normalizer = LabelNormalizer(strategy=self.cfg.label_norm_strategy)
            y_train_norm = normalizer.fit_transform(y_train)
            y_val_norm = normalizer.transform(y_val)
            y_test_norm = normalizer.transform(y_test)

            # ------------------------------------------------------------------
            # Build DataLoaders with labels attached to each graph
            # ------------------------------------------------------------------
            bs = self.cfg.batch_size

            def _make_loader(
                indices: List[int],
                y_values: np.ndarray,
                shuffle: bool,
                drop_last: bool = False,
            ) -> DataLoader:
                graphs = []
                for i, idx in enumerate(indices):
                    g = copy.copy(dataset[idx])
                    g.y = torch.tensor([y_values[i]], dtype=torch.float32)
                    graphs.append(g)
                return DataLoader(
                    graphs,
                    batch_size=bs,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )

            train_loader = _make_loader(
                train_idx, y_train_norm, shuffle=True,
                drop_last=len(train_idx) > bs,
            )
            val_loader = _make_loader(val_idx, y_val_norm, shuffle=False)
            test_loader = _make_loader(test_idx, y_test_norm, shuffle=False)

            # ------------------------------------------------------------------
            # GLM feature normalisation (fit on train only)
            # ------------------------------------------------------------------
            if glm_col_range is not None and glm_normalize:
                col_start, col_end = glm_col_range
                glm_norm = GLMFeatureNormalizer(col_start, col_end)
                # Collect the train graphs from the loader to fit stats
                train_graphs = [g for g in train_loader.dataset]
                glm_norm.fit(train_graphs)
                glm_norm.transform(train_graphs)
                # Apply same stats to val and test graphs
                val_graphs = [g for g in val_loader.dataset]
                glm_norm.transform(val_graphs)
                test_graphs = [g for g in test_loader.dataset]
                glm_norm.transform(test_graphs)

                
            # ------------------------------------------------------------------
            # Train and evaluate
            # ------------------------------------------------------------------
            model = model_factory()
            model = model.to(device)

            # Build an epoch-snapshot callback when periodic or best saving is requested.
            # The callback is None in sweep mode (save_every_n_epochs==0, save_on_best==False)
            # so no unnecessary I/O occurs.
            epoch_end_callback = None
            if self.cfg.save_every_n_epochs > 0 or self.cfg.save_on_best:
                def _make_epoch_callback(_fold_idx: int):
                    def _callback(epoch: int, state_dict: dict, is_best: bool) -> None:
                        do_periodic = (
                            self.cfg.save_every_n_epochs > 0
                            and (epoch + 1) % self.cfg.save_every_n_epochs == 0
                        )
                        do_best = self.cfg.save_on_best and is_best
                        if do_periodic or do_best:
                            checkpoint_manager.save_epoch_checkpoint(
                                _fold_idx, epoch, state_dict, is_best=is_best
                            )
                    return _callback
                epoch_end_callback = _make_epoch_callback(fold_idx)

            result = trainer.fit(
                model, train_loader, val_loader, normalizer, fold_idx,
                on_epoch_end_callback=epoch_end_callback,
            )

            # Evaluate the checkpoint selected by validation, not the final in-memory weights.
            model.load_state_dict(result.best_model_state_dict)

            y_true_fold, y_pred_fold = trainer.predict(model, test_loader, normalizer)
            all_y_true.append(y_true_fold)
            all_y_pred.append(y_pred_fold)
            test_metrics = trainer.evaluate(model, test_loader, normalizer, "test")

            try:
                trainer.logger.log_prediction_scatter(y_true_fold, y_pred_fold, fold_idx)
            except NotImplementedError:
                pass  # logger does not implement this method — skip silently
            except Exception as _exc:
                log.warning(
                    "log_prediction_scatter failed for fold %d (continuing): %s",
                    fold_idx, _exc, exc_info=True,
                )

            log.info(
                "Fold %d test metrics: %s", fold_idx + 1, test_metrics
            )

            # Log test metrics to wandb under fold_N/test/* for clarity in the UI.
            try:
                trainer.logger.log_fold_metrics(
                    fold_idx,
                    test_metrics,
                    split="test",
                    epoch=result.best_epoch,
                )
            except NotImplementedError:
                pass
            except Exception as _exc:
                log.warning(
                    "log_fold_metrics failed for fold %d (continuing): %s",
                    fold_idx, _exc, exc_info=True,
                )

            try:
                trainer.logger.log_fold_splits(fold_idx, train_ids, val_ids, test_ids)
            except NotImplementedError:
                pass
            except Exception as _exc:
                log.warning(
                    "log_fold_splits failed for fold %d (continuing): %s",
                    fold_idx, _exc, exc_info=True,
                )

            # ------------------------------------------------------------------
            # Checkpoint
            # ------------------------------------------------------------------
            try:
                checkpoint_manager.save_fold_checkpoint(
                    fold_idx,
                    result.best_model_state_dict,
                    result.last_model_state_dict,
                    normalizer,
                    result.best_val_metrics,
                    result.best_epoch,
                    result.last_val_metrics,
                    result.last_epoch,
                    model_config=model_config,
                    feature_config=feature_config,
                )
            except NotImplementedError:
                raise RuntimeError("Checkpointing failed — check your checkpoint manager implementation") from None

            fold_results.append(result)
            fold_test_metrics.append(test_metrics)

        aggregated = compute_metrics(
            np.concatenate(all_y_true), np.concatenate(all_y_pred)
        )
        return CVResult(
            fold_results=fold_results,
            fold_test_metrics=fold_test_metrics,
            aggregated=aggregated,
        )
