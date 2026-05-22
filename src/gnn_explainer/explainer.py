"""Post-training GNNExplainer runner.

After cross-validation (standard, sweep, or finetuning), this module
re-loads the best-checkpoint model for each fold and runs
:class:`torch_geometric.explain.GNNExplainer` on that fold's held-out
**test** graphs.

Outputs
-------
For each fold ``i``:

* ``<output_dir>/fold_<i>/importance_matrix.npy``
    Symmetric ``[N, N]`` float32 array where entry ``[u, v]`` is the mean
    edge-mask importance for the brain connection u↔v, averaged over all
    test subjects in the fold.

* ``<output_dir>/fold_<i>/subjects/<subject_id>_edge_mask.pt``  (optional)
    Per-subject edge-mask tensor of shape ``[E]``.

Aggregated across folds:

* ``<output_dir>/importance_matrix_avg.npy``
    Fold-average importance matrix.
* ``<output_dir>/importance_matrix_std.npy``
    Fold standard deviation of importance matrices.

Design notes
------------
edge_size scaling
~~~~~~~~~~~~~~~~~
The total regularisation loss added at every GNNExplainer step is::

    edge_size_coeff × num_nodes × Σ edge_mask

For brain connectivity graphs the number of nodes (ROIs) is fixed (e.g. 100
for a Schaefer-100 parcellation), but the *density* varies by modality:
functional connectivity is often dense (avg_degree ≈ 60–90) while structural
tractography graphs are sparser (avg_degree ≈ 10–30).  A high ``edge_size``
will push explanations toward very few edges — useful to isolate the most
critical connections, but risky if set too high.

Rule of thumb: ``edge_size ≈ 1 / avg_degree``.  The default (0.005) matches
the PyG default and works well for avg_degree ≈ 200 (fully-connected graph).
Override via the YAML field or CLI, e.g.::

    explainer.edge_size=0.02
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_geometric.data
from torch_geometric.data import Data
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.configs.explainer_config import ExplainerConfig
from src.configs.trainer_config import TrainerConfig

log = logging.getLogger(__name__)

_console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Thin wrapper so GNNExplainer can call model(x, edge_index, **kwargs)
# ---------------------------------------------------------------------------

class _DataWrapper(torch.nn.Module):
    """Adapt a :class:`~src.models.base_model.BrainGNN` to the
    ``(x, edge_index, **kwargs)`` calling convention expected by the
    PyG :class:`~torch_geometric.explain.Explainer` API.

    Parameters
    ----------
    model : BrainGNN
        Trained brain GNN model.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch = kwargs.get("batch", None)
        edge_attr = kwargs.get("edge_attr", None)

        # Build a minimal PyG Data object
        data = Data(x=x, edge_index=edge_index)
        if edge_attr is not None:
            data.edge_attr = edge_attr
        if batch is not None:
            data.batch = batch

        return self.model(data)  # [B, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_mask_to_importance_matrix(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes: int,
) -> np.ndarray:
    """Convert a flat edge-mask vector to a symmetric *N × N* importance matrix.

    For directed graphs (both u→v and v→u stored) the entry ``[u, v]`` is
    the average of the mask values for the two directed edges, giving a
    symmetric matrix.  For graphs that only store one direction the matrix
    is symmetrised afterwards.

    Parameters
    ----------
    edge_index : torch.Tensor
        Shape ``[2, E]``, integer.
    edge_mask : torch.Tensor
        Shape ``[E]``, float.
    num_nodes : int

    Returns
    -------
    np.ndarray
        Shape ``[num_nodes, num_nodes]``, float32.
    """
    mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    count = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    mask_np = edge_mask.detach().cpu().float().numpy()

    for k in range(len(src)):
        u, v = int(src[k]), int(dst[k])
        mat[u, v] += mask_np[k]
        count[u, v] += 1

    # Average where multiple edges exist (can happen in multigraphs)
    valid = count > 0
    mat[valid] /= count[valid]

    # Symmetrize: take average of (u,v) and (v,u) entries
    mat = (mat + mat.T) / 2.0
    return mat


def _compute_avg_degree(
    graphs: List[torch_geometric.data.Data],
) -> float:
    """Return the average node degree across a list of PyG graphs."""
    degrees = []
    for g in graphs:
        n = g.num_nodes
        if n and n > 0:
            degrees.append(g.edge_index.shape[1] / n)
    return float(np.mean(degrees)) if degrees else 1.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class GNNExplainerRunner:
    """Orchestrates post-training GNNExplainer analysis across all CV folds.

    Parameters
    ----------
    cfg : ExplainerConfig
        Explainer hyperparameters and output settings.
    """

    def __init__(self, cfg: ExplainerConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        model_factory: Callable[[], torch.nn.Module],
        trainer_cfg: TrainerConfig,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
    ) -> Path:
        """Run GNNExplainer for every CV fold and save importance matrices.

        Two checkpoint layouts are supported:

        * Nested CV (``ckpt_root/rep_<R>/fold_<K>/``) — selected when
          ``ckpt_root/nested_cv_result.json`` is present. Outer splits are
          reconstructed from the persisted ``outer_seeds``. Per-fold outputs
          land under ``explanations/rep_<R>/fold_<K>/``; when
          ``n_repetitions > 1`` cross-rep mean/std maps are also written
          under ``explanations/aggregate/``.
        * Legacy (``ckpt_root/fold_<K>/``) — used otherwise. Splits are
          reconstructed via :class:`~src.training.cross_validation.CrossValidator`
          and per-fold outputs land under ``explanations/fold_<K>/``.

        For every fold the model is reconstructed from the checkpoint's saved
        ``model_config.json`` / ``feature_config.json`` (written by
        :class:`~src.training.fold_checkpoint.CheckpointManager`).
        *model_factory* is used as a fallback only when those files are
        absent.

        Parameters
        ----------
        dataset, labels, model_factory, trainer_cfg, glm_col_range, glm_normalize
            See class-level docs.

        Returns
        -------
        Path
            The directory where explanation outputs were saved.
        """
        # --- Resolve output directory ----------------------------------------
        if self.cfg.output_dir:
            output_dir = Path(self.cfg.output_dir)
        else:
            output_dir = Path(trainer_cfg.checkpoint_dir) / "explanations"
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info("GNNExplainer outputs → %s", output_dir)

        # --- Device ----------------------------------------------------------
        if trainer_cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(trainer_cfg.device)
        log.info("GNNExplainer device: %s", device)

        # --- Resolve edge_size (auto or fixed) --------------------------------
        avg_degree = _compute_avg_degree(dataset)
        if self.cfg.edge_size == "auto":
            resolved_edge_size = 1.0 / avg_degree if avg_degree > 0 else 0.005
            log.info(
                "edge_size=auto → using 1 / avg_degree = 1 / %.2f = %.4f",
                avg_degree,
                resolved_edge_size,
            )
        else:
            resolved_edge_size = float(self.cfg.edge_size)
            recommended = 1.0 / avg_degree if avg_degree > 0 else 0.005
            log.info(
                "Dataset average node degree: %.2f  "
                "(recommended edge_size ≈ %.4f, using configured: %.4f)",
                avg_degree,
                recommended,
                resolved_edge_size,
            )

        ckpt_root = Path(trainer_cfg.checkpoint_dir)
        if (ckpt_root / "nested_cv_result.json").exists():
            return self._run_nested(
                ckpt_root=ckpt_root,
                output_dir=output_dir,
                dataset=dataset,
                labels=labels,
                model_factory=model_factory,
                trainer_cfg=trainer_cfg,
                device=device,
                resolved_edge_size=resolved_edge_size,
                glm_col_range=glm_col_range,
                glm_normalize=glm_normalize,
            )
        return self._run_legacy(
            ckpt_root=ckpt_root,
            output_dir=output_dir,
            dataset=dataset,
            labels=labels,
            model_factory=model_factory,
            trainer_cfg=trainer_cfg,
            device=device,
            resolved_edge_size=resolved_edge_size,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
        )

    # ------------------------------------------------------------------
    # Layout dispatchers
    # ------------------------------------------------------------------

    def _run_legacy(
        self,
        *,
        ckpt_root: Path,
        output_dir: Path,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        model_factory: Callable[[], torch.nn.Module],
        trainer_cfg: TrainerConfig,
        device: torch.device,
        resolved_edge_size: float,
        glm_col_range: Optional[Tuple[int, int]],
        glm_normalize: bool,
    ) -> Path:
        from src.training.cross_validation import CrossValidator

        cross_validator = CrossValidator(cfg=trainer_cfg)
        fold_splits = list(cross_validator.split(dataset, labels))
        n_folds = len(fold_splits)

        fold_importance_matrices: List[np.ndarray] = []
        for fold_idx, (train_idx, _val_idx, test_idx) in enumerate(fold_splits):
            _console.rule(
                f"[bold cyan]GNNExplainer — Fold {fold_idx + 1}/{n_folds}[/]  "
                f"[dim](test: {len(test_idx)} subjects)[/]",
                style="cyan dim",
            )
            fold_ckpt_dir = ckpt_root / f"fold_{fold_idx}"
            fold_out_dir = output_dir / f"fold_{fold_idx}"
            fold_avg = self._explain_one_fold(
                fold_ckpt_root=ckpt_root,
                fold_idx=fold_idx,
                fold_ckpt_dir=fold_ckpt_dir,
                fold_out_dir=fold_out_dir,
                train_idx=list(train_idx),
                test_idx=list(test_idx),
                dataset=dataset,
                model_factory=model_factory,
                device=device,
                resolved_edge_size=resolved_edge_size,
                glm_col_range=glm_col_range,
                glm_normalize=glm_normalize,
                fold_log_tag=f"Fold {fold_idx}",
            )
            if fold_avg is not None:
                fold_importance_matrices.append(fold_avg)

        if not fold_importance_matrices:
            log.error(
                "No fold explanations were produced.  Check checkpoints and "
                "enable logging at DEBUG level for details."
            )
            return output_dir

        global_avg = np.mean(np.stack(fold_importance_matrices, axis=0), axis=0)
        global_std = np.std(np.stack(fold_importance_matrices, axis=0), axis=0)
        np.save(output_dir / "importance_matrix_avg.npy", global_avg)
        np.save(output_dir / "importance_matrix_std.npy", global_std)
        log.info(
            "GNNExplainer complete — global importance matrix saved to %s  "
            "(N=%d, max=%.4f, mean=%.4f).",
            output_dir,
            global_avg.shape[0],
            float(global_avg.max()),
            float(global_avg.mean()),
        )
        self._log_top_edges(global_avg)
        return output_dir

    def _run_nested(
        self,
        *,
        ckpt_root: Path,
        output_dir: Path,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        model_factory: Callable[[], torch.nn.Module],
        trainer_cfg: TrainerConfig,
        device: torch.device,
        resolved_edge_size: float,
        glm_col_range: Optional[Tuple[int, int]],
        glm_normalize: bool,
    ) -> Path:
        import json as _json

        import pandas as pd
        from sklearn.model_selection import StratifiedKFold

        with open(ckpt_root / "nested_cv_result.json") as f:
            nested_meta = _json.load(f)
        outer_seeds: List[int] = list(nested_meta["outer_seeds"])
        n_outer = int(nested_meta["n_outer_folds"])
        n_repetitions = int(nested_meta["n_repetitions"])
        log.info(
            "Nested layout detected — reps=%d  outer_folds=%d  seeds=%s",
            n_repetitions, n_outer, outer_seeds,
        )

        # Same bin construction as NestedCrossValidator so splits match.
        bins = pd.qcut(
            labels, q=trainer_cfg.stratify_bins, labels=False, duplicates="drop",
        )

        # fold_matrices_by_idx[k] = list of per-rep mean importance matrices for outer fold k
        fold_matrices_by_idx: Dict[int, List[np.ndarray]] = {k: [] for k in range(n_outer)}
        all_matrices: List[np.ndarray] = []

        if self.cfg.rep is not None and not (
            0 <= self.cfg.rep < n_repetitions
        ):
            raise ValueError(
                f"explainer.rep={self.cfg.rep} is outside the available range "
                f"[0, {n_repetitions})"
            )

        for rep, seed in enumerate(outer_seeds):
            if self.cfg.rep is not None and rep != self.cfg.rep:
                continue
            skf = StratifiedKFold(
                n_splits=n_outer, shuffle=True, random_state=seed,
            )
            for fold_idx, (train_val_idx, test_idx) in enumerate(
                skf.split(np.arange(len(dataset)), bins)
            ):
                _console.rule(
                    f"[bold cyan]GNNExplainer — rep {rep + 1}/{n_repetitions} "
                    f"fold {fold_idx + 1}/{n_outer}[/]  "
                    f"[dim](test: {len(test_idx)} subjects)[/]",
                    style="cyan dim",
                )
                fold_ckpt_dir = ckpt_root / f"rep_{rep}" / f"fold_{fold_idx}"
                fold_out_dir = output_dir / f"rep_{rep}" / f"fold_{fold_idx}"
                # NestedCrossValidator refits the outer model on the
                # full outer-TrainVal pool, so GLM normalisation must
                # be re-fit on that same pool — not on the inner-train slice.
                fold_avg = self._explain_one_fold(
                    fold_ckpt_root=ckpt_root / f"rep_{rep}",
                    fold_idx=fold_idx,
                    fold_ckpt_dir=fold_ckpt_dir,
                    fold_out_dir=fold_out_dir,
                    train_idx=train_val_idx.tolist(),
                    test_idx=test_idx.tolist(),
                    dataset=dataset,
                    model_factory=model_factory,
                    device=device,
                    resolved_edge_size=resolved_edge_size,
                    glm_col_range=glm_col_range,
                    glm_normalize=glm_normalize,
                    fold_log_tag=f"rep={rep} fold={fold_idx}",
                )
                if fold_avg is not None:
                    fold_matrices_by_idx[fold_idx].append(fold_avg)
                    all_matrices.append(fold_avg)

        if not all_matrices:
            log.error(
                "No fold explanations were produced.  Check checkpoints and "
                "enable logging at DEBUG level for details."
            )
            return output_dir

        # Per-fold cross-rep aggregation. Only meaningful when more than one
        # repetition contributed maps for that fold index, but we always write
        # the directory so downstream readers don't have to special-case n_rep=1.
        aggregate_dir = output_dir / "aggregate"
        for fold_idx, mats in fold_matrices_by_idx.items():
            if not mats:
                continue
            fold_agg_dir = aggregate_dir / f"fold_{fold_idx}"
            fold_agg_dir.mkdir(parents=True, exist_ok=True)
            stack = np.stack(mats, axis=0)
            np.save(fold_agg_dir / "importance_matrix_avg.npy", stack.mean(axis=0))
            np.save(fold_agg_dir / "importance_matrix_std.npy", stack.std(axis=0))

        global_avg = np.mean(np.stack(all_matrices, axis=0), axis=0)
        global_std = np.std(np.stack(all_matrices, axis=0), axis=0)
        np.save(output_dir / "importance_matrix_avg.npy", global_avg)
        np.save(output_dir / "importance_matrix_std.npy", global_std)
        log.info(
            "GNNExplainer complete — global importance matrix saved to %s  "
            "(N=%d, max=%.4f, mean=%.4f).",
            output_dir,
            global_avg.shape[0],
            float(global_avg.max()),
            float(global_avg.mean()),
        )
        self._log_top_edges(global_avg)
        return output_dir

    # ------------------------------------------------------------------
    # Per-fold worker
    # ------------------------------------------------------------------

    def _explain_one_fold(
        self,
        *,
        fold_ckpt_root: Path,
        fold_idx: int,
        fold_ckpt_dir: Path,
        fold_out_dir: Path,
        train_idx: List[int],
        test_idx: List[int],
        dataset: List[torch_geometric.data.Data],
        model_factory: Callable[[], torch.nn.Module],
        device: torch.device,
        resolved_edge_size: float,
        glm_col_range: Optional[Tuple[int, int]],
        glm_normalize: bool,
        fold_log_tag: str,
    ) -> Optional[np.ndarray]:
        """Explain one outer fold; return the per-fold mean importance matrix.

        ``fold_ckpt_root`` is the directory whose ``CheckpointManager`` view
        treats ``fold_<fold_idx>`` as a child — i.e. ``ckpt_root`` in the
        legacy layout and ``ckpt_root/rep_<R>`` in the nested layout.
        """
        from torch_geometric.explain import Explainer, GNNExplainer
        from torch_geometric.explain.config import ModelConfig as ExplainerModelConfig

        from src.training.fold_checkpoint import CheckpointManager
        from src.training.glm_normalizer import GLMFeatureNormalizer

        ckpt_path = fold_ckpt_dir / "model_best.pt"
        if not ckpt_path.exists():
            log.warning(
                "Checkpoint not found: %s — skipping %s.",
                ckpt_path, fold_log_tag,
            )
            return None

        num_nodes = dataset[0].num_nodes if dataset else 0
        model, _ = CheckpointManager(fold_ckpt_root).load_model_for_fold(
            fold_idx, model_factory, num_nodes, variant="best",
        )
        model.eval()
        model.to(device)
        log.debug("Loaded model weights from %s", ckpt_path)

        test_graphs = [copy.copy(dataset[i]) for i in test_idx]

        if glm_col_range is not None and glm_normalize:
            col_start, col_end = glm_col_range
            glm_norm = GLMFeatureNormalizer(col_start, col_end)
            train_graphs_for_norm = [copy.copy(dataset[i]) for i in train_idx]
            glm_norm.fit(train_graphs_for_norm)
            glm_norm.transform(test_graphs)
            log.debug(
                "%s: GLM features (cols [%d:%d)) z-scored on %d train "
                "subjects and applied to %d test subjects.",
                fold_log_tag, col_start, col_end, len(train_idx), len(test_idx),
            )

        wrapped_model = _DataWrapper(model)
        explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(
                epochs=self.cfg.epochs,
                lr=self.cfg.lr,
                coeffs={
                    "edge_size": resolved_edge_size,
                    "edge_ent": self.cfg.edge_ent,
                    "node_feat_size": self.cfg.node_feat_size,
                    "node_feat_ent": self.cfg.node_feat_ent,
                },
            ),
            explanation_type="model",
            edge_mask_type="object",
            node_mask_type=self.cfg.node_mask_type,
            model_config=ExplainerModelConfig(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )

        fold_matrices: List[np.ndarray] = []
        if self.cfg.save_subject_masks:
            subjects_dir = fold_out_dir / "subjects"
            subjects_dir.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(style="cyan"),
            MofNCompleteColumn(),
            BarColumn(bar_width=28, style="cyan", complete_style="bold cyan"),
            TextColumn("[dim]{task.description}[/]"),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "explaining subjects", total=len(test_graphs),
            )
            for graph_local_idx, g in enumerate(test_graphs):
                subject_id = getattr(
                    g, "subject_id", f"subject_{graph_local_idx:04d}",
                )
                progress.update(task, description=f"subject [bold]{subject_id}[/]")

                x = g.x.to(device)
                edge_index = g.edge_index.to(device)
                edge_attr = g.edge_attr.to(device) if g.edge_attr is not None else None
                num_nodes = g.num_nodes

                kwargs = {}
                if edge_attr is not None:
                    kwargs["edge_attr"] = edge_attr

                try:
                    explanation = explainer(x=x, edge_index=edge_index, **kwargs)
                except Exception as exc:
                    log.warning(
                        "%s subject %s: GNNExplainer failed (%s). Skipping.",
                        fold_log_tag, subject_id, exc,
                    )
                    progress.advance(task)
                    continue

                edge_mask = explanation.edge_mask.detach().cpu()
                if self.cfg.save_subject_masks:
                    mask_path = subjects_dir / f"{subject_id}_edge_mask.pt"
                    torch.save(edge_mask, mask_path)

                importance_mat = _edge_mask_to_importance_matrix(
                    edge_index.cpu(), edge_mask, num_nodes,
                )
                fold_matrices.append(importance_mat)
                progress.advance(task)

        if not fold_matrices:
            log.warning("%s: no valid explanations produced.", fold_log_tag)
            return None

        fold_avg = np.mean(np.stack(fold_matrices, axis=0), axis=0)
        fold_std = np.std(np.stack(fold_matrices, axis=0), axis=0)
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(fold_out_dir / "importance_matrix.npy", fold_avg)
        np.save(fold_out_dir / "importance_matrix_std.npy", fold_std)
        log.info(
            "%s: average importance matrix saved (N=%d, max=%.4f).",
            fold_log_tag, fold_avg.shape[0], float(fold_avg.max()),
        )
        return fold_avg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_top_edges(
        self, importance_matrix: np.ndarray, top_k: int = 10
    ) -> None:
        """Log the *top_k* most important brain connections to the console."""
        n = importance_matrix.shape[0]
        # Upper-triangle indices (exclude diagonal)
        triu_idx = np.triu_indices(n, k=1)
        vals = importance_matrix[triu_idx]
        sorted_order = np.argsort(vals)[::-1][:top_k]

        log.info("Top-%d most important brain connections (global average):", top_k)
        for rank, flat_k in enumerate(sorted_order, start=1):
            u = triu_idx[0][flat_k]
            v = triu_idx[1][flat_k]
            score = vals[flat_k]
            log.info("  #%2d  node %3d ↔ node %3d   importance = %.5f", rank, u, v, score)

