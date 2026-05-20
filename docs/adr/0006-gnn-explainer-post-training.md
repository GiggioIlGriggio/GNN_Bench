# ADR-0006: GNNExplainer runs post-training on saved checkpoints

**Status**: Accepted  
**Date**: 2025-04

## Context

GNNExplainer is computationally expensive (it optimises edge masks per subject). Running it inside the training loop would slow down sweep trials and make checkpointing more complex. We also only want explanations for the best model, not every trial.

## Decision

`GNNExplainerRunner` (`src/gnn_explainer/explainer.py`) is a separate post-training step:

- It loads saved fold checkpoints from disk (reads `model_config.json` + `feature_config.json` alongside the state dict to reconstruct the exact architecture).
- It runs `torch_geometric.explain.GNNExplainer` on each fold's test subjects.
- Outputs a per-fold `importance_matrix.npy` (`[N, N]` symmetric, averaged over test subjects) and optionally per-subject `edge_mask.pt` tensors.
- Aggregates across folds: `importance_matrix_avg.npy` + `importance_matrix_std.npy`.

In sweep mode, `GNNExplainerRunner` is only invoked for the **current best trial** (tracked by `HydraSweep`). When a better trial is found later, it overwrites the previous explainer outputs.

The `edge_size` regularisation coefficient must be tuned per modality (see docstring rule of thumb: `edge_size ≈ 1 / avg_degree`).

## Consequences

- Explainer outputs are always tied to the best-performing model, not an arbitrary trial.
- Re-running the explainer on an existing checkpoint directory does not require re-training.
- The model architecture must be reconstructable from the saved JSON files — any change to `ModelConfig` or `FeatureConfig` that affects architecture must also update the checkpoint JSON.
- Controlled by `ExplainerConfig.enabled` (default `false`). Disabled by default so normal runs are not slowed down.
