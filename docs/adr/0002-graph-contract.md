# ADR-0002: Strict PyG Data graph contract

**Status**: Accepted  
**Date**: 2025-03

## Context

Multiple dataset implementations produce PyG `Data` objects consumed by multiple model implementations. Without a contract, each model would need to defensively check tensor shapes and dtypes, and subtle shape bugs (e.g. `edge_attr` being 1-D instead of 2-D) would only surface as cryptic CUDA errors deep in GNN layers.

## Decision

Every `Data` object produced by `BrainGraphDataset.build_graph` must satisfy the following contract, enforced by `validate_graph_contract()` in `src/datasets/base_dataset.py`:

| Attribute | Shape | Dtype |
|---|---|---|
| `x` | `[num_nodes, node_feat_dim]` | `float32` |
| `edge_index` | `[2, num_edges]` | `int64` |
| `edge_attr` | `[num_edges, edge_feat_dim]` | `float32` |
| `y` | `[1]` | `float32` |

Additionally, every graph must carry `data.subject_id: str` so cross-validation can recover subject identifiers from integer indices.

`build_graph` calls `validate_graph_contract()` before returning and raises `ValueError` on violation.

## Consequences

- Models can assume the contract holds — no defensive shape checks needed in forward passes.
- New dataset implementations will fail fast with a clear error if they produce malformed graphs.
- Composite-label targets use a placeholder `y = tensor([0.0])` at graph-build time; the real per-fold label is attached by `CrossValidator._make_loader`. This is intentional — label assignment is the CV layer's responsibility.
- The contract does **not** constrain `num_nodes` or `num_edges` — those vary by subject and thresholding.
