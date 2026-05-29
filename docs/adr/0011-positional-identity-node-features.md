# ADR-0011: Laplacian PE and ID-GNN-Fast cycle-count node features

## Status
Accepted — 2026-05-28

## Context
We add two topology-derived initial node features:
- `laplacian_pe`: k eigenvectors of the smallest non-zero eigenvalues of the
  symmetric normalized Laplacian (ported in spirit from benchmarking-gnns).
- `cycle_counts`: ID-GNN-Fast identity features, `log1p([A^l]_vv)` for l=2..L.

Both are defined on a **binarized** adjacency, but brain graphs are weighted and
may be signed (FC) or fully dense (no thresholding).

## Decision
1. **Binarize locally.** Each feature builds a dense 0/1 symmetric adjacency
   (no self-loops) from `_get_primary_edges` solely for its own computation.
   The graph's weighted `edge_index`/`edge_attr` are unchanged everywhere else.
2. **Strictness.** `laplacian_pe` raises on a fully-dense or disconnected graph
   (disconnection detected via zero-eigenvalue multiplicity). `cycle_counts`
   raises only on the fully-dense case.
3. **Eigenvector sign ambiguity → train-time augmentation.** LapPE eigenvector
   signs are arbitrary. Because features are baked into `data.x` once at build
   time and there is no per-epoch transform hook, the model is made
   sign-invariant by randomly flipping the LapPE columns of `batch.x` per graph
   each training step inside `Trainer._train_one_epoch`. Eval paths
   (`predict`/`evaluate`) never flip, so validation/test are deterministic.
   This deliberately crosses the feature→training boundary; the column range is
   located via a single-source-of-truth offset map recorded by
   `build_node_features` and threaded into `TrainerConfig.sign_flip_cols`.
4. **Column bookkeeping.** `build_node_features` records each feature's column
   offsets from actual tensor widths; `get_glm_column_range` and the new
   `get_laplacian_pe_column_range` read this map (replacing bespoke arithmetic).

## Consequences
- The "train-only" invariant is covered by a Trainer test, but remains a sharp
  edge: any future refactor that moves forward passes out of `_train_one_epoch`
  must preserve it, or validation will be silently augmented.
- Strict raising means a subject whose thresholded graph fragments fails the run;
  thresholding must reliably yield a single connected component.
- Cost: one dense `numpy.linalg.eigh` per subject at build time (~ms at N=400).
