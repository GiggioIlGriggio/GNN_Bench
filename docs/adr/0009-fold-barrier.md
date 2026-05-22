# ADR-0009: Per-fold leakage protections coordinate behind a Fold barrier

**Status**: Accepted
**Date**: 2026-05
**Supplements**: [ADR-0004](./0004-no-data-leakage.md), [ADR-0008](./0008-nested-cross-validation.md)

## Context

ADR-0004 identifies three per-fold leakage risks — label normalisation, GLM feature normalisation, and composite-label construction — and mandates that all three be fit on the outer-train pool only and applied to val / test / refit splits.

The first implementation kept the three protections as independent classes (`LabelNormalizer`, `GLMFeatureNormalizer`, `LabelBuilder`), each fit and threaded by hand inside the cross-validator. The two CV implementations (`NestedCrossValidator._make_split_loaders` and the legacy `CrossValidator.run()`) reimplement the same five-step orchestration almost verbatim, and downstream consumers (`GNNExplainerRunner`, `Finetuner`) reach into individual transformers when they reload a checkpoint. The most concrete cost: `GLMFeatureNormalizer`'s fitted state was never persisted, so the Explainer re-fits it from the recovered training indices every time, which couples explanation reproducibility to upstream split derivation rather than to the saved checkpoint.

Three axes of growing friction:

1. Adding a fourth per-fold-fit transformer would require edits in three places (the dataset loader, both CV classes, the consumer that reloads checkpoints).
2. `LabelNormalizer` was pickled to `normalizer.pkl`; `GLMFeatureNormalizer` was not persisted at all; `LabelBuilder`'s composite-label state had its own pickle path. Three different on-disk shapes for one conceptual cluster, with one of them missing.
3. The `Trainer` accepted a pre-fitted `LabelNormalizer` parameter without any guard that it had been fitted — the "fitted-on-train" invariant lived in the orchestrating CV class, not in a named module.

## Decision

The three per-fold leakage protections are unified behind a single `FoldBarrier` module (`src/training/fold_barrier.py`). The barrier owns the composite-label `LabelBuilder` (when configured), the `LabelNormalizer`, and the `GLMFeatureNormalizer` as one fitted bundle per outer fold.

**Lifecycle and interface.**

- `fit(train_graphs, train_labels_or_components)` — fits all three transformers on the outer-train pool exactly once. The second argument is a 1-D label vector when composite labels are not configured, or a `pd.DataFrame` slice of `label_components` when they are.
- `transform_graphs(graphs) → List[Data]` — returns new graphs whose `x` tensor has been GLM-normalised on a cloned slice. Inputs are never mutated and the returned graphs never share storage with the input dataset.
- `transform_labels(labels_or_components) → np.ndarray` — applies composite construction (if configured) followed by z-scoring.
- `inverse_transform_labels(y_norm) → np.ndarray` — denormalises predictions back to the original target scale.

The barrier is a stateful "transform-anything-consistent-with-my-fit" object. It does not know about splits, batch sizes, or DataLoader construction. Callers that need split-specific behaviour (inner train vs. inner val vs. outer test) make multiple barrier calls; callers that need batching wrap the resulting graphs themselves.

**Serialisation.** A barrier persists as a typed state-dict at `barrier.pt` inside the fold's [Checkpoint](../../CONTEXT.md#checkpoint-layout) directory. Each transformer exposes `state_dict() -> dict[str, Any]` and `load_state_dict(state)`; the barrier aggregates them under one dict and writes one file via `torch.save`. `pickle` is not used — the format survives class renames and partial-write detection is trivial.

**Trainer coupling.** The `Trainer` is *not* a consumer of `FoldBarrier`. It accepts an `inverse_transform: Callable[[np.ndarray], np.ndarray]` parameter for denormalising predictions; callers pass `barrier.inverse_transform_labels`. The Trainer's interface stays narrow and the barrier's surface can evolve without disturbing the training loop.

**Both CV paths consume the barrier.** The canonical `NestedCrossValidator` and the legacy `CrossValidator` both delegate to `FoldBarrier`; the duplicated five-step dance is removed from both. The legacy `HydraSweep` retention decision (ADR-0008) is unchanged — the same module is shared, not retired.

## Considered alternatives

- **Keep the three transformers separate; introduce only a thin helper function for "fit + transform + persist".** Cheaper change but it leaves the lifecycle invariant as prose in ADR-0004 rather than as a module. The Explainer's re-fit path persists; consumers continue reaching into individual transformers. Rejected.
- **Bundle the three transformers and also have the barrier build DataLoaders.** Closer to today's `_make_split_loaders`. Couples the leakage-protection concept to PyG batching policy; the barrier ends up importing `TrainerConfig.batch_size` indirectly. Rejected — a "stateful transform-anything" barrier is the cleaner split of concerns.
- **Keep `LabelBuilder` outside the barrier on the grounds that its fit input (a `pd.DataFrame` slice) is different from the others'.** A real consideration. Rejected because the shared lifecycle outweighs the per-fit-input-shape difference, the persisted barrier becomes truly self-sufficient only when composite-label state is inside it, and the input-shape difference is localised to one named argument of `fit()`.
- **Have the `Trainer` consume `FoldBarrier` directly.** Cleaner single named seam, but it pulls a leakage-protection abstraction into a strictly training-loop module for one method (`inverse_transform_labels`). Rejected — the callable parameter is the narrowest interface that does the job.
- **Preserve `normalizer.pkl` for backwards compatibility and add `barrier.pt` alongside.** Two formats in flight for the duration of the transition; load-time disambiguation logic. Rejected — old checkpoints' model weights remain loadable, but full fold reload requires re-running. Cheaper than carrying two formats.

## Consequences

- A persisted `barrier.pt` is the source of truth for what was fit on a given outer fold's train pool. `GNNExplainerRunner`, `Finetuner`, and any future "evaluate a saved fold on a new cohort" tool load the barrier and call `transform_graphs` / `transform_labels` without ever consulting the original train indices. Explanation reproducibility decouples from upstream split derivation.
- Adding a fourth per-fold-fit transformer (e.g. a new node-feature normaliser) is a single-module change inside `FoldBarrier`. Both CV paths pick it up for free.
- Existing checkpoints with `normalizer.pkl` and no `barrier.pt` cannot be reloaded as full fold artifacts. Their model weights remain loadable; B-fresh assumes any run whose fitted leakage state matters is cheaper to re-run than to migrate.
- The contract of the GLM step changes from "the caller copies graphs first; we mutate in place" to "we return new graphs". A latent fragility of the previous code (shallow-copied PyG `Data` objects sharing tensor storage with the original dataset) is closed at the seam rather than relied upon at the call site.
- ADR-0004's three-leakage-protections framing is unchanged conceptually; the implementation now matches that framing as a single coordinator instead of three transformers plus duplicated orchestration. ADR-0008's per-fold checkpoint layout gains one file (`barrier.pt`) and loses one file (`normalizer.pkl`).
