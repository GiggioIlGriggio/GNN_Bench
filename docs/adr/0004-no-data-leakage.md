# ADR-0004: All normalisation fit on training subjects only (no data leakage)

**Status**: Accepted  
**Date**: 2025-03

## Context

Brain connectivity studies often have small N (tens to low hundreds of subjects). Data leakage from the test set into any normalisation step produces optimistically biased metrics that do not generalise. Three leakage risks were identified:

1. **Label normalisation**: fitting a z-score scaler on all subjects before splitting.
2. **GLM feature normalisation**: z-scoring per-node GLM columns across all subjects.
3. **Composite label construction**: fitting PCA or computing column statistics on all subjects when deriving a composite target.

## Decision

All three are handled per-fold inside `CrossValidator.run()`:

- **`LabelNormalizer`**: `fit_transform(y_train)` → `transform(y_val)`, `transform(y_test)`.
- **`GLMFeatureNormalizer`**: `fit(train_graphs)` → `transform(val_graphs)`, `transform(test_graphs)`. Applied only when `FeatureConfig.glm_normalize=true` and `glm_contrasts` is non-empty.
- **`LabelBuilder` (composite)**: `fit_transform(label_components.iloc[train_idx])` → `transform(label_components.iloc[val_idx])`, `transform(label_components.iloc[test_idx])`. This also prevents PCA from seeing test subject scores.

The `get_labels()` shortcut on `BrainGraphDataset` uses the one-shot `LabelBuilder.build()` path, which is fine for stratification binning (full labels needed) but **must not** be used as fold labels when composite targets are configured.

## Consequences

- Pooled test metrics (aggregated across folds) are unbiased estimates of generalisation performance.
- `CrossValidator.run()` raises `ValueError` if exactly one of `label_builder`/`label_components` is provided — both or neither must be passed to prevent accidental leakage.
- `LabelNormalizer` and `GLMFeatureNormalizer` are serialised alongside the model checkpoint so that inference at test time can invert the normalisation correctly.
