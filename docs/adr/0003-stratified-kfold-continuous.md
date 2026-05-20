# ADR-0003: Stratified K-fold with quantile binning for continuous labels

**Status**: Accepted  
**Date**: 2025-03

## Context

Standard `StratifiedKFold` requires discrete class labels. Our regression targets are continuous (age, cognitive scores). A random split risks placing all high-scoring or low-scoring subjects in one fold, producing unrepresentative train/test distributions and inflated variance in cross-fold metrics.

## Decision

`CrossValidator.split()` bins continuous labels into `cfg.stratify_bins` quantile buckets using `pd.qcut(..., duplicates="drop")` before passing to `sklearn.StratifiedKFold`. The bins are computed from the full label array (not per fold) for consistent stratification across folds.

Within each fold, the train+val pool from the outer `StratifiedKFold` is further split into train and val by random shuffle (seeded by `cfg.seed + fold_idx`).

## Consequences

- Each fold's test set has approximately the same label distribution as the overall dataset.
- `cfg.stratify_bins` defaults to 5. Setting it too high (more bins than subjects per bin) triggers scikit-learn warnings; `duplicates="drop"` handles degenerate cases gracefully.
- The seed is set per fold (`cfg.seed + fold_idx`) for reproducibility while ensuring different val splits across folds.
- The stratification bins are computed on the **full** labels array, not per-fold labels, so they are consistent even in composite-label mode where per-fold labels differ.
