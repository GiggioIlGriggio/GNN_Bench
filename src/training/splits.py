"""Stratified fold-index generation shared by the GNN and sklearn nested-CV runners.

Extracted verbatim from NestedCrossValidator so BOTH runners produce
byte-identical folds for the same labels + seeds (the basis for
baseline-vs-GNN comparability).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def stratify_bins(labels: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin continuous labels for stratified splitting."""
    return pd.qcut(labels, q=n_bins, labels=False, duplicates="drop")


def outer_folds(
    *, n: int, bins: np.ndarray, seeds: List[int], n_outer: int,
) -> List[Tuple[List[int], List[int]]]:
    """Enumerate (train_val_idx, test_idx) for every (rep, fold), rep-major order.

    Mirrors NestedCrossValidator.run: one StratifiedKFold per outer seed.
    """
    out: List[Tuple[List[int], List[int]]] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for train_val_idx, test_idx in skf.split(np.arange(n), bins):
            out.append((train_val_idx.tolist(), test_idx.tolist()))
    return out


def inner_split(
    train_val_idx: List[int], bins: np.ndarray, inner_seed: int,
) -> Tuple[List[int], List[int]]:
    """Stratified 4-way split → first fold's 3:1 train:val (matches ADR-0003)."""
    idx_arr = np.asarray(train_val_idx)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=inner_seed)
    first_train, first_val = next(skf.split(idx_arr, bins[idx_arr]))
    return idx_arr[first_train].tolist(), idx_arr[first_val].tolist()
