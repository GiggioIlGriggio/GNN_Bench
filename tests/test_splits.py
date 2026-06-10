import numpy as np
from src.training.splits import stratify_bins, outer_folds, inner_split


def _labels(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n)


def test_stratify_bins_matches_qcut():
    import pandas as pd
    labels = _labels()
    got = stratify_bins(labels, n_bins=5)
    exp = pd.qcut(labels, q=5, labels=False, duplicates="drop")
    assert np.array_equal(got, exp)


def test_outer_folds_deterministic_and_partition():
    labels = _labels()
    bins = stratify_bins(labels, n_bins=5)
    folds_a = outer_folds(n=len(labels), bins=bins, seeds=[42, 43], n_outer=5)
    folds_b = outer_folds(n=len(labels), bins=bins, seeds=[42, 43], n_outer=5)
    assert folds_a == folds_b
    assert len(folds_a) == 10
    for rep in range(2):
        rep_folds = folds_a[rep * 5:(rep + 1) * 5]
        test_union = sorted(i for _, te in rep_folds for i in te)
        assert test_union == list(range(len(labels)))


def test_inner_split_is_subset_and_3to1():
    labels = _labels()
    bins = stratify_bins(labels, n_bins=5)
    train_val = list(range(0, 100))
    tr, va = inner_split(train_val, bins, inner_seed=42 * 1000 + 0)
    assert set(tr).issubset(train_val) and set(va).issubset(train_val)
    assert set(tr).isdisjoint(va)
    assert len(tr) + len(va) == len(train_val)
    assert 2.0 < len(tr) / len(va) < 4.0
