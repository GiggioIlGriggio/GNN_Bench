import json
import numpy as np
from src.training.splits import stratify_bins, outer_folds


def test_same_bins_same_folds_diff_bins_diff_folds():
    n, seeds, n_outer = 60, [0, 1], 5
    rng = np.random.default_rng(0)
    age = rng.normal(size=n)
    vwm = rng.normal(size=n)
    bins_age = stratify_bins(age, 4)
    bins_vwm = stratify_bins(vwm, 4)
    folds_age = outer_folds(n=n, bins=bins_age, seeds=seeds, n_outer=n_outer)
    folds_vwm = outer_folds(n=n, bins=bins_vwm, seeds=seeds, n_outer=n_outer)
    folds_vwm2 = outer_folds(n=n, bins=bins_vwm, seeds=seeds, n_outer=n_outer)
    # outer_folds returns list of (train_val_idx, test_idx) tuples where each
    # idx is a plain list (see splits.py: train_val_idx.tolist()), so == works.
    # identical bins -> identical folds (determinism)
    assert [t for t, _ in folds_vwm] == [t for t, _ in folds_vwm2]
    # different bins -> at least one fold differs (proves the override is necessary)
    assert [s for _, s in folds_age] != [s for _, s in folds_vwm]


def test_manifest_roundtrip_shape(tmp_path):
    # minimal manifest contract used by SourceBackboneProvider
    manifest = {
        "folds": [
            {"rep": 0, "fold": 0, "train_val_idx": [1, 2], "test_idx": [0]},
        ]
    }
    p = tmp_path / "fold_indices.json"
    p.write_text(json.dumps(manifest))
    loaded = json.loads(p.read_text())
    rec = loaded["folds"][0]
    assert (rec["rep"], rec["fold"]) == (0, 0)
    assert rec["train_val_idx"] == [1, 2]
