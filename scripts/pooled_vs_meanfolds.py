#!/usr/bin/env python
"""Recompute pooled vs mean-of-folds regression metrics from nested_cv_result.json.

Why this exists
---------------
The nested-CV pipeline reports **mean-of-folds** metrics
(``NestedCVResult.mean_metrics``): the average of each per-outer-fold R²
(``src/training/nested_cross_validation.py:_aggregate``). A flat-CV protocol such
as ``EF_neural_substrate`` instead reports **pooled** R²: concatenate every
out-of-fold prediction into one vector and compute a single R² against the global
label mean.

These two estimators of the *same* predictions diverge sharply when per-fold R²
has high variance (small outer-test folds + inner-HPO instability): mean-of-folds
is dragged down by the worst folds, while pooling rewards the shared signal.
Quantifying that gap is the whole point of the identity→VWM R² investigation.

This tool reads the OOF predictions the pipeline already persists
(``fold_results[].y_true`` / ``y_pred``) and reports BOTH flavours using the
project's exact R² formula (``src.training.metrics.compute_metrics`` =
coefficient of determination, ``metrics.py:62-64``) so the numbers are directly
comparable to anything the pipeline logs.

Usage
-----
    PYTHONPATH=$(pwd) .venv/bin/python scripts/pooled_vs_meanfolds.py \
        path/to/nested_cv_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.metrics import compute_metrics  # noqa: E402


def _concat(folds, key):
    return np.concatenate([np.asarray(fr[key], dtype=float) for fr in folds])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path, help="path to a nested_cv_result.json")
    args = ap.parse_args()

    with open(args.result) as f:
        data = json.load(f)

    folds = data["fold_results"]
    if not folds:
        raise SystemExit(f"No fold_results in {args.result}")

    n_rep = data.get("n_repetitions")
    n_outer = data.get("n_outer_folds")
    inner = data.get("inner_hpo_trials")
    hpo_metric = data.get("hpo_metric")

    # --- mean-of-folds (the pipeline's reported number) -------------------
    per_fold_r2 = [fr["outer_test_metrics"]["r2"] for fr in folds]
    mof_mean = data["mean_metrics"]["r2"]
    mof_std = data["std_metrics"]["r2"]

    # --- pooled over ALL folds (concat every OOF prediction) --------------
    yt_all = _concat(folds, "y_true")
    yp_all = _concat(folds, "y_pred")
    pooled_all = compute_metrics(yt_all, yp_all)

    # --- per-repetition pooled (most apples-to-apples with flat-5fold) ----
    # Within one repetition every subject is OOF exactly once, so pooling the
    # 5 outer folds reproduces a flat-5fold pooled R² over the full N subjects.
    reps = sorted({fr["rep"] for fr in folds})
    per_rep_pooled = []
    for rep in reps:
        frs = [fr for fr in folds if fr["rep"] == rep]
        yt = _concat(frs, "y_true")
        yp = _concat(frs, "y_pred")
        per_rep_pooled.append(compute_metrics(yt, yp)["r2"])
    per_rep_pooled = np.asarray(per_rep_pooled, dtype=float)

    # --- report -----------------------------------------------------------
    print(f"file              : {args.result}")
    print(f"run_name          : {data.get('run_name')}")
    print(f"model             : {data.get('model_name')}")
    print(
        f"protocol          : n_repetitions={n_rep}  n_outer_folds={n_outer}  "
        f"inner_hpo_trials={inner}  hpo_metric={hpo_metric}"
    )
    print(f"total OOF preds   : {yt_all.size}  ({len(folds)} fold-results)")
    print()
    print("R² estimators (same predictions, different aggregation)")
    print("-------------------------------------------------------")
    print(f"  mean-of-folds     : {mof_mean:+.4f}  ± {mof_std:.4f}  (pipeline's reported R²)")
    print(f"  pooled (all folds): {pooled_all['r2']:+.4f}  over {yt_all.size} preds")
    if len(reps) > 1:
        print(
            f"  pooled (per-rep)  : {per_rep_pooled.mean():+.4f}  ± "
            f"{per_rep_pooled.std(ddof=1):.4f}  (mean of {len(reps)} per-rep pooled R²)"
        )
    print()
    print("Per-fold test R²")
    print("----------------")
    for fr, r2 in zip(folds, per_fold_r2):
        print(f"  rep {fr['rep']} fold {fr['fold']}: {r2:+.4f}  (n_test={fr.get('n_test')})")
    print()
    print("Other pooled metrics (all folds)")
    print("--------------------------------")
    for k in ("mae", "rmse", "pearson_r"):
        print(f"  {k:10s}: {pooled_all[k]:.4f}")


if __name__ == "__main__":
    main()
