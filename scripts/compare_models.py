#!/usr/bin/env python
"""Cross-model statistical comparison from saved NestedCVResult JSONs.

Usage::

    python scripts/compare_models.py \\
        --inputs path/to/runA/nested_cv_result.json path/to/runB/...json [...] \\
        --metric mae \\
        --output-dir reports/comparison/

For each pair of input runs the script runs the Bouckaert-Frank corrected
resampled paired t-test (see :mod:`src.training.statistical_tests`) over
per-outer-fold scores of the chosen ``--metric``. Pairwise raw and
Benjamini-Hochberg-adjusted p-values are written as CSV matrices plus a
markdown summary that ranks the runs by mean of the chosen metric.

The script does not need any model code — it consumes only the JSON
artifacts produced by ``NestedCVResult.save``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.training.nested_cross_validation import NestedCVResult
from src.training.statistical_tests import (
    PairwiseComparison,
    pairwise_compare,
    to_pvalue_matrix,
)

log = logging.getLogger(__name__)

_METRIC_DIRECTION = {
    "mae": "lower",
    "rmse": "lower",
    "r2": "higher",
    "pearson_r": "higher",
}


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to two or more nested_cv_result.json files.",
    )
    p.add_argument(
        "--metric",
        default="mae",
        choices=sorted(_METRIC_DIRECTION),
        help="Outer-test metric to compare on (default: mae).",
    )
    p.add_argument(
        "--output-dir",
        default="reports/comparison",
        help="Directory where CSV matrices and the markdown summary are written.",
    )
    p.add_argument(
        "--names",
        nargs="+",
        default=None,
        help=(
            "Optional display names for each input run, in the same order "
            "as --inputs. Defaults to NestedCVResult.run_name."
        ),
    )
    return p.parse_args(argv)


def _load_inputs(paths: List[Path], names: List[str] | None) -> Dict[str, NestedCVResult]:
    runs: Dict[str, NestedCVResult] = {}
    if names is not None and len(names) != len(paths):
        raise ValueError(
            f"--names ({len(names)}) and --inputs ({len(paths)}) must match in length"
        )
    for i, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(f"NestedCVResult not found: {path}")
        result = NestedCVResult.load(path)
        name = names[i] if names is not None else result.run_name
        if name in runs:
            raise ValueError(
                f"Duplicate run name {name!r}; pass --names to disambiguate."
            )
        runs[name] = result
    return runs


def _validate_consistency(runs: Dict[str, NestedCVResult]) -> None:
    expected = next(iter(runs.values()))
    for name, r in runs.items():
        if r.n_outer_folds != expected.n_outer_folds:
            raise ValueError(
                f"{name} has n_outer_folds={r.n_outer_folds} but expected "
                f"{expected.n_outer_folds} (from {next(iter(runs))})"
            )
        if r.n_repetitions != expected.n_repetitions:
            raise ValueError(
                f"{name} has n_repetitions={r.n_repetitions} but expected "
                f"{expected.n_repetitions}"
            )
        if len(r.fold_results) != len(expected.fold_results):
            raise ValueError(
                f"{name} has {len(r.fold_results)} fold results but expected "
                f"{len(expected.fold_results)}"
            )


def _write_matrix_csv(
    path: Path, names: List[str], mat: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("model," + ",".join(names) + "\n")
        for i, name in enumerate(names):
            row = ",".join(
                "" if np.isnan(mat[i, j]) else f"{mat[i, j]:.6g}"
                for j in range(len(names))
            )
            f.write(f"{name},{row}\n")


def _write_markdown(
    path: Path,
    runs: Dict[str, NestedCVResult],
    metric: str,
    comparisons: List[PairwiseComparison],
) -> None:
    direction = _METRIC_DIRECTION[metric]
    ranked = sorted(
        runs.items(),
        key=lambda kv: kv[1].mean_metrics[metric],
        reverse=(direction == "higher"),
    )

    lines: List[str] = []
    lines.append(f"# Cross-model comparison — metric: `{metric}` ({direction} is better)\n")
    lines.append("## Ranking\n")
    lines.append("| Rank | Run | Mean | Std |\n|------|-----|------|-----|\n")
    for i, (name, r) in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {name} | {r.mean_metrics[metric]:.4f} | "
            f"{r.std_metrics[metric]:.4f} |\n"
        )

    lines.append("\n## Pairwise Bouckaert-Frank corrected t-test\n")
    lines.append(
        "Two-sided p-values. `p_adj` is Benjamini-Hochberg-adjusted across "
        "all pairwise comparisons. `mean_diff` is `A - B`.\n"
    )
    lines.append(
        "| A | B | t | p_raw | p_adj | mean_diff | sig (q<0.05) |\n"
        "|---|---|---|-------|-------|-----------|--------------|\n"
    )
    for c in comparisons:
        sig = "**yes**" if c.p_adjusted < 0.05 else "no"
        lines.append(
            f"| {c.model_a} | {c.model_b} | {c.t_stat:+.3f} | "
            f"{c.p_raw:.4g} | {c.p_adjusted:.4g} | {c.mean_diff:+.4f} | {sig} |\n"
        )

    lines.append("\n## Protocol\n")
    sample = next(iter(runs.values()))
    lines.append(
        f"- Repetitions: {sample.n_repetitions}\n"
        f"- Outer folds: {sample.n_outer_folds}\n"
        f"- Inner HPO trials: {sample.inner_hpo_trials}\n"
        f"- HPO metric: {sample.hpo_metric}\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if len(args.inputs) < 2:
        raise SystemExit("Need at least two --inputs to compare")

    runs = _load_inputs([Path(p) for p in args.inputs], args.names)
    _validate_consistency(runs)

    scores = {name: r.per_fold_scores(args.metric) for name, r in runs.items()}
    n_outer = next(iter(runs.values())).n_outer_folds
    comparisons = pairwise_compare(scores, n_outer_folds=n_outer)

    names_sorted, raw_mat = to_pvalue_matrix(comparisons, use_adjusted=False)
    _, adj_mat = to_pvalue_matrix(comparisons, use_adjusted=True)

    out_dir = Path(args.output_dir)
    _write_matrix_csv(out_dir / f"{args.metric}_pvalues_raw.csv", names_sorted, raw_mat)
    _write_matrix_csv(out_dir / f"{args.metric}_pvalues_bh.csv", names_sorted, adj_mat)
    _write_markdown(out_dir / f"{args.metric}_summary.md", runs, args.metric, comparisons)

    log.info("Wrote comparison artifacts to %s", out_dir.resolve())
    for c in comparisons:
        log.info(
            "  %s vs %s: t=%+.3f  p_raw=%.4g  p_adj=%.4g  mean_diff=%+.4f",
            c.model_a, c.model_b, c.t_stat, c.p_raw, c.p_adjusted, c.mean_diff,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
