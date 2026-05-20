"""Cross-model statistical tests for repeated nested CV (ADR-0008).

Two pure functions live here so they can be reused outside any specific
training run:

* :func:`corrected_resampled_paired_t_test` — Bouckaert & Frank (2004)
  / Nadeau & Bengio (2003) correction for the dependence between repeated
  k-fold CV scores. Returns ``(t_stat, p_two_sided, df, mean_diff)``.

* :func:`benjamini_hochberg` — step-up adjusted p-values for a family of
  pairwise comparisons, controlling FDR at the user's chosen level.

Both work on per-outer-fold metric arrays. The caller supplies any
metric (lower-is-better or higher-is-better) — the corrected t-test only
inspects the sign of the mean difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Corrected resampled paired t-test
# ---------------------------------------------------------------------------


def corrected_resampled_paired_t_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_outer_folds: int,
) -> Tuple[float, float, int, float]:
    """Bouckaert–Frank corrected paired t-test for repeated k-fold CV.

    Standard paired t-tests over CV scores are anti-conservative because
    successive train sets overlap heavily. The correction inflates the
    variance estimator by ``n_test / n_train``, which for a balanced
    k-fold split is ``1 / (k - 1)``. The same formula is referred to as
    the *Bouckaert–Frank* test in Mhanna et al. (MIDL 2026) and as the
    *Nadeau–Bengio corrected resampled t-test* elsewhere.

    Parameters
    ----------
    scores_a, scores_b : Sequence[float]
        Per-outer-fold scores of the two models being compared. Must have
        the same length ``n = n_repetitions * n_outer_folds``.
    n_outer_folds : int
        ``k`` in the k-fold protocol. Used to derive the correction term
        ``1 / (k - 1)``.

    Returns
    -------
    Tuple[float, float, int, float]
        ``(t_stat, p_two_sided, df, mean_diff)``.

    Notes
    -----
    The two-sided p-value is computed from the Student's-t survival
    function. ``df = n - 1``. When ``var(d) == 0`` the t-statistic is
    undefined; we return ``t = 0.0`` and ``p = 1.0`` (no evidence either
    way) in that degenerate case.

    Raises
    ------
    ValueError
        If the two score arrays differ in length, contain fewer than
        two values, or ``n_outer_folds < 2``.
    """
    from scipy import stats

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"Score arrays must match shape: {a.shape} vs {b.shape}"
        )
    if a.ndim != 1 or a.size < 2:
        raise ValueError("Need at least 2 paired scores per model")
    if n_outer_folds < 2:
        raise ValueError(
            f"n_outer_folds must be >= 2 for the correction; got {n_outer_folds}"
        )

    d = a - b
    n = d.size
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    if var_d == 0.0:
        return 0.0, 1.0, n - 1, mean_d

    correction = 1.0 / n + 1.0 / (n_outer_folds - 1)
    t_stat = mean_d / np.sqrt(correction * var_d)
    df = n - 1
    p_two_sided = 2.0 * stats.t.sf(abs(t_stat), df)
    return float(t_stat), float(p_two_sided), df, mean_d


# ---------------------------------------------------------------------------
# Benjamini–Hochberg adjusted p-values
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted p-values for a family of comparisons.

    Step-up procedure: with sorted p-values
    ``p_(1) <= ... <= p_(m)``, the adjusted value at rank ``i`` is

    ``q_(i) = min_{j >= i} (m / j * p_(j))``  (clipped at 1).

    The result is in the original (input) order — element ``i`` is the
    adjusted p-value for the ``i``-th raw input.

    Parameters
    ----------
    p_values : Sequence[float]
        Raw two-sided p-values from each pairwise test.

    Returns
    -------
    np.ndarray
        Adjusted p-values, same length and order as the input.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    raw = ranked * m / ranks
    # Step-up monotonicity: enforce non-increasing from the top.
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    out = np.empty_like(adjusted_sorted)
    out[order] = adjusted_sorted
    return out


# ---------------------------------------------------------------------------
# Pairwise comparison helpers
# ---------------------------------------------------------------------------


@dataclass
class PairwiseComparison:
    """One pairwise test result.

    Attributes
    ----------
    model_a, model_b : str
        Names of the two compared runs.
    t_stat : float
    p_raw : float
        Two-sided uncorrected p-value from the Bouckaert–Frank test.
    p_adjusted : float
        BH-adjusted p-value across the full pairwise family.
    mean_diff : float
        Mean of (score_a - score_b) over outer folds.
    df : int
        Degrees of freedom used.
    """

    model_a: str
    model_b: str
    t_stat: float
    p_raw: float
    p_adjusted: float
    mean_diff: float
    df: int


def pairwise_compare(
    runs: Dict[str, Sequence[float]],
    *,
    n_outer_folds: int,
) -> List[PairwiseComparison]:
    """Run all unordered ``(A, B)`` pairwise tests and BH-correct them.

    Parameters
    ----------
    runs : dict
        Mapping of model run name → per-outer-fold score array. All arrays
        must have the same length.
    n_outer_folds : int
        Number of outer folds in the protocol (passes through to the
        correction formula).

    Returns
    -------
    List[PairwiseComparison]
        One entry per unordered pair, in deterministic (sorted-name) order.
    """
    names = sorted(runs)
    if len(names) < 2:
        raise ValueError(f"Need at least 2 model runs, got {len(names)}")
    lengths = {len(np.asarray(runs[n])) for n in names}
    if len(lengths) != 1:
        raise ValueError(
            f"All runs must have the same number of scores; got lengths {lengths}"
        )

    pairs: List[Tuple[str, str]] = []
    raw_results: List[Tuple[float, float, int, float]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs.append((a, b))
            raw_results.append(
                corrected_resampled_paired_t_test(
                    runs[a], runs[b], n_outer_folds=n_outer_folds,
                )
            )

    p_adj = benjamini_hochberg([r[1] for r in raw_results])
    out: List[PairwiseComparison] = []
    for (a, b), (t_stat, p_raw, df, mean_diff), q in zip(pairs, raw_results, p_adj):
        out.append(
            PairwiseComparison(
                model_a=a,
                model_b=b,
                t_stat=t_stat,
                p_raw=float(p_raw),
                p_adjusted=float(q),
                mean_diff=mean_diff,
                df=df,
            )
        )
    return out


def to_pvalue_matrix(
    comparisons: Iterable[PairwiseComparison],
    *,
    use_adjusted: bool,
) -> Tuple[List[str], np.ndarray]:
    """Build a symmetric m×m p-value matrix from pairwise results.

    The diagonal is filled with ``NaN`` (no self-comparison).

    Parameters
    ----------
    comparisons : Iterable[PairwiseComparison]
    use_adjusted : bool
        ``True`` → fill with BH-adjusted p-values; ``False`` → raw values.

    Returns
    -------
    Tuple[List[str], np.ndarray]
        Sorted model-name list and the m×m matrix indexed in that order.
    """
    comp_list = list(comparisons)
    names = sorted({c.model_a for c in comp_list} | {c.model_b for c in comp_list})
    idx = {n: i for i, n in enumerate(names)}
    m = len(names)
    mat = np.full((m, m), np.nan, dtype=float)
    for c in comp_list:
        p = c.p_adjusted if use_adjusted else c.p_raw
        i, j = idx[c.model_a], idx[c.model_b]
        mat[i, j] = p
        mat[j, i] = p
    return names, mat
