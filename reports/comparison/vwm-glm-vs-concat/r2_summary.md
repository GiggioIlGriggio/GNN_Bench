# Cross-model comparison — metric: `r2` (higher is better)
## Ranking
| Rank | Run | Mean | Std |
|------|-----|------|-----|
| 1 | enet-glm-only | 0.2439 | 0.0449 |
| 2 | enet-concat | 0.2003 | 0.0631 |

## Pairwise Bouckaert-Frank corrected t-test
Two-sided p-values. `p_adj` is Benjamini-Hochberg-adjusted across all pairwise comparisons. `mean_diff` is `A - B`.
| A | B | t | p_raw | p_adj | mean_diff | sig (q<0.05) |
|---|---|---|-------|-------|-----------|--------------|
| enet-concat | enet-glm-only | -1.334 | 0.1884 | 0.1884 | -0.0436 | no |

## Protocol
- Repetitions: 10
- Outer folds: 5
- Inner HPO trials: 20
- HPO metric: val_r2
