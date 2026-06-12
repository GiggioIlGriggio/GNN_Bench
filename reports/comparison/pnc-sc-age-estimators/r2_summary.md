# Cross-model comparison — metric: `r2` (higher is better)
## Ranking
| Rank | Run | Mean | Std |
|------|-----|------|-----|
| 1 | mlp-pnc-sc-age | 0.5209 | 0.0893 |
| 2 | enet-pnc-sc-age | 0.5162 | 0.0558 |
| 3 | xgb-pnc-sc-age | 0.5019 | 0.0525 |

## Pairwise Bouckaert-Frank corrected t-test
Two-sided p-values. `p_adj` is Benjamini-Hochberg-adjusted across all pairwise comparisons. `mean_diff` is `A - B`.
| A | B | t | p_raw | p_adj | mean_diff | sig (q<0.05) |
|---|---|---|-------|-------|-----------|--------------|
| enet-pnc-sc-age | mlp-pnc-sc-age | -0.095 | 0.925 | 0.925 | -0.0048 | no |
| enet-pnc-sc-age | xgb-pnc-sc-age | +0.714 | 0.4784 | 0.925 | +0.0143 | no |
| mlp-pnc-sc-age | xgb-pnc-sc-age | +0.419 | 0.677 | 0.925 | +0.0191 | no |

## Protocol
- Repetitions: 10
- Outer folds: 5
- Inner HPO trials: 20
- HPO metric: val_r2
