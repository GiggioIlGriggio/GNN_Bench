# Cross-model comparison — metric: `r2` (higher is better)
## Ranking
| Rank | Run | Mean | Std |
|------|-----|------|-----|
| 1 | xgb-pnc-glm-age-361342 | 0.1064 | 0.0462 |
| 2 | enet-pnc-glm-age-361341 | 0.0741 | 0.0400 |
| 3 | mlp-pnc-glm-age-fixinput-361347 | 0.0399 | 0.0806 |

## Pairwise Bouckaert-Frank corrected t-test
Two-sided p-values. `p_adj` is Benjamini-Hochberg-adjusted across all pairwise comparisons. `mean_diff` is `A - B`.
| A | B | t | p_raw | p_adj | mean_diff | sig (q<0.05) |
|---|---|---|-------|-------|-----------|--------------|
| enet-pnc-glm-age-361341 | mlp-pnc-glm-age-fixinput-361347 | +0.939 | 0.3524 | 0.3524 | +0.0342 | no |
| enet-pnc-glm-age-361341 | xgb-pnc-glm-age-361342 | -1.283 | 0.2057 | 0.3085 | -0.0323 | no |
| mlp-pnc-glm-age-fixinput-361347 | xgb-pnc-glm-age-361342 | -1.633 | 0.109 | 0.3085 | -0.0665 | no |

## Protocol
- Repetitions: 10
- Outer folds: 5
- Inner HPO trials: 20
- HPO metric: val_r2
