# Cross-model comparison — metric: `r2` (higher is better)
## Ranking
| Rank | Run | Mean | Std |
|------|-----|------|-----|
| 1 | enet-pnc-glm-vwm | 0.2439 | 0.0449 |
| 2 | mlp-pnc-glm-vwm | 0.2272 | 0.0559 |
| 3 | xgb-pnc-glm-vwm | 0.2238 | 0.0530 |

## Pairwise Bouckaert-Frank corrected t-test
Two-sided p-values. `p_adj` is Benjamini-Hochberg-adjusted across all pairwise comparisons. `mean_diff` is `A - B`.
| A | B | t | p_raw | p_adj | mean_diff | sig (q<0.05) |
|---|---|---|-------|-------|-----------|--------------|
| enet-pnc-glm-vwm | mlp-pnc-glm-vwm | +0.591 | 0.5575 | 0.8363 | +0.0167 | no |
| enet-pnc-glm-vwm | xgb-pnc-glm-vwm | +1.184 | 0.2422 | 0.7266 | +0.0201 | no |
| mlp-pnc-glm-vwm | xgb-pnc-glm-vwm | +0.104 | 0.9176 | 0.9176 | +0.0034 | no |

## Protocol
- Repetitions: 10
- Outer folds: 5
- Inner HPO trials: 20
- HPO metric: val_r2
