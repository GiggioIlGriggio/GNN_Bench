# Cross-model comparison — metric: `r2` (higher is better)
## Ranking
| Rank | Run | Mean | Std |
|------|-----|------|-----|
| 1 | enet-glmscalar | 0.2439 | 0.0449 |
| 2 | mlp-glmscalar | 0.2272 | 0.0559 |
| 3 | xgb-glmscalar | 0.2238 | 0.0530 |
| 4 | gnn-id-glmdiag | 0.2179 | 0.0933 |
| 5 | gnn-glmdiag | 0.1939 | 0.1002 |

## Pairwise Bouckaert-Frank corrected t-test
Two-sided p-values. `p_adj` is Benjamini-Hochberg-adjusted across all pairwise comparisons. `mean_diff` is `A - B`.
| A | B | t | p_raw | p_adj | mean_diff | sig (q<0.05) |
|---|---|---|-------|-------|-----------|--------------|
| enet-glmscalar | gnn-glmdiag | +0.995 | 0.3248 | 0.9176 | +0.0500 | no |
| enet-glmscalar | gnn-id-glmdiag | +0.574 | 0.5684 | 0.9176 | +0.0260 | no |
| enet-glmscalar | mlp-glmscalar | +0.591 | 0.5575 | 0.9176 | +0.0167 | no |
| enet-glmscalar | xgb-glmscalar | +1.184 | 0.2422 | 0.9176 | +0.0201 | no |
| gnn-glmdiag | gnn-id-glmdiag | -0.344 | 0.7319 | 0.9176 | -0.0240 | no |
| gnn-glmdiag | mlp-glmscalar | -0.621 | 0.5377 | 0.9176 | -0.0333 | no |
| gnn-glmdiag | xgb-glmscalar | -0.566 | 0.5738 | 0.9176 | -0.0299 | no |
| gnn-id-glmdiag | mlp-glmscalar | -0.206 | 0.838 | 0.9176 | -0.0093 | no |
| gnn-id-glmdiag | xgb-glmscalar | -0.123 | 0.903 | 0.9176 | -0.0059 | no |
| mlp-glmscalar | xgb-glmscalar | +0.104 | 0.9176 | 0.9176 | +0.0034 | no |

## Protocol
- Repetitions: 10
- Outer folds: 5
- Inner HPO trials: 20
- HPO metric: val_r2
