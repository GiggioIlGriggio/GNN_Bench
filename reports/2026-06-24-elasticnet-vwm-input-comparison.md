# ElasticNet VWM input comparison — GLM vs connectivity vs concatenation

**Date:** 2026-06-24
**Batch:** [EXPERIMENTS.md → Batch 2026-06-24](../EXPERIMENTS.md#batch-2026-06-24--elasticnet-vwm-input-comparison-glm-vs-connectivity-vs-concatenation)
**Branch:** `feature/elasticnet-vwm-concat` · deploy SHA `755ac5b` · wandb project `baselines`

## TL;DR

Does adding raw connectivity to the GLM-activation vector improve a non-graph
ElasticNet at predicting PNC VWM (`pnc_VWMdprime`)? One new run (`enet-pnc-scglm-vwm`,
the `model.mlp_input=both` concatenation) joins the two existing arms from the
2026-06-11 batch, all at the GNN's 10×5×20 nested-CV protocol on identical folds.

1. **Concatenation does not beat GLM-only.** GLM-only pooled r² = **0.246**; the
   GLM ⊕ connectivity concat = **0.204** — nominally *lower*. On matched folds the
   corrected resampled paired t-test gives `mean_diff −0.044, t=−1.33, p=0.19` —
   **not significant**.
2. **Connectivity adds no detectable VWM signal on top of GLM.** The point estimate
   trends in the *dilution* direction (the 400 GLM dims are 0.5% of the 80,200-dim
   concat), but the drop is within fold noise. The graph weights contribute nothing
   measurable beyond the GLM contrast.
3. **GLM ≫ connectivity-alone** (0.246 vs 0.110) replicates the 2026-06-11 finding.

## 1. What was tested

| Arm | features / mlp_input | input dim | status |
|---|---|---|---|
| GLM only | `glm_scalar` / `node_features` | 400 | reuse (361221, 2026-06-11) |
| Connectivity only (SC) | `default` / `adjacency` | 79,800 | reuse (361214, 2026-06-11) — context only |
| **GLM ⊕ connectivity** | `glm_scalar` / `both` | **80,200** | **new (362523)** |

Estimator ElasticNet; dataset PNC; target `pnc_VWMdprime`. Protocol 10 reps × 5
outer folds × 20 inner Optuna trials, maximize `val_r2`; per-fold `StandardScaler`
(train-only); folds byte-identical via `src/training/splits.py`. The concat is the
exact union of the two existing arms' inputs — `flatten.py`'s `input_mode="both"`
concatenates `[adjacency_vector(79,800) ++ node_feature_vector(400)]`, no new code.

## 2. Results

| experiment_name | input (dim) | r² (mean-of-folds) | r² (pooled) | Pearson r (pooled) | cohort (n_pooled) | run |
|---|---|---|---|---|---|---|
| `enet-pnc-glm-vwm` (361221)   | glm_scalar (400)     | **0.244 ± 0.045** | **0.246** | 0.496 | 940 (9400) | [40hguv2e](https://wandb.ai/teampolpetta/baselines/runs/40hguv2e) |
| `enet-pnc-scglm-vwm` (362523) | both (80,200)        | 0.200 ± 0.063 | 0.204 | 0.452 | 940 (9400) | [sjg97qk2](https://wandb.ai/teampolpetta/baselines/runs/sjg97qk2) |
| `enet-pnc-sc-vwm` (361214)    | adjacency (79,800)   | 0.109 ± 0.033 | 0.110 | 0.333 | 973 (9730)* | [4fwu06yf](https://wandb.ai/teampolpetta/baselines/runs/4fwu06yf) |

`±` is fold dispersion, not a standard error (the 50 folds are correlated).

## 3. Statistical test (the headline question)

Corrected resampled paired t-test (`scripts/compare_models.py`, ADR-0008) on `r2`,
matched 940-subject folds (both arms `n_pooled=9400`):

```
enet-concat vs enet-glm-only : mean_diff −0.044  t=−1.33  p=0.19  (ns)
```

**No significant difference.** Adding connectivity neither helps nor (significantly)
hurts; the trend is toward dilution but within noise. Artifacts:
`reports/comparison/vwm-glm-vs-concat/`.

## 4. Caveats

- **Conn-only is on a different (larger) cohort and is context-only.** `enet-pnc-sc-vwm`
  used `features=default`, so it never loaded the GLM map and ran on **973** subjects
  (`n_pooled=9730`) vs the 940-subject GLM cohort of the other two arms. It is therefore
  **not fold-matched** and cannot enter a valid paired test against the GLM-bearing arms;
  it is reported only to anchor the connectivity-alone ceiling (~0.11). A fully matched
  conn-only cell (`features=glm_scalar model.mlp_input=adjacency`, 940 cohort) was not run.
- **Dimensional imbalance is the mechanism, not a bug.** Concat is 99.5% connectivity by
  dimension; per-column `StandardScaler` + shared L1/L2 means the 400 GLM columns compete
  with 79,800 connectivity columns. The dilution direction is expected.
- **Representation comparison.** All three arms are ElasticNet; this isolates *input*, not
  estimator or architecture. The GNN-vs-classical question was settled in the 2026-06-11
  batch (classical ≈ GNN on GLM→VWM, all p_adj=0.92).

## 5. Provenance / how to reproduce

- New run: job **362523**, `node01` (CPU partition `cluster`), deploy SHA `755ac5b`,
  `slurm/train_sklearn.sh`, wandb `teampolpetta/baselines` run `sjg97qk2`.
- Recipe: `dataset=pnc model=elasticnet features=glm_scalar labels=pnc_VWMdprime
  model.mlp_input=both experiment_name=enet-pnc-scglm-vwm trainer.n_repetitions=10
  trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.hpo_metric=val_r2
  trainer.search_space=configs/sweeper/elasticnet.yaml`.
- Reuse artifacts: `enet-pnc-glm-vwm-361221`, `enet-pnc-sc-vwm-361214` (cluster
  `checkpoints/<exp>-<jobid>/nested_cv_result.json`).
- Pooled metrics:
  `.claude/skills/backfill-experiment-results/scripts/compute-pooled-metrics.py <json>`.
- Comparison: `scripts/compare_models.py --metric r2 --inputs <glm.json> <concat.json>
  --output-dir reports/comparison/vwm-glm-vs-concat`.
