# Classical-ML baselines (XGBoost / ElasticNet / MLP) vs the GNN

**Date:** 2026-06-11
**Batch:** [EXPERIMENTS.md → Batch 2026-06-11](../EXPERIMENTS.md#batch-2026-06-11--classical-ml-baselines-xgboost--elasticnet--mlp-vs-the-gnn-30-run-matrix)
**Branch:** `feature/classical-ml-baselines` @ `c5a6f7b` · wandb project `baselines`
**Spec:** `docs/superpowers/specs/2026-06-10-classical-ml-connectivity-glm-baselines-design.md`

## TL;DR

A 30-cell matrix of non-graph baselines (XGBoost, ElasticNet, MLP) learning
directly from connectivity (top-20% thresholded weighted upper-triangle, 79,800-dim)
and from the GLM-activation vector (`glm_scalar`, 400-dim), at the GNN's exact
10×5×20 nested-CV protocol on identical folds.

1. **PNC age is strongly recovered from connectivity** (SC pooled r² ≈ 0.50–0.52,
   FC ≈ 0.31–0.37) — the pipeline is correctly wired.
2. **PNC VWM is only weakly predictable from connectivity** (≈ 0.10–0.12) **but
   best predicted by the GLM-activation vector** (≈ 0.23–0.25) — ~2× the
   connectivity ceiling, from a 400-dim input vs 79,800-dim.
3. **The GNN does not beat the non-graph models for VWM-from-GLM.** A corrected
   resampled paired t-test on matched folds finds **no significant difference**
   between the classical models (`glm_scalar`) and the GNN's best GLM cells
   (`glm_diagonal` through GCN): all p_adj = 0.92; classical is nominally *higher*
   (ElasticNet 0.246 vs GNN-id-glmdiag 0.222 pooled).
4. **The choice of classical estimator does not matter** on the signal-bearing
   cells — ElasticNet ≈ XGBoost ≈ MLP, statistically indistinguishable.
5. **ORBIT (N≈95–130) is underpowered** — near-zero/negative almost everywhere; the
   only positive is, again, GLM→VWM (xgb pooled 0.143).

## 1. What was tested

| Axis | Values |
|---|---|
| Estimator | XGBoost, ElasticNet (new sklearn path); MLP (existing torch `model=mlp`) |
| Input | connectivity SC, connectivity FC (79,800-dim) · GLM `glm_scalar` (400-dim) |
| Target | age, VWM (GLM input is VWM-only) |
| Dataset | PNC (N≈940–993), ORBIT (N≈94–137) |

24 connectivity cells (input × target × dataset × estimator) + 6 GLM cells
(target=VWM × dataset × estimator) = **30 runs**. Protocol: 10 reps × 5 outer folds
× 20 inner Optuna trials, maximize `val_r2`. Folds are **byte-identical** to the
GNN via the shared `src/training/splits.py`; per-fold `StandardScaler` (train-only)
for the sklearn estimators; `FoldBarrier` reused for label/GLM normalization. The
connectivity vector is the exact top-20% thresholded weighted `Data` the GNN sees
(fairest head-to-head).

**Compute.** sklearn on CPU nodes; MLP on GPU (the GNN Trainer, epochs=300 +
early-stop patience 35). XGBoost·PNC·connectivity (79,800-dim) is ~65–70h on CPU at
this budget — over the 24h wall — so those 4 cells ran on **GPU**
(`+model.model_params.device=cuda`), finishing in 2h50m–6h20m with results matching
ElasticNet. All 30 cells `COMPLETED`.

## 2. Connectivity predicts age, not VWM (PNC)

| PNC, pooled r² | SC | FC |
|---|---|---|
| **age** | 0.50–0.52 (enet 0.516, xgb 0.501, mlp 0.521) | 0.31–0.37 |
| **VWM** | 0.11–0.11 | 0.10–0.12 |

Age is strongly linearly decodable from structural connectivity; VWM is barely
above chance from the same 79,800-dim vector. Note **SC→age > FC→age** here (≈0.51
vs ≈0.33) — the reverse of the usual FC-leads-for-age literature ordering, a
property of this PNC processing, not a bug (age recovery itself confirms the
pipeline).

## 3. The GLM-activation vector is the best VWM predictor — and the GNN doesn't beat it

Predicting PNC VWM (`pnc_VWMdprime`) from the 400-dim `glm_scalar` vector:

| run | input | model | pooled r² | mean-of-folds r² |
|---|---|---|---|---|
| `enet-pnc-glm-vwm` | glm_scalar | ElasticNet | **0.246** | 0.244 ± 0.045 |
| `mlp-pnc-glm-vwm`  | glm_scalar | MLP | 0.230 | 0.227 ± 0.056 |
| `xgb-pnc-glm-vwm`  | glm_scalar | XGBoost | 0.226 | 0.224 ± 0.053 |
| `gcn-pnc-sc-vwm-id-glmdiag` (360774) | glm_diagonal | GCN | 0.222 | 0.218 ± 0.093 |
| `gcn-pnc-sc-vwm-glmdiag` (360772) | glm_diagonal | GCN | 0.193 | 0.194 ± 0.100 |

vs the **connectivity** ceiling for the same target (≈0.10–0.12). The GLM
activation pattern carries roughly twice the VWM signal of raw connectivity.

**Statistical test (the spec's headline question).** Corrected resampled paired
t-test (`scripts/compare_models.py`, ADR-0008) on `r2`, over matched folds (all
five runs share the same 940 subjects / 50 folds — verified `n_pooled=9400`):

```
enet-glmscalar vs gnn-id-glmdiag : mean_diff +0.026  p_adj 0.92  (ns)
enet-glmscalar vs gnn-glmdiag    : mean_diff +0.050  p_adj 0.92  (ns)
mlp-glmscalar  vs gnn-id-glmdiag : mean_diff +0.009  p_adj 0.92  (ns)
xgb-glmscalar  vs gnn-id-glmdiag : mean_diff +0.006  p_adj 0.92  (ns)
```

**No model — classical or GNN — significantly outperforms any other.** The GNN's
graph message-passing and the `glm_diagonal` encoding add nothing detectable over a
plain linear/tree/MLP model on the scalar GLM vector. Artifacts:
`reports/comparison/glm-vwm-classical-vs-gnn/`.

(Encoding caveat: the classical models use `glm_scalar` and the GCN uses
`glm_diagonal` — both are "predict VWM from the GLM activation pattern," so this is a
representation+model comparison, not a pure architecture swap. There is no existing
GNN run on plain connectivity or on `glm_scalar` to make the swap purer; the GNN
batches were all GLM-node-feature studies.)

**Specificity control (2026-06-12 follow-up).** Does the `glm_scalar` vector predict
**age** too — i.e. is its VWM signal generic maturation/quality rather than WM-specific?
**No.** Strict GLM-only estimators give GLM→age pooled r² 0.075 (ElasticNet) / 0.106
(XGBoost) / 0.041 (pinned-input MLP) — **2–6× below** GLM→VWM (0.23–0.25, the *same*
vector) and **~5–12× below** connectivity→age (SC 0.51). The GLM activation pattern is
substantially **VWM-specific**, not a generic brain-maturation proxy; the residual age
signal is small but non-zero (partial confound). *Caveat made concrete here:* the MLP
GLM cells use **HPO-selected input** (the `mlp` sweeper sweeps `mlp_input`) — for **age**
the HPO escapes to connectivity (the sweeping `mlp-pnc-glm-age` recovers ~0.57 ≈
connectivity→age, GLM-only in 0/50 folds), so a **pinned-input** MLP is the valid GLM-only
estimator; for **VWM** it kept GLM input, so `mlp-pnc-glm-vwm` above is unaffected. Detail:
[`reports/2026-06-12-glm-age-specificity-control.md`](2026-06-12-glm-age-specificity-control.md)
· [EXPERIMENTS.md Batch 2026-06-12](../EXPERIMENTS.md#batch-2026-06-12--glmage-specificity-control-does-the-glm-activation-vector-predict-age).

## 4. Estimators are interchangeable

Within a cell (identical folds), the three classical estimators never differ
significantly (`compare_models.py`, `r2`):

- **PNC GLM→VWM:** all pairs p_adj ≥ 0.73 (max mean-diff 0.020).
- **PNC SC→age:** all pairs p_adj ≥ 0.92 (max mean-diff 0.019).

The p≫n regime (79,800 features, N≈940) does **not** favor XGBoost over ElasticNet,
as the spec anticipated. Artifacts under `reports/comparison/pnc-*-estimators/`.

## 5. ORBIT is underpowered

At N≈94–137, pooled r² is near-zero or negative for almost every ORBIT cell
(connectivity and GLM), with fold dispersion swamping any signal. The single
consistent positive is **GLM→VWM** (xgb pooled 0.143, mlp 0.116, enet 0.049),
echoing PNC's GLM>connectivity-for-VWM pattern. Read **pooled** r² on ORBIT —
mean-of-folds is dominated by ~25-sample fold-local means. ORBIT VWM uses a bounded
proportion-correct label (vs PNC's d-prime), so absolute r² is not comparable across
datasets; only within-dataset contrasts transfer.

## 6. Full results (reference)

All 30 cells, both r² flavours, per-fold dispersion, and wandb links:
[EXPERIMENTS.md → Batch 2026-06-11](../EXPERIMENTS.md#batch-2026-06-11--classical-ml-baselines-xgboost--elasticnet--mlp-vs-the-gnn-30-run-matrix).
Pairwise-comparison CSVs + markdown: `reports/comparison/`.

## 7. Caveats

- **± is fold dispersion, not a standard error** — the 50 folds are correlated (10
  reps reuse subjects; within a rep 5 folds share training data). For
  model-vs-model use the corrected t-test, never ±-overlap.
- **GNN comparison is representation+model, not pure architecture** (§3 caveat).
- **ORBIT underpowered** (§5); treat as direction-only.
- **SC→age > FC→age** on PNC is atypical; flagged for follow-up, not yet explained.
- The GNN GLM numbers come from batch 2026-06-04 (jobs 360772/360774); the older
  2026-05-29 batch was found run-to-run unstable — the 06-04 runs are the
  reproducible reference used here.

## 8. Provenance / how to reproduce

- Branch `feature/classical-ml-baselines` @ `c5a6f7b`; container rebuilt with
  `xgboost==2.1.4`; wandb project `baselines`, entity `teampolpetta`.
- Job IDs: ElasticNet 361213–361222; XGBoost CPU (ORBIT+GLM) 361223–361228;
  XGBoost GPU (PNC connectivity) 361231–361234; MLP 361235–361244.
- Submit recipe and per-cell overrides: EXPERIMENTS.md (Batch 2026-06-11).
- Pooled metrics:
  `.claude/skills/backfill-experiment-results/scripts/compute-pooled-metrics.py`
  on each `checkpoints/<cell>-<jobid>/nested_cv_result.json`.
- Comparisons: `scripts/compare_models.py --inputs <jsons> --metric r2`.
