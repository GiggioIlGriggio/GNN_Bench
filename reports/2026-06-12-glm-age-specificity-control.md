# GLM→age specificity control — is the GLM-activation vector VWM-specific?

**Date:** 2026-06-12
**Batch:** [EXPERIMENTS.md → Batch 2026-06-12](../EXPERIMENTS.md#batch-2026-06-12--glmage-specificity-control-does-the-glm-activation-vector-predict-age)
**Branch:** `feature/glm-age-specificity-control` @ `6634879` (sweeper variant `48e5d3d`) · wandb project `baselines`, entity `teampolpetta`
**Follows up:** [`reports/2026-06-11-classical-ml-baselines.md`](2026-06-11-classical-ml-baselines.md) §3

## TL;DR

The 2026-06-11 headline was that PNC working-memory (VWM) is best predicted by the
400-dim `glm_scalar` activation vector (pooled r² ≈ 0.23–0.25), beating connectivity
(≈0.10–0.12) and tying the GNN. **But the GLM vector had never been used to predict
age** — so we could not tell whether it carries *VWM-specific* signal or is a generic
developmental / SNR / engagement proxy. This control predicts PNC `age_at_cnb` from
the same `glm_scalar` vector, same 10×5×20 nested-CV protocol.

1. **The GLM vector predicts age weakly** — the three strict GLM-only estimators give
   pooled r² **0.075 (ElasticNet) / 0.106 (XGBoost) / 0.041 (pinned-input MLP)** —
   range 0.04–0.11, statistically indistinguishable (§4).
2. **That is far below GLM→VWM** (0.246 / 0.226 / 0.230 from the *same* vector): the
   GLM activation pattern predicts VWM **2–6× better than it predicts age** (ElasticNet
   3.3×, XGBoost 2.1×, MLP 5.6×).
3. **And far below what age allows** — connectivity recovers age at SC ≈0.51 / FC ≈0.33,
   so the GLM vector captures only **~1/5–1/12** of the decodable age signal. The GLM
   vector is **not** a strong generic-maturation proxy.
4. **Verdict: the GLM pattern is substantially VWM-specific**, with a small non-zero
   age component (partial confound, not a dominant one). The 2026-06-11 VWM headline
   stands; a maximally rigorous VWM claim could age-residualize, but specificity is clear.
5. **The "MLP" cell needs care.** With input mode left to the HPO, the MLP *escapes to
   connectivity* for age (recovers ≈0.573 ≈ connectivity→age), because for
   age — unlike VWM — connectivity ≫ GLM (it picked GLM-only input in **0 of 50
   folds**). A pinned-input MLP (GLM-only) lands at **0.041**, with the other
   strict estimators.

## 1. The question

PNC is an 8–21yo developmental sample where connectivity predicts age strongly. If the
`glm_scalar` vector — which carries the 2-back-vs-0-back task contrast — predicted age
as well as it predicts VWM (or as well as connectivity does), its VWM result would be
suspect as a generic maturation/quality confound. Interpretation grid (set before the run):

| GLM→age r² | Meaning |
|---|---|
| ≈ 0 | **Strong VWM-specificity** — GLM pattern carries WM signal, not generic maturation. |
| moderate but < GLM→VWM (~0.24) | **Partial confound** — some age signal, VWM still stronger. |
| ≈ or > GLM→VWM | Generic brain-maturation/quality proxy; VWM result **not** specific. |

Design: identical to the 2026-06-11 PNC GLM→VWM cells except `labels=pnc_default`
(`age_at_cnb`) replaces `labels=pnc_VWMdprime`. Input = `glm_scalar`
(`contrast-2back_vs_0back`, z-map, mean-aggregated, 400-dim, 1/node). N = **945**
(993 − 48 GLM-missing; age non-null for all) — a +5 superset of the 940 GLM→VWM
subjects, so GLM→age folds are **not** byte-identical to GLM→VWM and the cross-target
contrast is **magnitude-based**, not a paired test (per the batch plan).

## 2. Result — the GLM vector predicts age far worse than VWM

Strict GLM-only estimators (input pinned to the 400-dim `node_features` vector):

| estimator | GLM→**age** pooled r² | mean-of-folds r² | GLM→**VWM** pooled r² (2026-06-11) | VWM/age |
|---|---|---|---|---|
| ElasticNet | **0.075** | 0.074 ± 0.040 | 0.246 | 3.3× |
| XGBoost | **0.106** | 0.106 ± 0.046 | 0.226 | 2.1× |
| MLP (input pinned) | **0.041** | 0.040 ± 0.081 | 0.230 | 5.6× |

(`pooled` = single r² over all 9,450 out-of-fold predictions = 945 subjects × 10 reps;
± is fold dispersion over 50 folds, not a SE.)

All three strict estimators agree the GLM vector carries **2–6× more VWM signal than age
signal**. For context, the same connectivity-vs-GLM asymmetry runs the other way for
the two targets:

| PNC pooled r² | from `glm_scalar` (400-dim) | from connectivity (79,800-dim) |
|---|---|---|
| **VWM** | 0.23–0.25 | 0.10–0.12 |
| **age** | **0.04–0.11** | 0.50–0.52 (SC) / 0.31–0.37 (FC) |

The GLM vector is the *better* input for VWM and the *far worse* input for age — the
double dissociation that specificity requires.

## 3. The MLP input-escape — the sweeping cell measures connectivity→age, not GLM→age

`configs/sweeper/mlp.yaml` sweeps `model.mlp_input ∈ {node_features, adjacency, both}`
in the inner HPO. For **VWM** this is benign — connectivity→VWM (~0.10) < GLM→VWM
(~0.24), so the HPO keeps the GLM vector and `mlp-pnc-glm-vwm` is a genuine GLM→VWM
number. For **age** it is not: connectivity→age (~0.51) ≫ GLM→age (~0.10), so the HPO
**abandons the GLM vector for connectivity**. The sweeping `mlp-pnc-glm-age` (361343)
reports pooled r² = **0.573** — i.e. it recovers (and slightly exceeds) the existing
`mlp-pnc-sc-age` (0.521) connectivity result, **not** GLM→age. The `best_hparams` are
decisive: in **0 of 50 folds** did the inner HPO keep GLM-only input — it chose `both`
(GLM + connectivity) in 27 folds and `adjacency` in 23 (49/50 `weighted`). Even the
`both` folds (mean fold r² 0.581) ≈ the `adjacency` folds (0.565), both ≈ connectivity→age
— the GLM channel adds nothing detectable once connectivity is available for age.

To get a genuine third strict GLM-only estimator we re-ran the MLP with input pinned
(`configs/sweeper/mlp_fixedinput.yaml`, which drops the `mlp_input`/`mlp_adjacency_type`
sweep dims, + `model.mlp_input=node_features`). Pinned, the MLP lands at
**0.041** (pooled) — with ElasticNet and XGBoost, confirming the ~0.5 of the sweeping
cell was **entirely input-escape**, not nonlinear GLM→age signal (an MLP cannot extract
r²≈0.5 from a 400-dim vector on which nonlinear XGBoost gets 0.106).

This is the MLP input-mode caveat (the standard `mlp` sweeper sweeps `mlp_input`),
materialised as a *complete* escape for the age target. The 2026-06-11 batch ran the
same sweeping MLP, but for VWM the HPO *kept* GLM-only input (connectivity→VWM < GLM→VWM),
so `mlp-pnc-glm-vwm` there is unaffected — the asymmetry is target-specific.

## 4. Estimators are interchangeable (within-target, strict GLM-only)

Corrected resampled paired t-test (`scripts/compare_models.py`, ADR-0008) on `r2` over
the identical 50 folds (all strict GLM-only cells share N=945):

```
enet-pnc-glm-age vs xgb-pnc-glm-age          : mean_diff −0.032  t −1.28  p_adj 0.31  (ns)
enet-pnc-glm-age vs mlp-pnc-glm-age-fixinput : mean_diff +0.034  t +0.94  p_adj 0.35  (ns)
xgb-pnc-glm-age  vs mlp-pnc-glm-age-fixinput : mean_diff +0.067  t +1.63  p_adj 0.31  (ns)
```

As on the 2026-06-11 signal-bearing cells, the choice of strict estimator does not
matter — ElasticNet ≈ XGBoost ≈ pinned-MLP (all pairs p_adj ≥ 0.31). Artifacts:
`reports/comparison/pnc-glm-age-estimators/`.

## 5. Verdict

GLM→age ≈ **0.04–0.11** (ElasticNet 0.075, XGBoost 0.106, pinned MLP 0.041) sits in the
grid's middle band but near its low end: a **small, non-zero age component**, **2–6× below**
GLM→VWM and **~5–12× below** connectivity→age.
The GLM activation pattern is therefore **substantially VWM-specific**, not a generic
brain-maturation/quality proxy — if it were, it would predict age comparably to
connectivity (0.51), which it does not. The 2026-06-11 VWM headline is **strengthened**.
The residual age signal is real but minor; a fully rigorous VWM claim could regress age
out first, but the specificity conclusion does not depend on it.

## 6. Caveats

- **Cross-target comparison is magnitude-based, not paired** — GLM→age (N=945) and
  GLM→VWM (N=940) use different subject sets (+5), so folds are not byte-identical; the
  within-target estimator test (§4) *is* paired and valid.
- **± is fold dispersion, not a SE** — 50 folds are correlated; use the corrected t-test
  for model-vs-model, never ±-overlap.
- **The sweeping MLP cell is connectivity→age, not GLM→age** (§3) — reported only as a
  positive control; the strict GLM→age estimators are ElasticNet, XGBoost, and the
  pinned-input MLP.
- **PNC-only by design** — ORBIT (N≈95) is underpowered for this control (2026-06-11 §5);
  not run unless asked.

## 7. Provenance / reproduce

- Branch `feature/glm-age-specificity-control` @ `6634879`; pinned-MLP sweeper added in
  `48e5d3d` (`configs/sweeper/mlp_fixedinput.yaml`). wandb project `baselines`, entity
  `teampolpetta`.
- Job IDs: `enet-pnc-glm-age` 361341, `xgb-pnc-glm-age` 361342, `mlp-pnc-glm-age`
  (sweeping) 361343, `mlp-pnc-glm-age-fixinput` (pinned) 361347. Smokes 361338–340, 361346.
- Submit recipe + per-cell overrides: EXPERIMENTS.md (Batch 2026-06-12). Only delta from
  the 2026-06-11 GLM→VWM recipe is `labels=pnc_default`.
- Pooled metrics: concatenate `y_true`/`y_pred` across the 50 `fold_results` in each
  `checkpoints/<cell>-<jobid>/nested_cv_result.json`. Comparisons:
  `scripts/compare_models.py --inputs <jsons> --metric r2`.
