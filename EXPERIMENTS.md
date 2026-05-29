# EXPERIMENTS

A running log of the cluster experiments launched from this repo. Newest batch
on top. Each experiment is reproducible from the fields below alone.

## Entry format / template

Copy this block for every new experiment. Fill `TBD` fields once the job is
submitted (`DEPLOY_SHA`, `JOB_ID`, wandb URL) and again once it finishes
(`Results`).

```
### <experiment_name>
- **Date:** YYYY-MM-DD
- **Description:** what was tested and why, in one or two sentences.
- **Dataset / target:** dataset + label/target.
- **Changed parameters:** the Hydra overrides that distinguish this run from the
  shared baseline recipe (the full recipe is recorded once in "Shared recipe").
- **Commit SHA (DEPLOY_SHA):** the SHA the job actually ran on (reported by
  `cluster-submit`).
- **Job ID:** Slurm job id (and `-J` display name == experiment_name).
- **wandb run:** project/entity + run URL.
- **Command:** the exact `cluster-submit` line used.
- **Results:** mean ± std over outer folds (`pearson_r`, `r2`, `mae`, `rmse`),
  appended after the run completes. `TBD` until then.
```

---

## Batch 2026-05-29 — VWM GLM node-feature encodings (7-run matrix)

**Goal.** Compare GLM-derived node-feature encodings for predicting working
memory (`VWM_overall_dprime`) on PNC structural connectivity, holding the model
(GCN) and the nested-CV protocol fixed. The matrix crosses three "carrier"
features (`identity`, `sc_row` SC-profile, `laplacian_pe`) against two GLM forms
(`glm_scalar`, `glm_diagonal`), plus a standalone `glm_diagonal` baseline.

**Normalization policy (decided 2026-05-29).** `glm_normalize=true` for **all**
7 runs. A 2×2 smoke diagnosis (jobs 360728/360729/360742/360743) found the
true-vs-false effect to be within fold-to-fold noise, but turning normalization
*off* collapsed the `identity_glm_scalar` form to R²≈−0.03 (a scaling/conditioning
failure where the lone raw GLM column is unusable next to the 0/1 identity
carrier). `true` is the principled default for the 6 mixed-scale concatenation
presets and removes the scalar→true / diagonal→false confound baked into the
stock presets. Forced explicitly via `features.glm_normalize=true` on every run.

**Shared recipe (mirrors the identity/age baseline job 360724).**
```
dataset=pnc model=gcn labels=pnc_VWMdprime \
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml \
trainer.hpo_metric=val_r2 \
features.glm_normalize=true
```
- 300 epochs (trainer default); contrast `contrast-2back_vs_0back`; edge feature
  `weight`; GLM map type `zmap`.
- Wall-clock cap: `--time=2-00:00:00` (2 days) passed to `sbatch`.
- wandb: project `orbitglm`, entity `teampolpetta`.
- All 7 deployed from a single SHA on branch `feature/vwm-glm-presets`.

**Matrix.**

| # | experiment_name | features preset | node_features | dim |
|---|---|---|---|---|
| 1 | `gcn-pnc-sc-vwm-glmdiag`          | `glm_diagonal`              | `[glm_diagonal]`            | 400 |
| 2 | `gcn-pnc-sc-vwm-id-glmscalar`     | `identity_glm_scalar`       | `[identity, glm_scalar]`    | 401 |
| 3 | `gcn-pnc-sc-vwm-id-glmdiag`       | `identity_glm_diagonal`     | `[identity, glm_diagonal]`  | 800 |
| 4 | `gcn-pnc-sc-vwm-scprof-glmscalar` | `scprofile_glm_scalar`      | `[sc_row, glm_scalar]`      | 401 |
| 5 | `gcn-pnc-sc-vwm-scprof-glmdiag`   | `scprofile_glm_diagonal`    | `[sc_row, glm_diagonal]`    | 800 |
| 6 | `gcn-pnc-sc-vwm-lappe-glmscalar`  | `laplacian_pe_glm_scalar`   | `[laplacian_pe, glm_scalar]`| 9   |
| 7 | `gcn-pnc-sc-vwm-lappe-glmdiag`    | `laplacian_pe_glm_diagonal` | `[laplacian_pe, glm_diagonal]` | 408 |

---

### gcn-pnc-sc-vwm-glmdiag
- **Date:** 2026-05-29
- **Description:** Standalone GLM diagonal embedding baseline — GLM activation on each node's own diagonal cell, no carrier feature.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=glm_diagonal` (+ shared recipe; `features.glm_normalize=true` overrides this preset's default of `false`).
- **Commit SHA (DEPLOY_SHA):** TBD
- **Job ID:** TBD (`-J gcn-pnc-sc-vwm-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** TBD
- **Results:** TBD

### gcn-pnc-sc-vwm-id-glmscalar
- **Date:** 2026-05-29
- **Description:** Identity one-hot carrier ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** TBD
- **Job ID:** TBD (`-J gcn-pnc-sc-vwm-id-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** TBD
- **Results:** TBD

### gcn-pnc-sc-vwm-id-glmdiag
- **Date:** 2026-05-29
- **Description:** Identity one-hot carrier ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** TBD
- **Job ID:** TBD (`-J gcn-pnc-sc-vwm-id-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** TBD
- **Results:** TBD

### gcn-pnc-sc-vwm-scprof-glmscalar
- **Date:** 2026-05-29
- **Description:** SC connectivity-profile carrier (`sc_row`) ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=scprofile_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** TBD
- **Job ID:** TBD (`-J gcn-pnc-sc-vwm-scprof-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** TBD
- **Results:** TBD

### gcn-pnc-sc-vwm-scprof-glmdiag
- **Date:** 2026-05-29
- **Description:** SC connectivity-profile carrier (`sc_row`) ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=scprofile_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** TBD
- **Job ID:** TBD (`-J gcn-pnc-sc-vwm-scprof-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** TBD
- **Results:** TBD

### gcn-pnc-sc-vwm-lappe-glmscalar
- **Date:** 2026-05-29
- **Description:** Laplacian PE carrier (k=8, listed first for sign-flip) ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=laplacian_pe_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** TBD
- **Job ID:** TBD (`-J gcn-pnc-sc-vwm-lappe-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** TBD
- **Results:** TBD

### gcn-pnc-sc-vwm-lappe-glmdiag
- **Date:** 2026-05-29
- **Description:** Laplacian PE carrier (k=8, listed first for sign-flip) ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=laplacian_pe_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** TBD
- **Job ID:** TBD (`-J gcn-pnc-sc-vwm-lappe-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** TBD
- **Results:** TBD
