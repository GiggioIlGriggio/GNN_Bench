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

## Batch 2026-06-01 — GLM value↔identity decoupling (value-permutation ablation)

**Goal.** Decouple the one-hot **node-identity structure** of `glm_diagonal` from
the **GLM activation magnitude** placed on it, on PNC SC → `VWM_overall_dprime`,
GCN + nested CV fixed. Tests thesis
`glm-value-identity-decoupling-permutation` (H1: the diagonal's signal is the
one-hot distinctness prior, not the value/binding). Introduces the additive
`features.glm_value_permute` knob (`none|per_subject|fixed`, seeded by
`glm_permute_seed`): GLM values are permuted across nodes *before* placement,
one-hot support intact. `none` is a byte-identical no-op, so the 2026-05-29
`glm_diagonal` run (job 360744) is reused unchanged as the correct-binding arm.

**Run matrix (4 arms; 3 new + 1 reused).** All 400-wide, used standalone.

| # | experiment_name | features preset | value binding |
|---|---|---|---|
| 1 | `gcn-pnc-sc-vwm-identity`          | `identity`              | — (constant 1.0; no GLM) |
| 2 | `gcn-pnc-sc-vwm-glmdiag` (REUSE 360744) | `glm_diagonal`    | correct (π=e) |
| 3 | `gcn-pnc-sc-vwm-glmdiag-permsubj` | `glm_diagonal_permsubj` | scrambled, fresh π per subject |
| 4 | `gcn-pnc-sc-vwm-glmdiag-permfixed`| `glm_diagonal_permfixed`| scrambled, one shared π |

**Pre-registered significance rule (locked before reading results).** Primary:
ΔR² counts as real only if it exceeds the *larger* arm's fold std (≈0.11).
Secondary/tiebreak: corrected resampled t-test (`statistical_tests.py`,
ADR-0008), p<0.05. Degeneracy guard: if arm 1 (`identity`) scores R²≈0 on VWM
the matrix is uninformative — re-examine GCN/protocol before reading the GLM arms.

**Recipe notes / drift from the thesis text.**
- **Epochs:** thesis recipe says "300 epochs", but the reused arm 2 (360744)
  ran on the default `trainer.epochs=100` (unchanged since the initial commit).
  To stay comparable, arms 1/3/4 also inherit the default 100 (no override).
- **wandb project:** `slurm/train.sh` now hardcodes `logging.project="Baseline
  Launches"` (added after the 2026-05-29 batch). Arms 1/3/4 force
  `logging.project=orbitglm` via RUN_ARGS so they land with arm 2 (360744).
- `glm_normalize=true` on all GLM arms (baked into the perm presets), matching
  the 2026-05-29 batch. Under `per_subject` each column mixes nodes' values
  across subjects by construction — the intended corruption.

**Shared recipe.** `dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2`, `--time=2-00:00:00`, wandb entity `teampolpetta`.

### gcn-pnc-sc-vwm-identity
- **Date:** 2026-06-01
- **Description:** Pure one-hot identity baseline on VWM (no GLM channel) — the
  distinctness-prior floor the GLM-diagonal arms are predicted (H1) to track.
  New: the only prior `identity` baseline (360724) was on `age`, not VWM.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity` (+ shared recipe). glm_normalize n/a (no GLM).
- **Commit SHA (DEPLOY_SHA):** `c2e973ba9952d2054d43476ec270e0f969ee074b`
- **Job ID:** 360785 (`-J gcn-pnc-sc-vwm-identity`), gpunode01/rtx2080.
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** `cluster-submit --node gpunode01 --gpu rtx2080 slurm/train.sh -J gcn-pnc-sc-vwm-identity --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity features=identity dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-glmdiag (reused — correct binding, π=e)
- **Date:** 2026-05-29 (reuse job 360744, conditions unchanged).
- **Description:** GLM-diagonal with correct value↔node binding. Arm 2 of the
  decoupling matrix; reused because `glm_value_permute=none` is a no-op.
- **Results:** pearson_r 0.487 ± 0.057, r2 0.176 ± 0.112 (50 outer folds). See 2026-05-29 batch.

### gcn-pnc-sc-vwm-glmdiag-permsubj
- **Date:** 2026-06-01
- **Description:** GLM-diagonal, one-hot intact, GLM values permuted by a FRESH
  random π per subject (seeded from the subject id). Destroys usable
  value↔identity binding entirely.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=glm_diagonal_permsubj` (bakes
  `glm_value_permute=per_subject glm_permute_seed=0 glm_normalize=true`) (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `c2e973ba9952d2054d43476ec270e0f969ee074b`
- **Job ID:** 360783 (`-J gcn-pnc-sc-vwm-glmdiag-permsubj`), gpunode03/rtxa6000.
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** `cluster-submit --node gpunode03 --gpu rtxa6000 slurm/train.sh -J gcn-pnc-sc-vwm-glmdiag-permsubj --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-glmdiag-permsubj features=glm_diagonal_permsubj dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-glmdiag-permfixed
- **Date:** 2026-06-01
- **Description:** GLM-diagonal, one-hot intact, GLM values permuted by ONE
  shared fixed π across all subjects. Binding is wrong but stable (H3 probe:
  can the GCN learn to invert a consistent-but-wrong value→node code?).
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=glm_diagonal_permfixed` (bakes
  `glm_value_permute=fixed glm_permute_seed=0 glm_normalize=true`) (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `c2e973ba9952d2054d43476ec270e0f969ee074b`
- **Job ID:** 360784 (`-J gcn-pnc-sc-vwm-glmdiag-permfixed`), gpunode01/v100.
- **wandb run:** orbitglm/teampolpetta — TBD
- **Command:** `cluster-submit --node gpunode01 --gpu v100 slurm/train.sh -J gcn-pnc-sc-vwm-glmdiag-permfixed --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-glmdiag-permfixed features=glm_diagonal_permfixed dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

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
- wandb: project `orbitglm`, entity `teampolpetta` (run name == experiment_name).
- All 7 deployed from a single SHA: **`a1899eb`** on branch
  `feature/vwm-glm-presets`. Submitted 2026-05-29.

**Matrix.**

| # | experiment_name | Job ID | features preset | node_features | dim |
|---|---|---|---|---|---|
| 1 | `gcn-pnc-sc-vwm-glmdiag`          | 360744 | `glm_diagonal`              | `[glm_diagonal]`            | 400 |
| 2 | `gcn-pnc-sc-vwm-id-glmscalar`     | 360745 | `identity_glm_scalar`       | `[identity, glm_scalar]`    | 401 |
| 3 | `gcn-pnc-sc-vwm-id-glmdiag`       | 360746 | `identity_glm_diagonal`     | `[identity, glm_diagonal]`  | 800 |
| 4 | `gcn-pnc-sc-vwm-scprof-glmscalar` | 360747 | `scprofile_glm_scalar`      | `[sc_row, glm_scalar]`      | 401 |
| 5 | `gcn-pnc-sc-vwm-scprof-glmdiag`   | 360748 | `scprofile_glm_diagonal`    | `[sc_row, glm_diagonal]`    | 800 |
| 6 | `gcn-pnc-sc-vwm-lappe-glmscalar`  | 360749 | `laplacian_pe_glm_scalar`   | `[laplacian_pe, glm_scalar]`| 9   |
| 7 | `gcn-pnc-sc-vwm-lappe-glmdiag`    | 360750 | `laplacian_pe_glm_diagonal` | `[laplacian_pe, glm_diagonal]` | 408 |

---

### gcn-pnc-sc-vwm-glmdiag
- **Date:** 2026-05-29
- **Description:** Standalone GLM diagonal embedding baseline — GLM activation on each node's own diagonal cell, no carrier feature.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=glm_diagonal` (+ shared recipe; `features.glm_normalize=true` overrides this preset's default of `false`).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360744 (`-J gcn-pnc-sc-vwm-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-glmdiag` (direct URL registers on run start)
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-glmdiag features=glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-id-glmscalar
- **Date:** 2026-05-29
- **Description:** Identity one-hot carrier ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360745 (`-J gcn-pnc-sc-vwm-id-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-id-glmscalar` (direct URL registers on run start)
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-id-glmscalar --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-id-glmscalar features=identity_glm_scalar dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-id-glmdiag
- **Date:** 2026-05-29
- **Description:** Identity one-hot carrier ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360746 (`-J gcn-pnc-sc-vwm-id-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-id-glmdiag` (direct URL registers on run start)
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-id-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-id-glmdiag features=identity_glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-scprof-glmscalar
- **Date:** 2026-05-29
- **Description:** SC connectivity-profile carrier (`sc_row`) ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=scprofile_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360747 (`-J gcn-pnc-sc-vwm-scprof-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-scprof-glmscalar` (direct URL registers on run start)
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-scprof-glmscalar --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-scprof-glmscalar features=scprofile_glm_scalar dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-scprof-glmdiag
- **Date:** 2026-05-29
- **Description:** SC connectivity-profile carrier (`sc_row`) ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=scprofile_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360748 (`-J gcn-pnc-sc-vwm-scprof-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-scprof-glmdiag` (direct URL registers on run start)
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-scprof-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-scprof-glmdiag features=scprofile_glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-lappe-glmscalar
- **Date:** 2026-05-29
- **Description:** Laplacian PE carrier (k=8, listed first for sign-flip) ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=laplacian_pe_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360749 (`-J gcn-pnc-sc-vwm-lappe-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-lappe-glmscalar` (direct URL registers on run start)
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-lappe-glmscalar --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-lappe-glmscalar features=laplacian_pe_glm_scalar dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD

### gcn-pnc-sc-vwm-lappe-glmdiag
- **Date:** 2026-05-29
- **Description:** Laplacian PE carrier (k=8, listed first for sign-flip) ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=laplacian_pe_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360750 (`-J gcn-pnc-sc-vwm-lappe-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-lappe-glmdiag` (direct URL registers on run start)
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-lappe-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-lappe-glmdiag features=laplacian_pe_glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** TBD
