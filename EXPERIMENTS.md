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

## Batch 2026-06-03 — VWM GLM node-features RE-RUN (reproducibility audit)

**What.** A verbatim re-run of the [2026-05-29 batch](#batch-2026-05-29--vwm-glm-node-feature-encodings-7-run-matrix)
(same recipe, same 10×5 nested-CV protocol, same 20-trial inner HPO), on SHA
**`7cef2f6`** (branch `feature/per-run-checkpoint-dir` = `c80b62a` + **ADR-0012**
per-run checkpoint dir; no training-code change). Purpose: recover each run's
per-fold predictions — overwritten in the original batch — so **pooled r²** and the
ADR-0008 corrected test become computable. Jobs **360772–360778**, submitted
2026-06-01. Per-run artifacts at `checkpoints/<experiment_name>-<jobid>/`.

> **⚠️ Caveat — these numbers are a *second draw*, not a reproduction.** On
> byte-identical training code, identical seeds, and identical data splits, the
> re-run does **not** reproduce the 2026-05-29 numbers (mean-of-folds r² shifts by
> up to 0.083; rankings reshuffle). Cause: training is nondeterministic on GPU
> (`torch.use_deterministic_algorithms(True)` is not set → PyG scatter atomics),
> amplified by a noise-dominated single-holdout inner HPO that flips the selected
> config in 46/50 folds. The ADR-0008 corrected paired t-test finds **no
> significant difference** among the top conditions even within this single draw.
> **Do not read the ordering below as findings.** Full analysis:
> [`reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md`](reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md).

**Results** — both r² flavours (mean-of-folds = average of 50 fold-local r²s;
pooled = one r² over all 9,400 out-of-fold predictions vs the global mean). Sorted
by mean-of-folds r² (the tuning metric). `Δ` = re-run minus 2026-05-29 mean-of-folds r².

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Δ vs orig | Pearson r | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| `gcn-pnc-sc-vwm-id-glmdiag`       | 360774 | 0.218 ± 0.093 | 0.222 | +0.005 | 0.522 ± 0.046 | 0.488 ± 0.033 | 0.630 ± 0.038 |
| `gcn-pnc-sc-vwm-lappe-glmdiag`    | 360778 | 0.213 ± 0.081 | 0.217 | +0.037 | 0.509 ± 0.051 | 0.485 ± 0.025 | 0.632 ± 0.034 |
| `gcn-pnc-sc-vwm-glmdiag`          | 360772 | 0.194 ± 0.100 | 0.193 | +0.018 | 0.490 ± 0.090 | 0.492 ± 0.035 | 0.640 ± 0.053 |
| `gcn-pnc-sc-vwm-id-glmscalar`     | 360773 | 0.098 ± 0.144 | 0.104 | −0.014 | 0.400 ± 0.117 | 0.522 ± 0.042 | 0.675 ± 0.049 |
| `gcn-pnc-sc-vwm-scprof-glmdiag`   | 360776 | 0.020 ± 0.284 | 0.017 | −0.083 | 0.442 ± 0.075 | 0.551 ± 0.097 | 0.702 ± 0.104 |
| `gcn-pnc-sc-vwm-lappe-glmscalar`  | 360777 | −0.037 ± 0.096 | −0.037 | −0.010 | −0.029 ± 0.067 | 0.561 ± 0.032 | 0.727 ± 0.049 |
| `gcn-pnc-sc-vwm-scprof-glmscalar` | 360775 | −0.060 ± 0.302 | −0.058 | −0.042 | 0.309 ± 0.102 | 0.567 ± 0.091 | 0.730 ± 0.095 |

**Corrected significance (ADR-0008, on this batch's per-fold r²).** None of the key
pairs are significant: id-glmdiag vs glmdiag Δ0.024 (p≈0.73); id-glmdiag vs
lappe-glmdiag Δ0.004 (p≈0.93); diagonal vs scalar (identity) Δ0.120 (p≈0.11);
diagonal vs scalar (sc_row) Δ0.079 (p≈0.71). (p via normal approx; Student-t only
widens them.)

**What changed vs 2026-05-29.** The rankings reshuffled — `lappe-glmdiag` rose to
≈0.21 and now ties `id-glmdiag` and beats standalone `glmdiag`, contradicting the
original "laplacian_pe is a wash." `scprof-glmdiag` fell from 0.103 to 0.020. The
**scalar-form weakness** (sc_row/laplacian_pe scalar carriers near or below zero;
`lappe-glmscalar` dead) is the one effect that reproduces qualitatively. The
`±std` is dispersion across folds, **not** a standard error.

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

**Results summary** (sorted by mean-of-folds R², the tuning metric; N=50 outer
folds = 10 reps × 5 folds; pooled = one R² over all 9,400 out-of-fold
predictions vs the global mean). All 4 arms `COMPLETED`. `±` is dispersion across
folds, **not** a standard error.

| experiment_name | Job | value binding | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|---|
| `gcn-pnc-sc-vwm-glmdiag-permfixed` | 360784 | scrambled, one shared π | **0.194 ± 0.093** | 0.197 | 0.488 ± 0.073 | 0.493 ± 0.032 | 0.640 ± 0.039 | [2hds44mj](https://wandb.ai/teampolpetta/orbitglm/runs/2hds44mj) |
| `gcn-pnc-sc-vwm-glmdiag` (reuse 360744) | 360744 | correct (π=e) | 0.176 ± 0.112 | n/a¹ | 0.487 ± 0.057 | 0.499 ± 0.039 | 0.647 ± 0.044 | [97893t1o](https://wandb.ai/teampolpetta/orbitglm/runs/97893t1o) |
| `gcn-pnc-sc-vwm-glmdiag-permsubj` | 360783 | scrambled, fresh π per subject | 0.006 ± 0.082 | n/a¹ | 0.246 ± 0.088 | 0.551 ± 0.027 | 0.711 ± 0.035 | [1biyzrr7](https://wandb.ai/teampolpetta/orbitglm/runs/1biyzrr7) |
| `gcn-pnc-sc-vwm-identity` | 360785 | — (constant 1.0; no GLM) | −0.028 ± 0.181 | n/a¹ | 0.260 ± 0.115 | 0.560 ± 0.053 | 0.720 ± 0.059 | [3oyjdbbg](https://wandb.ai/teampolpetta/orbitglm/runs/3oyjdbbg) |

¹ Pooled R² is recoverable **only** for permfixed (360784). This batch ran on
the pre-ADR-0012 code (branch base `c80b62a`), which writes every run's per-fold
predictions to one shared `checkpoints/nested_cv_result.json`. The three new arms
ran concurrently and overwrote each other; only the last to finish (permfixed,
2026-06-02 12:02) survives. permsubj/identity predictions are gone, and the
reused arm 2 (360744) was overwritten back in the 2026-05-29 batch.

**Pre-registered rule outcomes.** ⚠️ **The degeneracy guard fired:** arm 1
`identity` scores R²≈0 (−0.028 ± 0.181). Per the locked rule, the matrix is to be
treated as *uninformative pending a GCN/protocol re-examination* — do not read the
ordering below as a confirmed finding. Caveat to the caveat: the GLM arms are
**not** globally degenerate (permfixed 0.194, glmdiag 0.176 on the same pipeline),
so this is a real "pure one-hot carries no VWM signal" floor, not a broken run.
Applying the primary ΔR²>fold-std rule anyway yields a clean dichotomy —
{permfixed ≈ glmdiag} ≫ {permsubj ≈ identity ≈ 0}:
- permfixed − identity = 0.222 > 0.181 (real); glmdiag − permsubj = 0.170 > 0.112 (real).
- permfixed − glmdiag = 0.018 (not real); permsubj − identity = 0.034 (not real).

**Provisional reading (treat as hypothesis, not result).** If the floor is
trusted, the signal lives in **cross-subject-consistent per-node value structure**,
not in the biologically-correct binding *or* the one-hot distinctness prior alone:
a fixed-but-wrong permutation (permfixed) matches correct binding, a per-subject
permutation (permsubj) collapses to the identity floor, and pure one-hot
(identity, no values) carries nothing. This **refutes H1 as stated** ("the
diagonal's signal is the one-hot distinctness prior") — distinctness alone (arm 1)
is ≈0. Hard caveats: (1) this pipeline is nondeterministic / not run-to-run
reproducible (see main-branch
[`reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md`](reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md)),
so this is a single draw; (2) the pre-registered ADR-0008 corrected resampled
t-test is **not computable** for the key pairs — only permfixed retained per-fold
predictions. Re-running the batch on the ADR-0012 per-run-checkpoint code (now on
`main`) would make both the corrected test and all-arm pooled R² recoverable.

### gcn-pnc-sc-vwm-identity
- **Date:** 2026-06-01
- **Description:** Pure one-hot identity baseline on VWM (no GLM channel) — the
  distinctness-prior floor the GLM-diagonal arms are predicted (H1) to track.
  New: the only prior `identity` baseline (360724) was on `age`, not VWM.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity` (+ shared recipe). glm_normalize n/a (no GLM).
- **Commit SHA (DEPLOY_SHA):** `c2e973ba9952d2054d43476ec270e0f969ee074b`
- **Job ID:** 360785 (`-J gcn-pnc-sc-vwm-identity`), gpunode01/rtx2080.
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-identity` — https://wandb.ai/teampolpetta/orbitglm/runs/3oyjdbbg
- **Command:** `cluster-submit --node gpunode01 --gpu rtx2080 slurm/train.sh -J gcn-pnc-sc-vwm-identity --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity features=identity dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.260 ± 0.115, r2 −0.028 ± 0.181 (mean-of-folds), r2 n/a (pooled — pre-ADR-0012, shared `checkpoints/` predictions overwritten by 360784), mae 0.560 ± 0.053, rmse 0.720 ± 0.059 (50 outer folds). **Degeneracy guard fires: identity floor ≈ 0.** [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/3oyjdbbg)

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
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-glmdiag-permsubj` — https://wandb.ai/teampolpetta/orbitglm/runs/1biyzrr7
- **Command:** `cluster-submit --node gpunode03 --gpu rtxa6000 slurm/train.sh -J gcn-pnc-sc-vwm-glmdiag-permsubj --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-glmdiag-permsubj features=glm_diagonal_permsubj dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.246 ± 0.088, r2 0.006 ± 0.082 (mean-of-folds), r2 n/a (pooled — pre-ADR-0012, shared `checkpoints/` predictions overwritten by 360784), mae 0.551 ± 0.027, rmse 0.711 ± 0.035 (50 outer folds). Per-subject scramble collapses to the identity floor. [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/1biyzrr7)

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
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-glmdiag-permfixed` — https://wandb.ai/teampolpetta/orbitglm/runs/2hds44mj
- **Command:** `cluster-submit --node gpunode01 --gpu v100 slurm/train.sh -J gcn-pnc-sc-vwm-glmdiag-permfixed --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-glmdiag-permfixed features=glm_diagonal_permfixed dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.488 ± 0.073, r2 0.194 ± 0.093 (mean-of-folds), r2 0.197 (pooled, N=9400), mae 0.493 ± 0.032, rmse 0.640 ± 0.039 (50 outer folds). A fixed-but-wrong binding ≈ correct binding. [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/2hds44mj)

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

**Results summary** (mean ± std over the 50 outer folds; sorted by R², the tuning
metric). All 7 jobs `COMPLETED`. Each row links to its wandb run.

| Rank | experiment_name | R² | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|
| 1 | `gcn-pnc-sc-vwm-id-glmdiag`       | **0.213 ± 0.101** | **0.517 ± 0.054** | 0.488 ± 0.038 | 0.632 ± 0.044 | [cfryaekb](https://wandb.ai/teampolpetta/orbitglm/runs/cfryaekb) |
| 2 | `gcn-pnc-sc-vwm-glmdiag`          | 0.176 ± 0.112 | 0.487 ± 0.057 | 0.499 ± 0.039 | 0.647 ± 0.044 | [97893t1o](https://wandb.ai/teampolpetta/orbitglm/runs/97893t1o) |
| 2 | `gcn-pnc-sc-vwm-lappe-glmdiag`    | 0.176 ± 0.109 | 0.463 ± 0.136 | 0.499 ± 0.036 | 0.646 ± 0.046 | [in632wvk](https://wandb.ai/teampolpetta/orbitglm/runs/in632wvk) |
| 4 | `gcn-pnc-sc-vwm-id-glmscalar`     | 0.112 ± 0.144 | 0.426 ± 0.073 | 0.518 ± 0.046 | 0.670 ± 0.057 | [ukyq0a3n](https://wandb.ai/teampolpetta/orbitglm/runs/ukyq0a3n) |
| 5 | `gcn-pnc-sc-vwm-scprof-glmdiag`   | 0.103 ± 0.146 | 0.453 ± 0.059 | 0.526 ± 0.054 | 0.675 ± 0.064 | [dk2e07ba](https://wandb.ai/teampolpetta/orbitglm/runs/dk2e07ba) |
| 6 | `gcn-pnc-sc-vwm-scprof-glmscalar` | −0.018 ± 0.233 | 0.313 ± 0.104 | 0.554 ± 0.071 | 0.717 ± 0.076 | [gh854kdc](https://wandb.ai/teampolpetta/orbitglm/runs/gh854kdc) |
| 7 | `gcn-pnc-sc-vwm-lappe-glmscalar`  | −0.027 ± 0.092 | −0.015 ± 0.075 | 0.559 ± 0.026 | 0.723 ± 0.044 | [wo201iy1](https://wandb.ai/teampolpetta/orbitglm/runs/wo201iy1) |

**Findings.** (1) `glm_diagonal` beats `glm_scalar` for *every* carrier — the
scalar form collapses to R²≈0 for the `sc_row` and `laplacian_pe` carriers, while
the diagonal embedding holds real signal. (2) `identity` is the only carrier that
improves on the standalone `glm_diagonal` baseline (0.213 vs 0.176); `sc_row`
hurts and `laplacian_pe` is a wash. (3) `laplacian_pe_glm_scalar` (#7) is dead
(r ≈ 0). **Winner: `identity_glm_diagonal`** (R² ≈ 0.21, r ≈ 0.52). Caveat: the
top three overlap within fold std, so the identity⊕diagonal lead over standalone
diagonal is not significant; the scalar collapses are.

> **Update (2026-06-03):** a verbatim re-run of this batch
> ([Batch 2026-06-03](#batch-2026-06-03--vwm-glm-node-features-re-run-reproducibility-audit))
> did **not** reproduce these numbers — the pipeline is nondeterministic and the
> rankings are not run-to-run robust; the ADR-0008 corrected test finds the top-three
> differences non-significant. Treat the ranking above as illustrative, not a finding.
> See [`reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md`](reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md).

---

### gcn-pnc-sc-vwm-glmdiag
- **Date:** 2026-05-29
- **Description:** Standalone GLM diagonal embedding baseline — GLM activation on each node's own diagonal cell, no carrier feature.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=glm_diagonal` (+ shared recipe; `features.glm_normalize=true` overrides this preset's default of `false`).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360744 (`-J gcn-pnc-sc-vwm-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-glmdiag` — https://wandb.ai/teampolpetta/orbitglm/runs/97893t1o
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-glmdiag features=glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.487 ± 0.057, r2 0.176 ± 0.112, mae 0.499 ± 0.039, rmse 0.647 ± 0.044 (50 outer folds). [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/97893t1o)

### gcn-pnc-sc-vwm-id-glmscalar
- **Date:** 2026-05-29
- **Description:** Identity one-hot carrier ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360745 (`-J gcn-pnc-sc-vwm-id-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-id-glmscalar` — https://wandb.ai/teampolpetta/orbitglm/runs/ukyq0a3n
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-id-glmscalar --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-id-glmscalar features=identity_glm_scalar dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.426 ± 0.073, r2 0.112 ± 0.144, mae 0.518 ± 0.046, rmse 0.670 ± 0.057 (50 outer folds). [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/ukyq0a3n)

### gcn-pnc-sc-vwm-id-glmdiag
- **Date:** 2026-05-29
- **Description:** Identity one-hot carrier ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360746 (`-J gcn-pnc-sc-vwm-id-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-id-glmdiag` — https://wandb.ai/teampolpetta/orbitglm/runs/cfryaekb
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-id-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-id-glmdiag features=identity_glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.517 ± 0.054, r2 0.213 ± 0.101, mae 0.488 ± 0.038, rmse 0.632 ± 0.044 (50 outer folds) — **best run**. [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/cfryaekb)

### gcn-pnc-sc-vwm-scprof-glmscalar
- **Date:** 2026-05-29
- **Description:** SC connectivity-profile carrier (`sc_row`) ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=scprofile_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360747 (`-J gcn-pnc-sc-vwm-scprof-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-scprof-glmscalar` — https://wandb.ai/teampolpetta/orbitglm/runs/gh854kdc
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-scprof-glmscalar --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-scprof-glmscalar features=scprofile_glm_scalar dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.313 ± 0.104, r2 −0.018 ± 0.233, mae 0.554 ± 0.071, rmse 0.717 ± 0.076 (50 outer folds) — scalar form collapses. [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/gh854kdc)

### gcn-pnc-sc-vwm-scprof-glmdiag
- **Date:** 2026-05-29
- **Description:** SC connectivity-profile carrier (`sc_row`) ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=scprofile_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360748 (`-J gcn-pnc-sc-vwm-scprof-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-scprof-glmdiag` — https://wandb.ai/teampolpetta/orbitglm/runs/dk2e07ba
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-scprof-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-scprof-glmdiag features=scprofile_glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.453 ± 0.059, r2 0.103 ± 0.146, mae 0.526 ± 0.054, rmse 0.675 ± 0.064 (50 outer folds). [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/dk2e07ba)

### gcn-pnc-sc-vwm-lappe-glmscalar
- **Date:** 2026-05-29
- **Description:** Laplacian PE carrier (k=8, listed first for sign-flip) ⊕ per-node GLM scalar value.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=laplacian_pe_glm_scalar` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360749 (`-J gcn-pnc-sc-vwm-lappe-glmscalar`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-lappe-glmscalar` — https://wandb.ai/teampolpetta/orbitglm/runs/wo201iy1
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-lappe-glmscalar --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-lappe-glmscalar features=laplacian_pe_glm_scalar dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r −0.015 ± 0.075, r2 −0.027 ± 0.092, mae 0.559 ± 0.026, rmse 0.723 ± 0.044 (50 outer folds) — dead run, no signal. [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/wo201iy1)

### gcn-pnc-sc-vwm-lappe-glmdiag
- **Date:** 2026-05-29
- **Description:** Laplacian PE carrier (k=8, listed first for sign-flip) ⊕ GLM diagonal embedding.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=laplacian_pe_glm_diagonal` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `a1899eb99bd5c33ab6c1d7b5423b2ba164524141`
- **Job ID:** 360750 (`-J gcn-pnc-sc-vwm-lappe-glmdiag`)
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-lappe-glmdiag` — https://wandb.ai/teampolpetta/orbitglm/runs/in632wvk
- **Command:** `cluster-submit slurm/train.sh -J gcn-pnc-sc-vwm-lappe-glmdiag --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-lappe-glmdiag features=laplacian_pe_glm_diagonal dataset=pnc model=gcn labels=pnc_VWMdprime features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2"`
- **Results:** pearson_r 0.463 ± 0.136, r2 0.176 ± 0.109, mae 0.499 ± 0.036, rmse 0.646 ± 0.046 (50 outer folds). [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/in632wvk)
