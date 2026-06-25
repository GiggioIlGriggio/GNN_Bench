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

## Batch 2026-06-24 — identity + glm_diagonal → VWM (protocol-matched mirror of the age run 362508)

**What.** The VWM half of the identity+glm_diagonal investigation (age arm: Batch
2026-06-23, directly below). The age run showed a clean identity anchor *rescues* the
multiplicatively-gated `glm_diagonal` carrier from the 0.093 floor to pooled **0.356**,
because SC topology carries real age signal (`identity`→age = 0.456). VWM is the control
where **SC topology is useless** (`identity`→VWM ≈ 0; Batch 2026-06-04). **Prediction:**
with no topology signal to recover, `identity_glm_diagonal`→VWM should land at the GLM
carrier alone (`glm_diagonal`→VWM ~0.18–0.22) — **no topology bonus from the anchor** —
the mirror of the age case.

1 nested run (job **362601**), `gpunode03`/rtxa6000, SHA **`755ac5b`** (main at submit),
wandb project **`age_vwm_transfer`** / entity `teampolpetta`. Smoke (job 362600,
`COMPLETED 0:0`) validated the path and confirmed the 800-d `[identity(400),
glm_diagonal(400)]` wiring first — first `GCNConv.lin` = **51,200** params (= 800 × 64).

**Protocol — byte-comparable to the age run 362508.** Identical recipe field-for-field
**except** `labels=pnc_default → pnc_VWMdprime` and `experiment_name`. Same 940 cohort
(`pnc_vwm_cohort.txt`), same `source_age_pinned.yaml` search space, same default
`hpo_metric=val_mae`, **target-stratified** folds (no `stratify_target`) — VWM here, age
there = the clean mirror.

**Results.** N=50 outer folds (10 reps × 5 folds); pooled = one r² over all **9,400**
out-of-fold predictions (940 × 10 reps) vs the global mean. `COMPLETED 0:0` (10h53m);
pooled recomputed from `checkpoints/idglmdiag-vwm-362601/nested_cv_result.json` matched
the training-log mean-of-folds (no drift).

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `idglmdiag-vwm` | 362601 | 0.243 ± 0.083 | **0.246** | 0.527 ± 0.050 | 0.478 ± 0.030 | 0.620 ± 0.036 | [qnz1eyc6](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/qnz1eyc6) |

- **Results:** pearson_r 0.527 ± 0.050, r2 0.243 ± 0.083 (mean-of-folds); r2 0.246,
  pearson_r 0.511 (pooled, N=9,400), mae 0.478 ± 0.030, rmse 0.620 ± 0.036 (50 outer
  folds). [wandb run](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/qnz1eyc6)

**Paired contrast (the mirror — same 940 PNC cohort, same recipe, only the target differs):**

| target | `identity` (topology) alone | `identity_glm_diagonal` | `glm_diagonal` alone | anchor's topology bonus |
|---|---|---|---|---|
| **age** (topology useful) | 0.456 | **0.356** | 0.093 | large — rescues ~+0.26 over the GLM floor |
| **VWM** (topology useless) | ≈0 | **0.246** | ~0.18–0.22 | ≈none — sits at the GLM carrier |

**Headline.**
1. **Prediction confirmed — the mirror holds.** `identity_glm_diagonal`→VWM = pooled
   **0.246** (mean-of-folds 0.243 ± 0.083). The identity anchor adds **no meaningful
   topology bonus** for VWM: there is no SC-topology signal to recover (`identity`→VWM ≈
   0), so the 0.246 is carried entirely by the GLM channel — ≈ `glm_diagonal`→VWM
   (~0.18–0.22, within one fold std). Contrast the age arm, where the same anchor lifts
   the carrier by ~0.26 (to 0.356) because topology *does* carry age signal.
2. **The anchor only recovers signal that exists in topology.** This is the clean
   mechanistic mirror of the first-layer gating story (age batch below): `diag(g)` gates
   the topology readout via `s ⊙ g`; restoring a clean identity diagonal re-exposes the
   topology descriptor `s = Âᵀ𝟙` — but that only *helps* when topology is target-relevant.
   Age: topology has 0.456 of signal → recovered to 0.356. VWM: topology has ≈0 → nothing
   to recover → the concat just reflects the GLM carrier.
3. **Consistent with the off-protocol historical id-glmdiag→VWM (0.213, job 360746, Batch
   2026-05-29).** The protocol-matched 0.246 lands slightly above, as expected from the
   `hpo_metric` switch (`val_mae` here vs `val_r2` then) + the 940-cohort change; the
   conclusion is unchanged.

**Caveats.** (i) Single draw; the pipeline is nondeterministic run-to-run (~±0.02–0.03).
(ii) The `glm_diagonal`→VWM comparator (~0.18–0.22) is on a *different* cohort/protocol,
so "≈ GLM carrier alone" rests primarily on the in-protocol `identity`→VWM ≈ 0 (no
topology to add), not on a byte-matched `glm_diagonal`→VWM baseline.

**Command.** `cluster-submit --node gpunode03 --gpu rtxa6000 slurm/train.sh -J
idglmdiag-vwm --time=2-00:00:00
"--export=ALL,RUN_ARGS=experiment_name=idglmdiag-vwm features=identity_glm_diagonal
features.glm_normalize=true dataset=pnc
dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt model=gcn
labels=pnc_VWMdprime trainer.search_space=configs/sweeper/source_age_pinned.yaml
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 logging.project=age_vwm_transfer"`.

---

## Batch 2026-06-23 — identity + glm_diagonal → age (does a clean identity anchor rescue the gated GLM carrier?)

**What.** Resolves the puzzle left by [Batch 2026-06-13](#batch-2026-06-13--pnc-agevwm-transfer-leakage-safe-nested-cv-arms-b1b4--a4a5--2-age-sources):
on the **same SC edges**, `identity`→age = **0.456** but `glm_diagonal`→age = **0.093**
(`a4_glm_to_age`, age-stratified = 0.106). Both carriers are N×N **diagonal** node
features (`src/datasets/feature_builder.py:230`/`:371`): identity = `diag(1)`
(subject-invariant), glm_diagonal = `diag(g)` (subject-varying per-node 2back-vs-0back
GLM contrast). First-layer mean-pool algebra explains the gap: with mean pooling the
identity readout reads the SC topology descriptor `s = Âᵀ𝟙` (→ 0.456); the
glm_diagonal readout reads the **Hadamard mask `s ⊙ g`** — the GLM map
*multiplicatively gates* the topology, and a GCN has no primitive to divide `g` back
out, so it collapses to the node-feature floor (~0.10 ≈ GLM→age). **Prediction:**
concatenating a *clean constant identity anchor* (`features=identity_glm_diagonal` =
`[identity(400), glm_diagonal(400)]`, 800-d) should restore topology access → pooled
r² **≈0.45 confirms** the gating story, **≈0.10 falsifies** it.

1 nested run (job **362508**), `gpunode02`/rtx3090, SHA **`755ac5b`** (main), wandb
project **`age_vwm_transfer`** / entity `teampolpetta`. Smoke (job 362507,
`COMPLETED 0:0`) validated the path first.

**Cohort.** Shares the **940-subject** allowlist
(`configs/subject_lists/pnc_vwm_cohort.txt`) so the number is on the *same subjects*
as `src_age_id` (identity→age), `a4_glm_to_age` (glm→age), and `scprofile_to_age`.

**Recipe.** `features=identity_glm_diagonal features.glm_normalize=true dataset=pnc
dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt model=gcn
labels=pnc_default trainer.search_space=configs/sweeper/source_age_pinned.yaml
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 logging.project=age_vwm_transfer`, `--time=2-00:00:00`, default
`hpo_metric=val_mae`. **Age-stratified** (no `stratify_target`) — mirrors
`a4_glm_to_age`; the same caveat as `scprofile_to_age` applies (shares the cohort
with `src_age_id` but **not** byte-for-byte paired folds, since `src_age_id` was
VWM-stratified).

**Results.** N=50 outer folds (10 reps × 5 folds); pooled = one r² over all
**9,400** out-of-fold predictions (940 × 10 reps) vs the global mean. `COMPLETED
0:0` (18h23m); pooled recomputed from
`checkpoints/idglmdiag-age-362508/nested_cv_result.json` matched the training-log
mean-of-folds (no drift).

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `idglmdiag-age` | 362508 | 0.357 ± 0.099 | **0.356** | 0.642 ± 0.049 | 2.050 ± 0.166 | 2.655 ± 0.228 | [62i08npt](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/62i08npt) |

- **Results:** pearson_r 0.642 ± 0.049, r2 0.357 ± 0.099 (mean-of-folds); r2 0.356,
  pearson_r 0.615 (pooled, N=9,400), mae 2.050 ± 0.166, rmse 2.655 ± 0.228 (50 outer
  folds). [wandb run](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/62i08npt)

**Comparison (→ age, same 940 PNC cohort):**

| node-feature scheme → age | R² (pooled) | where SC lives | source |
|---|---|---|---|
| classical MLP on raw SC (no graph) | ≈0.50 | input vector | Batch 2026-06-11 |
| identity (one-hot nodes) — `src_age_id` | 0.456 | **edges** | Batch 2026-06-13 |
| connectivity_profile_sc — `scprofile_to_age` | ~~0.357~~ → **0.445** (corrected, 362627) | node features | Batch 2026-06-23 (scprofile) |
| **identity + glm_diagonal** (this run) | **0.356** | **edges** + node feat (GLM) | this batch |
| glm_diagonal — `a4_glm_to_age` | 0.106 | node features (GLM, not SC) | Batch 2026-06-13 |

**Headline.**
1. **Mechanism confirmed (direction).** The clean identity anchor lifts the GLM
   carrier from the 0.093/0.106 floor to pooled **0.356** — a ~3.4–3.8× jump, nowhere
   near the "≈0.10 falsifies" floor. The SC topology was **always extractable** via
   the edges; `glm_diagonal`-alone collapsed only because `diag(g)` multiplicatively
   gates the topology readout (`s ⊙ g`), which the GCN cannot invert. A constant
   identity diagonal restores it. The first-layer algebra holds.
2. **But concatenation *dilutes* — it does not stack.** 0.356 lands **~0.10 *below***
   pure identity→age (0.456), missing the "≈0.45 full-recovery" target. The GLM
   channel *competes with* the topology channel rather than adding its ~0.10 of
   standalone age signal on top: extra 400 subject-varying, weakly-age-relevant
   columns + shared first-layer/BatchNorm capacity cost more than they bring.
3. **The dilution is GLM-channel-specific, not "any dense node feature."**
   *(Corrected 2026-06-25.)* Originally read as `identity_glm_diagonal` (0.356) ≈
   `scprofile_to_age` (0.357), both diluting ~0.10 below identity — but the scprofile
   0.357 was an `lr=0.005` artifact; corrected it is **0.445 ≈ identity 0.456** (job
   362627, see the scprofile batch). So **only the GLM node-feature channel dilutes**:
   dumping the *SC row* into node features is neutral vs SC-in-edges (the GCN reads
   the same topology either way), whereas the per-node GLM-contrast channel genuinely
   competes with the topology readout (−0.10). SC-in-edges + one-hot nodes remains the
   strongest scheme; the **GLM channel** — not dense node features per se — is what
   subtracts. (`idglmdiag` itself is unaffected by the lr bug: it picked `lr=0.005` on
   24/50 folds with no divergence, group mean +0.336.)

**Caveats.** (i) `src_age_id`'s 0.456 was **VWM-stratified**; this run is
age-stratified → shares the cohort but not byte-for-byte paired folds. Stratification
moved glm→age by only +0.013 (a4 0.106 vs src_age_glm 0.093), so the ~0.10 dilution
gap is real, not a stratification artifact. (ii) Single draw; pipeline is
nondeterministic run-to-run (~±0.02–0.03). (iii) An *edge-ablated* `glm_diagonal→age`
(message passing removed → predicts ≈0.10, direct proof "just node features") is the
natural next control — not yet run.

**Command.** `cluster-submit --node gpunode02 --gpu rtx3090 slurm/train.sh -J
idglmdiag-age --time=2-00:00:00
"--export=ALL,RUN_ARGS=experiment_name=idglmdiag-age features=identity_glm_diagonal
features.glm_normalize=true dataset=pnc
dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt model=gcn
labels=pnc_default trainer.search_space=configs/sweeper/source_age_pinned.yaml
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 logging.project=age_vwm_transfer"`.

---

## Batch 2026-06-23 — connectivity-profile-SC → age (SC row AS node features)

**What.** Does pushing the **SC connectivity row into the node features**
(`features=connectivity_profile_sc`: each node = its row of the N×N SC matrix) help
predict **age** through a GCN, versus the existing GNN baseline where SC is carried
only in the **edges** with one-hot node identity (`src_age_id`, identity→age =
0.456, [Batch 2026-06-13](#batch-2026-06-13--pnc-agevwm-transfer-leakage-safe-nested-cv-arms-b1b4--a4a5--2-age-sources))?
A standalone, age-stratified control on the shared 940-subject PNC cohort. The VWM
node-feature matrices found the `scprofile` carrier the *weakest*; this asks the
same on age, where connectivity is strongly age-predictive (classical SC→age ≈0.50,
[Batch 2026-06-11](#batch-2026-06-11--classical-ml-baselines-xgboost--elasticnet--mlp-vs-the-gnn-30-run-matrix)).

1 nested run (job **362510**), `gpunode02`/rtx6000, SHA **`755ac5b`** (main), wandb
project **`age_vwm_transfer`** / entity `teampolpetta`. Smoke (job 362509,
`COMPLETED 0:0`) validated the path first.

> **⚠ Corrected 2026-06-25 — the original 362510 number was an `lr=0.005` sweep
> artifact; use the 362627 numbers below.** `source_age_pinned.yaml` offered
> `trainer.lr=0.005`, which on the *dense 400-d* `connectivity_profile_sc` node
> features drives a scale-collapse / divergent refit: the **6/50** folds that picked
> it went low-biased + variance-compressed (Pearson preserved 0.57–0.69, but r²
> −2.48…+0.05, group mean **−0.53**), dragging the whole run down. Dropping `0.005`
> from the source sweep (commit `0b3c773`; mirrors the transfer-sweeper fix
> `80e61ac`) and re-running (**job 362627**, `gpunode02`/rtx3090, branch
> `feature/backfill-scprofile-age`) removes it: pooled r² **0.357 → 0.445**, fold-SD
> **0.513 → 0.199**, worst fold **−2.48 → −0.29**, **0** folds at `lr=0.005`. The
> divergence is **feature-specific** — identity (`src_age_id`, 4/50 folds) and
> `identity_glm_diagonal` (362508, 24/50 folds) both picked `0.005` with **no**
> divergence; only the dense raw-SC rows are sensitive.

**Cohort.** Shares the **940-subject** allowlist
(`configs/subject_lists/pnc_vwm_cohort.txt`, the age→VWM-transfer cohort) so the
number is on the *same subjects* as `src_age_id` (identity→age) and `a4_glm_to_age`
(glm→age). `connectivity_profile_sc` uses no GLM map (`glm_contrasts: []`), so the
cohort's GLM constraint is irrelevant here.

**Recipe.** `features=connectivity_profile_sc dataset=pnc
dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt model=gcn
labels=pnc_default trainer.search_space=configs/sweeper/source_age_pinned.yaml
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 logging.project=age_vwm_transfer`, `--time=2-00:00:00`, default
`hpo_metric=val_mae`. **Age-stratified** (no `stratify_target`) — a standalone age
estimate, unlike `src_age_id` which was VWM-stratified to align with the transfer
arms (so the contrast below shares the cohort but **not** byte-for-byte paired folds).

**Results.** N=50 outer folds (10 reps × 5 folds); pooled = one r² over all
**9,400** out-of-fold predictions (940 × 10 reps) vs the global mean. Both runs
`COMPLETED 0:0`; pooled recomputed from each run's `nested_cv_result.json` matched
the training-log mean-of-folds (no drift).

| experiment_name | Job | sweep lr | R² (mean-of-folds) | R² (pooled) | Pearson r (pooled) | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|---|
| **`scprofile_to_age_safelr`** ✅ corrected | **362627** | {1e-4,5e-4,1e-3} | **0.446 ± 0.199** | **0.445** | 0.681 | 1.875 ± 0.348 | 2.440 ± 0.405 | [au4tkuq4](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/au4tkuq4) |
| `scprofile_to_age` ✗ buggy (superseded) | 362510 | {…,5e-3} | 0.357 ± 0.518 | 0.357 | 0.643 | 1.995 ± 0.713 | 2.561 ± 0.737 | [ss1dilnf](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/ss1dilnf) |

- **Corrected (362627):** pooled r² **0.445** (mean-of-folds 0.446 ± **0.199**),
  pooled pearson 0.681 (0.736 ± 0.041 mean-of-folds), mae 1.875 ± 0.348, rmse
  2.440 ± 0.405. **Zero** folds at `lr=0.005`; worst fold **−0.29** (was −2.48),
  fold-SD **0.199** — matching identity→age's 0.195. 3/50 mild negatives, all at
  normal lr with high Pearson (ordinary hard folds, not divergence).
  [wandb run](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/au4tkuq4)

**Comparison (→ age, same 940 PNC cohort):**

| node-feature scheme → age | R² (pooled) | where SC lives | source |
|---|---|---|---|
| classical MLP on raw SC (no graph) | ≈0.50 | input vector | Batch 2026-06-11 |
| identity (one-hot nodes) — `src_age_id` | 0.456 | **edges** | Batch 2026-06-13 |
| **connectivity_profile_sc** — `scprofile_to_age_safelr` (corrected, 362627) | **0.445** | **node features** | this batch |
| glm_diagonal — `a4_glm_to_age` | 0.106 | node features (GLM, not SC) | Batch 2026-06-13 |

**Headline.** *(Corrected 2026-06-25 — see the ⚠ note above.)* Putting the SC row
**into the node features** predicts age **on par with** carrying SC in the **edges**
with one-hot nodes — corrected pooled r² **0.445** (362627) ≈ identity→age **0.456**
(a 0.011 gap, well within the pipeline's run-to-run ±0.02–0.03 noise), and far above
the GLM vector (0.106). The original *"adds noise, not signal"* reading (0.357 <
0.456) was an `lr=0.005` divergence artifact, **not** a real deficit:
SC-as-node-features is **neutral** vs SC-as-edges — the GCN extracts the same
age-relevant structure from edge topology either way, and the dense 400-d SC node
vector neither adds nor subtracts once the bad-lr folds are gone. A non-graph MLP on
the raw SC matrix (≈0.50) still edges out both graph schemes. The fold distribution
is now well-behaved (r² −0.29…+0.64, fold-SD 0.199 ≈ identity's 0.195). **Caveat:**
age-stratified vs `src_age_id`'s VWM-stratified folds → shares the cohort but not
byte-for-byte paired; an age-stratified identity→age run would tighten the ≈.

**Command (corrected run 362627).** `cluster-submit --node gpunode02 --gpu rtx3090
slurm/train.sh -J scprofile-age-safelr --time=2-00:00:00
"--export=ALL,RUN_ARGS=experiment_name=scprofile_to_age_safelr
features=connectivity_profile_sc dataset=pnc
dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt model=gcn
labels=pnc_default trainer.search_space=configs/sweeper/source_age_pinned.yaml
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 logging.project=age_vwm_transfer"` — identical to the original
362510 recipe except the `source_age_pinned.yaml` sweep no longer offers `lr=0.005`
(commit `0b3c773`). **Avoid gpunode01** (it silently accepted-but-never-scheduled an
earlier smoke); `cluster-tail` blocks on a missing log here, so use `cluster-fetch`
for logs.

---

## Batch 2026-06-22 — PNC age→PCPT transfer (identity carrier; arms B-fullFT/B-frozen + A3/frozen-random/A5)

**What.** Does an **age-pretrained identity backbone** transfer to **PCPT accuracy**
(`PCPT_precision`) on PNC, under the same leakage-safe nested-CV transfer protocol as
the age→VWM batch (per-outer-fold backbone injection + a fail-closed fold-alignment
guard; infra PRs **#37 + #38**)? One **identity** age "source" backbone (job 362451)
is pretrained, then loaded into PCPT runs as either a full fine-tune (`from_age_ft`)
or a frozen backbone + fresh head (`from_age_frozen`). PCPT is **~5× more age-coupled
than VWM** (corr(age,PCPT)=0.533 ⇒ closed-form age floor **r²=0.284** vs VWM's 0.057),
so the controls are load-bearing: **A3** from-scratch, **frozen-random** (random frozen
backbone, no checkpoint) and **A5** (closed-form age floor). Sibling of Batch
2026-06-13 (age→VWM); single identity carrier (no GLM arm).

5 nested runs (jobs **362451–362455**), `gpunode02`/`rtx6000`, SHA **`f6d8c4f`**
(feature/age-pcpt-transfer), wandb project **`age_pcpt_transfer`** / entity
`teampolpetta`.

**Cohort (critical).** All arms share ONE **937-subject** TRITASK allowlist (graph ∩
non-NaN VWM ∩ non-NaN PCPT ∩ GLM map = VWM-940 − 3), generated **on the cluster**
(`slurm/make_cohort.sh` with the `also_require=pnc_PCPT_accuracy` 4th positional).
Identical subject set + PCPT-stratified outer folds across the source and every PCPT
arm → the `SourceBackboneProvider` guard passes (no `fold-index mismatch`). The source
stratifies via `stratify_target=PCPT_precision` (target stays age); the PCPT arms
stratify on their own target by default.

**Cluster CSV gotcha.** On the cluster `Tabular_data/PNC_ALL_SCORES.csv` lacks
`PCPT_precision` (only `…_PCPT.csv` has it; identical on the laptop, NOT on the
cluster). So the age source uses **`labels=pnc_age_pcptcsv`** (age target, PCPT
metadata) — *not* `pnc_default`, which `KeyError`s on the stratify column — and the
PCPT arms use `labels=pnc_PCPT_accuracy`. (See [cluster_pcpt_csv_split].)

**Shared recipe.** `dataset=pnc
dataset.subject_list_file=configs/subject_lists/pnc_tritask_cohort.txt model=gcn
features=identity trainer.n_repetitions=10 trainer.n_outer_folds=5
trainer.inner_hpo_trials=20 trainer.epochs=300 logging.project=age_pcpt_transfer`,
default `hpo_metric=val_mae`, wandb entity `teampolpetta`.
- **Source / A3:** `trainer.search_space=configs/sweeper/source_age_pinned.yaml`.
  Source adds `labels=pnc_age_pcptcsv stratify_target=PCPT_precision`; A3 is
  `labels=pnc_PCPT_accuracy transfer=none` (from scratch, same cohort + stratification).
- **B-arms / frozen-random:** `labels=pnc_PCPT_accuracy
  trainer.search_space=configs/sweeper/transfer_finetune.yaml` (lr grid
  1e-4/5e-4/1e-3 — **5e-3 dropped**; it diverged the identity full-FT in the VWM
  batch, see [age_vwm_transfer_dynamics]). B-arms add `transfer=<from_age_ft|from_age_frozen>
  transfer.source_checkpoint_root=checkpoints/src_age_id_pcpt-362451`; frozen-random
  is `transfer=frozen_random` (random frozen backbone, no checkpoint).

| arm | experiment_name | Job | transfer | source ckpt | frozen |
|---|---|---|---|---|---|
| B-fullFT | `b_fullft_pcpt` | 362452 | from_age_ft | src_age_id_pcpt-362451 | `[]` |
| B-frozen | `b_frozen_pcpt` | 362453 | from_age_frozen | src_age_id_pcpt-362451 | `['backbone']` |
| frozen-random | `frozen_random_pcpt` | 362454 | frozen_random | — (random) | `['backbone']` |
| A3 | `a3_scratch_pcpt` | 362455 | none | — | — |
| src | `src_age_id_pcpt` | 362451 | — (writes ckpt) | — | — |

**Results.** N=50 outer folds (10 reps × 5 folds); pooled = one r² over all **9,370**
out-of-fold predictions (937 subjects × 10 reps) vs the global mean → effective
**N=937**. `±` = dispersion across (correlated) folds, **not** a standard error. All 5
`COMPLETED 0:0`; each pooled run's recomputed mean-of-folds r² matched its
training-log value (no drift). Recovered from per-run
`checkpoints/<name>-<jobid>/nested_cv_result.json` (ADR-0012) + wandb (project
**`age_pcpt_transfer`**, *not* the `.cluster-helper.yaml` default `orbitglm` — see
[wandb_project_default_drift]).

*Transfer arms + controls → PCPT* (sorted by r²):

| arm | mode | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|---|
| **B-frozen** | age-id / frozen | 362453 | **0.154 ± 0.058** | **0.151** | 0.407 ± 0.067 | 0.080 ± 0.004 | 0.110 ± 0.012 | [gxg66rlo](https://wandb.ai/teampolpetta/age_pcpt_transfer/runs/gxg66rlo) |
| **B-fullFT** | age-id / full-FT | 362452 | 0.103 ± 0.107 | 0.101 | 0.394 ± 0.082 | 0.081 ± 0.005 | 0.113 ± 0.013 | [ywie9skh](https://wandb.ai/teampolpetta/age_pcpt_transfer/runs/ywie9skh) |
| frozen-random | random / frozen | 362454 | 0.058 ± 0.051 | 0.058 | 0.282 ± 0.078 | 0.084 ± 0.004 | 0.116 ± 0.011 | [7cshnaq7](https://wandb.ai/teampolpetta/age_pcpt_transfer/runs/7cshnaq7) |
| A3 | from scratch | 362455 | −0.065 ± 0.595 | −0.053 | 0.361 ± 0.084 | 0.088 ± 0.024 | 0.120 ± 0.026 | [bni3jjbj](https://wandb.ai/teampolpetta/age_pcpt_transfer/runs/bni3jjbj) |
| *A5* | age only, no graph | — | — | **0.284** | 0.533 | — | — | closed-form |

*Age source* (target = age, years):

| run | desc | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|---|
| `src_age_id_pcpt` | identity→age (source for B-arms), PCPT-stratified | 362451 | 0.481 ± 0.152 | 0.482 | 0.755 ± 0.035 | 1.806 ± 0.297 | 2.362 ± 0.336 | [reazqw8x](https://wandb.ai/teampolpetta/age_pcpt_transfer/runs/reazqw8x) |

**Corrected significance (ADR-0008).** Bouckaert–Frank corrected resampled paired
t-test on per-outer-fold r² (`src/training/statistical_tests.py`, the
`scripts/compare_models.py` test; BH across all 6 pairs of the 4 PCPT arms; folds are
shared — same cohort + PCPT stratification):

| A | B | Δr² (A−B) | p_raw | q (BH) | sig |
|---|---|---|---|---|---|
| B-frozen | frozen-random | +0.096 | 0.0003 | **0.002** | **yes** |
| B-frozen | B-fullFT | +0.051 | 0.246 | 0.691 | n.s. |
| B-frozen | A3 | +0.219 | 0.484 | 0.691 | n.s. |
| B-fullFT | frozen-random | +0.045 | 0.382 | 0.691 | n.s. |
| B-fullFT | A3 | +0.168 | 0.598 | 0.691 | n.s. |
| frozen-random | A3 | +0.123 | 0.691 | 0.691 | n.s. |

Only **B-frozen > frozen-random** survives correction (q=0.002). Contrasts against A3
carry large Δr² but stay n.s. — A3's per-fold r² std is **0.595** (from-scratch
identity diverges on some folds, like B1 in the VWM batch), which destroys the power
of any A3 contrast.

**Headline.**
1. **The age-pretrained backbone carries PCPT signal beyond a random projection.**
   B-frozen (pooled **0.151**) significantly beats frozen-random (0.058; q=0.002, the
   only surviving contrast) — the age *training*, not just any random frozen
   projection, encodes PCPT-relevant structure (head-only, backbone untouched).
2. **…but it does NOT beat chronological age.** Every transfer arm sits **below** the
   A5 age floor (B-frozen 0.151, B-fullFT 0.101 vs **0.284** pooled). Unlike age→VWM
   (where the frozen GLM backbone beat its 0.057 floor ~2×), PCPT's ~5× higher
   age-coupling means a closed-form age regression outpredicts the transferred GNN
   representation. The backbone carries *age-correlated* PCPT variance — less
   efficiently than age itself.
3. **Frozen > full-FT (point estimate); neither beats scratch significantly.**
   Ordering B-frozen (0.151) > B-fullFT (0.101) > frozen-random (0.058) > A3 (−0.053),
   but no contrast among {B-frozen, B-fullFT, A3} survives correction — A3's fold
   instability (std 0.595) under-powers the transfer-vs-scratch test. Consistent with
   the VWM batch: full fine-tuning the identity backbone is unstable and shows no
   transfer bonus over frozen.

**Caveats.** (i) A3 (from-scratch identity) has extreme fold variance (r² std 0.595 —
diverges on some folds), so transfer-vs-scratch contrasts are descriptive, not
powered. (ii) B-vs-A5 is descriptive (A5 is closed-form, no per-fold array). (iii)
Frozen / random-frozen arms ran *longer* than full-FT (frozen ~12h, frozen-random
~23h vs full-FT ~8h) — head-only converges slower / early-stops less (as in the VWM
batch). (iv) Nondeterminism unaddressed (single draw per arm). (v) A5=0.284 computed
on the 937 cohort (corr +0.533); not recomputed byte-exact on the cluster split.

---

## Batch 2026-06-13 — PNC age→VWM transfer (leakage-safe nested-CV; arms B1–B4 + A4/A5 + 2 age sources)

**What.** Does a backbone **pretrained to predict age** transfer to **VWM**
(`VWM_overall_dprime`) on PNC, under a leakage-safe nested-CV transfer protocol
(per-outer-fold backbone injection + a fail-closed fold-alignment guard; infra
PRs **#37 + #38**)? Two age "source" backbones are pretrained (one per carrier),
then loaded into VWM runs as either a full fine-tune (`from_age_ft`) or a frozen
backbone + fresh head (`from_age_frozen`). Carriers: **identity** (node id → the
GCN reads the full SC connectivity) and **glm_diagonal** (node features = the
per-node 2back-vs-0back GLM contrast). Arms:
- **B1/B2** transfer the **identity** age-source (job 361349); B1 full-FT, B2 frozen.
- **B3/B4** transfer the **glm_diagonal** age-source (job 361350); B3 full-FT, B4 frozen.
- **A4** = GLM→age specificity control (does the GLM vector carry *age*?), age-stratified.
- **A5** = closed-form age→VWM floor (no graph), `r²=corr(age,VWM)²` = **0.057** (corr=0.239).

7 nested runs (jobs **361349–361355**), `gpunode02`, SHA **`20be17b`** (main, post
#37+#38), wandb project **`age_vwm_transfer`** / entity `teampolpetta`.

**Cohort (critical).** All arms share ONE **940-subject** allowlist (graph ∩
non-NaN VWM ∩ GLM map = the binding constraint), generated **on the cluster**
(`slurm/make_cohort.sh`, job 361348 — identical count to local). Identical subject
set + VWM-stratified outer folds across the source and every VWM arm → the
`SourceBackboneProvider` guard passes (no `fold-index mismatch`). Each B-arm's
`features=` equals its source carrier (else the "did not fully load" guard fires).
Without the shared cohort the age source (target near-universal, ~9.5k) and the VWM
run (~1.5k) load different subjects and never align.

**Shared recipe.** `dataset=pnc
dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt model=gcn
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 logging.project=age_vwm_transfer`, `--time=2-00:00:00`, default
`hpo_metric=val_mae`, wandb entity `teampolpetta`.
- **Sources / A4:** `labels=pnc_default
  trainer.search_space=configs/sweeper/source_age_pinned.yaml`. Sources add
  `stratify_target=VWM_overall_dprime` (VWM-stratified folds so VWM arms align);
  A4 omits it (age-stratified standalone control).
- **B-arms:** `labels=pnc_VWMdprime transfer=<from_age_ft|from_age_frozen>
  transfer.source_checkpoint_root=checkpoints/<src_age_id-361349|src_age_glm-361350>
  trainer.search_space=configs/sweeper/transfer_finetune.yaml` (optimizer + head
  only — backbone shape frozen by the loaded weights). No `stratify_target` (the
  VWM target's default stratification matches the source's VWM stratification).

| arm | experiment_name | Job | carrier (`features=`) | source ckpt | transfer | frozen |
|---|---|---|---|---|---|---|
| B1 | `b1_age_vwm` | 361352 | identity     | src_age_id-361349  | from_age_ft     | `[]` |
| B2 | `b2_age_vwm` | 361353 | identity     | src_age_id-361349  | from_age_frozen | `['backbone']` |
| B3 | `b3_age_vwm` | 361354 | glm_diagonal | src_age_glm-361350 | from_age_ft     | `[]` |
| B4 | `b4_age_vwm` | 361355 | glm_diagonal | src_age_glm-361350 | from_age_frozen | `['backbone']` |
| A4 | `a4_glm_to_age` | 361351 | glm_diagonal | — (standalone) | — | — |
| src | `src_age_id`  | 361349 | identity     | — (writes ckpt) | — | — |
| src | `src_age_glm` | 361350 | glm_diagonal | — (writes ckpt) | — | — |

**Results.** N=50 outer folds (10 reps × 5 folds); pooled = one r² over all
**9,400** out-of-fold predictions (940 subjects × 10 reps) vs the global mean →
effective **N=940**. `±` = dispersion across (correlated) folds, **not** a standard
error. All 7 `COMPLETED 0:0`; each pooled run's recomputed mean-of-folds r² matched
its training-log value (no drift). Recovered from per-run
`checkpoints/<name>-<jobid>/nested_cv_result.json` (ADR-0012) + wandb (project
**`age_vwm_transfer`**, *not* the `.cluster-helper.yaml` default `orbitglm` — see
[wandb_project_default_drift]).

*Transfer arms → VWM* (sorted by r²):

| arm | carrier / mode | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|---|
| **B3** | glm_diagonal / full-FT | 361354 | **0.195 ± 0.074** | **0.197** | 0.495 ± 0.050 | 0.495 ± 0.028 | 0.640 ± 0.039 | [rhr7vx35](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/rhr7vx35) |
| **B4** | glm_diagonal / frozen | 361355 | 0.110 ± 0.062 | 0.112 | 0.346 ± 0.082 | 0.520 ± 0.026 | 0.673 ± 0.039 | [6ipvo2tm](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/6ipvo2tm) |
| **B2** | identity / frozen | 361353 | 0.043 ± 0.037 | 0.044 | 0.230 ± 0.067 | 0.538 ± 0.022 | 0.698 ± 0.035 | [qwrnzw4x](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/qwrnzw4x) |
| **B1** | identity / full-FT | 361352 | −0.039 ± 0.266 | −0.035 | 0.322 ± 0.060 | 0.563 ± 0.082 | 0.723 ± 0.084 | [f5w5nk82](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/f5w5nk82) |
| *A5* | age only, no graph | — | — | **0.057** | 0.239 | — | — | closed-form |

*Age sources + GLM→age control* (target = age, years):

| run | desc | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|---|
| `src_age_id`    | identity→age (source for B1/B2) | 361349 | 0.451 ± 0.197 | 0.456 | 0.753 ± 0.046 | 1.863 ± 0.308 | 2.418 ± 0.386 | [z4dq2y2d](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/z4dq2y2d) |
| `a4_glm_to_age` | GLM→age control (age-stratified) | 361351 | 0.106 ± 0.076 | 0.106 | 0.441 ± 0.051 | 2.476 ± 0.120 | 3.135 ± 0.172 | [wibzuma2](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/wibzuma2) |
| `src_age_glm`   | glm_diagonal→age (source for B3/B4) | 361350 | 0.089 ± 0.156 | 0.093 | 0.436 ± 0.061 | 2.497 ± 0.237 | 3.148 ± 0.285 | [e7vx47l0](https://wandb.ai/teampolpetta/age_vwm_transfer/runs/e7vx47l0) |

**Corrected significance (ADR-0008).** Bouckaert-Frank corrected resampled paired
t-test on per-outer-fold r² (`scripts/compare_models.py`, BH across the 6 B-arm
pairs; folds are shared across arms — same cohort + VWM stratification):

| A | B | Δr² (A−B) | p_raw | q (BH) | sig |
|---|---|---|---|---|---|
| B3 GLM-ft | B4 GLM-frozen | +0.085 | 0.061 | 0.142 | n.s. |
| B2 ID-frozen | B3 GLM-ft | −0.152 | 0.0005 | **0.003** | **yes** |
| B2 ID-frozen | B4 GLM-frozen | −0.068 | 0.071 | 0.142 | n.s. |
| B1 ID-ft | B3 GLM-ft | −0.235 | 0.112 | 0.168 | n.s. |
| B1 ID-ft | B4 GLM-frozen | −0.150 | 0.287 | 0.344 | n.s. |
| B1 ID-ft | B2 ID-frozen | −0.082 | 0.567 | 0.567 | n.s. |

Only **GLM-full-FT > ID-frozen** survives correction (q=0.003); the GLM
full-FT-vs-frozen gap trends positive but n.s. (q=0.14). B1 (ID full-FT) is
unstable (fold r² std 0.27 — full fine-tuning the identity backbone diverges on
some folds), which kills the power of any B1 contrast.

**Headline.**
1. **The GLM carrier transfers; the identity carrier doesn't.** GLM arms (B3 0.197,
   B4 0.112 pooled) sit well above the identity arms (B1 ≈0, B2 0.044); the only
   corrected-significant contrast is GLM-ft > ID-frozen (q=0.003). Identity→VWM
   stays at the ≈0 floor seen before (see Batch 2026-06-04 identity→VWM).
2. **A frozen age-pretrained GLM backbone beats age alone (~2×).** B4 (frozen,
   head-only) reaches pooled r² **0.112** vs the A5 age-only floor **0.057** — the
   age-trained GLM representation carries VWM-relevant structure beyond chronological
   age (head-only, no backbone update).
3. **Full fine-tuning ≈ from-scratch GLM→VWM (no transfer bonus, no harm).** B3
   (0.197) lands on the prior from-scratch GLM-diagonal→VWM plateau (≈0.18–0.22 on
   the ~940 GLM cohort; Batches 2026-05-29 / 2026-06-04). Under full-FT the age init
   is overwritten — the gain is intrinsic to the GLM features, not the age
   pretraining. (No same-batch A3 from-scratch arm; this contrast is vs prior batches.)
4. **GLM features barely encode age (specificity holds).** A4 GLM→age pooled 0.106
   ≈ src_age_glm 0.093, ~4–5× below identity/SC→age (src_age_id 0.456). The GLM
   2back-vs-0back contrast is VWM-specific, not a generic maturation signal —
   consistent with Batch 2026-06-12 (GLM→age specificity control).

**Caveats.** (i) No A3 (from-scratch VWM, same cohort) in this batch → "transfer vs
scratch" leans on prior GLM→VWM numbers, not a same-batch arm. (ii) Significance is
B-arm-vs-B-arm; B-vs-A5 is descriptive (A5 is closed-form, no per-fold array — use
B4-vs-floor qualitatively). (iii) Frozen arms (B2/B4) ran *longer* than full-FT
(14–15h vs 9–10h) — head-only converges slower / early-stops less. (iv) Nondeterminism
unaddressed (single draw per arm), as in prior batches.

---

## Batch 2026-06-12 — GLM→age specificity control (does the GLM-activation vector predict age?)

**Goal.** Specificity control for the 2026-06-11 headline (PNC VWM best predicted by
the 400-dim `glm_scalar` activation vector, pooled r²≈0.23–0.25, beating connectivity
≈0.10–0.12). The GLM vector was never used to predict **age** — so we cannot tell if
it carries *VWM-specific* signal or is a generic developmental / SNR / engagement
proxy that predicts everything. PNC is an 8–21yo sample where connectivity predicts
age strongly (SC r²≈0.51, FC≈0.33), so a non-trivial GLM→age is plausible. These
cells predict `age_at_cnb` from `glm_scalar`, read against the grid: **GLM→age ≈0** ⇒
strong VWM-specificity (strengthens the headline); **moderate but <0.24** ⇒ partial
confound; **≈ or >0.24** ⇒ the GLM vector is a generic maturation/quality proxy and
the VWM result is not specific. Report `reports/2026-06-12-glm-age-specificity-control.md`.

**Design.** Identical to the 2026-06-11 PNC GLM→VWM cells except `labels=pnc_default`
(target `age_at_cnb`) replaces `labels=pnc_VWMdprime`. **No pipeline code changes**
(one config added — the pinned-input sweeper variant, below). `enet` + `xgb` are the
strict control (their sweepers do not touch `model.mlp_input`; input pinned to
`node_features`). The standard `mlp` sweeper **does** sweep
`mlp_input ∈ {node_features, adjacency, both}`, and for the age target the inner HPO
**escaped to connectivity input — GLM-only in 0/50 folds** → the sweeping
`mlp-pnc-glm-age` measures connectivity→age (pooled 0.573 ≈ the existing
`mlp-pnc-sc-age` 0.521), **not** GLM→age. So a 4th cell, **`mlp-pnc-glm-age-fixinput`**
(sweeper `configs/sweeper/mlp_fixedinput.yaml` = mlp sweeper minus the
`mlp_input`/`mlp_adjacency_type` dims, + `model.mlp_input=node_features`), pins the
input → a genuine 3rd strict GLM-only estimator. N = **945** subjects (993 − 48
`glm_missing`; age non-null for all) — a **+5 superset** of the 940 GLM→VWM subjects,
so GLM→age folds are **not** byte-identical to GLM→VWM → cross-target comparison is
**magnitude-only**; within-target 3-estimator comparison (the 3 strict cells) is valid
(shared N=945 / 50 folds, seeds 42–51).

**Shared recipe.** 10 reps × 5 outer folds × 20 inner Optuna trials, maximize
`val_r2`; project `baselines`, entity `teampolpetta`; branch
`feature/glm-age-specificity-control` @ `6634879`; submitted 2026-06-12. sklearn
(enet/xgb) → CPU (`slurm/train_sklearn.sh`, already sets `logging.project=baselines`);
MLP → GPU (`slurm/train.sh`, epochs=300; pass `logging.project=baselines` to override
the script's `"Baseline Launches"` default).

Strict GLM-only cells (the genuine GLM→age controls) first; the input-escaped sweeping
MLP last (annotated — its 0.573 is connectivity→age, not GLM→age, so a naive r²-sort
would mislead).

| experiment_name | model | compute | DEPLOY_SHA | Job ID | wandb | r² (mean-of-folds) | r² (pooled) |
|---|---|---|---|---|---|---|---|
| `enet-pnc-glm-age` | ElasticNet | node01 (CPU) | `6634879` | 361341 | [an1jtxae](https://wandb.ai/teampolpetta/baselines/runs/an1jtxae) | 0.074 ± 0.040 | **0.075** |
| `xgb-pnc-glm-age`  | XGBoost    | node02 (CPU) | `6634879` | 361342 | [8c51zkuu](https://wandb.ai/teampolpetta/baselines/runs/8c51zkuu) | 0.106 ± 0.046 | **0.106** |
| `mlp-pnc-glm-age-fixinput` | MLP (input **pinned**, GLM-only) | gpunode03/rtxa6000 | `48e5d3d` | 361347 | [a0hgujtf](https://wandb.ai/teampolpetta/baselines/runs/a0hgujtf) | 0.040 ± 0.081 | **0.041** |
| `mlp-pnc-glm-age`  | MLP (HPO **escaped → connectivity**; *not* GLM→age) | gpunode01/rtx2080 | `6634879` | 361343 | [953nb607](https://wandb.ai/teampolpetta/baselines/runs/953nb607) | 0.574 ± 0.056 | 0.573 |

**Commands** (RUN_ARGS forwarded inline; only the delta from the shared recipe shown
in prose above — the full lines):
```
cluster-submit --node node01 slurm/train_sklearn.sh -J enet-pnc-glm-age \
 "--export=ALL,RUN_ARGS=experiment_name=enet-pnc-glm-age dataset=pnc model=elasticnet \
  features=glm_scalar labels=pnc_default model.mlp_input=node_features \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  trainer.hpo_metric=val_r2 trainer.search_space=configs/sweeper/elasticnet.yaml"
# xgb: swap model=xgboost, search_space=configs/sweeper/xgboost.yaml, --node node02, -J xgb-pnc-glm-age
cluster-submit --node gpunode01 --gpu rtx2080 slurm/train.sh -J mlp-pnc-glm-age \
 "--export=ALL,RUN_ARGS=experiment_name=mlp-pnc-glm-age dataset=pnc model=mlp \
  features=glm_scalar labels=pnc_default model.mlp_input=node_features logging.project=baselines \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  trainer.epochs=300 trainer.hpo_metric=val_r2 trainer.search_space=configs/sweeper/mlp.yaml"
# pinned-input MLP (3rd strict estimator), DEPLOY_SHA 48e5d3d:
cluster-submit --node gpunode03 --gpu rtxa6000 slurm/train.sh -J mlp-pnc-glm-age-fixinput \
 "--export=ALL,RUN_ARGS=experiment_name=mlp-pnc-glm-age-fixinput dataset=pnc model=mlp \
  features=glm_scalar labels=pnc_default model.mlp_input=node_features logging.project=baselines \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  trainer.epochs=300 trainer.hpo_metric=val_r2 trainer.search_space=configs/sweeper/mlp_fixedinput.yaml"
```

**Reference magnitudes** (2026-06-11 batch, same N≈940 GLM input): GLM→VWM pooled r² —
enet **0.246** / mlp 0.230 / xgb 0.226; connectivity→age — SC ≈0.50–0.52, FC ≈0.32–0.37.

**Smoke gate.** All passed at `n_repetitions=1 n_outer_folds=2 inner_hpo_trials=2`
(`COMPLETED 0:0`; jobs 361338–361340, + pinned-MLP smoke 361346): RUN_ARGS echoed,
target `age_at_cnb`, N=945.

**Results.** Strict GLM-only estimators (the genuine GLM→age controls): pooled r²
**0.075** (enet) / **0.106** (xgb) / **0.041** (pinned MLP); mean-of-folds 0.074±0.040 /
0.106±0.046 / 0.040±0.081 (N=9450 pooled = 945×10; ± = fold dispersion, not SE). All
three **statistically indistinguishable** — corrected resampled paired t-test on `r2`,
all pairs p_adj ≥ 0.31 (`reports/comparison/pnc-glm-age-estimators/`). The sweeping
`mlp-pnc-glm-age` (0.573 pooled) is **connectivity→age via HPO input-escape** (`both`
27 / `adjacency` 23 / GLM-only **0** of 50 folds), not a GLM→age number — reported as a
positive control only. **Verdict:** GLM→age ≈0.04–0.11 ≪ GLM→VWM ≈0.23–0.25 (same
vector; 2–6×) ≪ connectivity→age ≈0.51 SC (~5–12×) → the GLM activation pattern is
substantially **VWM-specific**, not a generic maturation/quality proxy; the residual
age signal is small but non-zero (partial confound). Full write-up + interpretation
grid: `reports/2026-06-12-glm-age-specificity-control.md`.

---

## Batch 2026-06-11 — Classical-ML baselines (XGBoost / ElasticNet / MLP) vs the GNN (30-run matrix)

**Goal.** Non-graph baselines that learn directly from connectivity (and from the
GLM-activation vector), to answer two questions the GNN batches left open:
(1) *can a non-graph model on raw connectivity match the GNN?* — the standard
connectome-predictive-modeling sanity check; (2) *how much VWM is predictable
from the 400-value GLM-activation vector alone (no graph)?* Estimators: **XGBoost**
and **ElasticNet** (new sklearn path, `SklearnNestedCrossValidator`) + **MLP**
(existing torch `model=mlp`, reused). Spec
`docs/superpowers/specs/2026-06-10-classical-ml-connectivity-glm-baselines-design.md`,
plan `…/plans/2026-06-10-classical-ml-baselines.md`. Report
`reports/2026-06-11-classical-ml-baselines.md`.

**Design (locked).** Connectivity feature = the GNN's **top-20% thresholded
weighted** upper-triangle (`N(N−1)/2 = 79,800` for Schaefer-400), zeros where edges
dropped — the exact thresholded `Data` the GNN sees (fairest head-to-head). GLM
feature = `glm_scalar` (`contrast-2back_vs_0back`, 400-dim, 1/node). Per-fold
`StandardScaler` fit on train only (sklearn); identical folds to the GNN via the
shared `splits.py`; `FoldBarrier` reused for label/GLM normalization.

**Shared recipe.** 10 reps × 5 outer folds × 20 inner Optuna trials, maximize
`val_r2`; project `baselines`, entity `teampolpetta`; branch
`feature/classical-ml-baselines` @ `c5a6f7b`; submitted 2026-06-11.
```
dataset=<pnc|pnc_fc|orbit|orbit_fc> model=<elasticnet|xgboost|mlp> \
features=<default|glm_scalar> labels=<pnc_default|pnc_VWMdprime|default|orbit_mri_VWM_HL_p> \
model.mlp_input=<adjacency|node_features> experiment_name=<cell> \
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
trainer.hpo_metric=val_r2 trainer.search_space=configs/sweeper/<model>.yaml
```
- **Compute.** sklearn → CPU nodes (`slurm/train_sklearn.sh`); MLP → GPU
  (`slurm/train.sh`, epochs=300 + early-stop patience 35, the GNN protocol).
- **XGBoost·PNC·connectivity** (79,800-dim) is ~65–70h on CPU at this budget —
  over the 24h wall — so those 4 cells ran on **GPU** (`slurm/train.sh`,
  `+model.model_params.device=cuda`), finishing in 2h50m–6h20m with results
  matching ElasticNet (one benign predict-time CPU→GPU fallback warning; training
  is on `cuda:0`).
- **Tooling fixes this batch:** patched `cluster-submit` to forward an inline
  `RUN_ARGS=` env var to remote sbatch (it was silently dropped → jobs ran on
  Hydra defaults); patched `backfill-experiment-results` to recognize the sklearn
  runner's `"Classical-ML nested CV complete"` log line.

**Matrix (30 cells).** `<est>-<dataset>-<sc|fc|glm>-<age|vwm>`. Connectivity =
`features=default model.mlp_input=adjacency`; GLM = `features=glm_scalar
model.mlp_input=node_features` (VWM-only). All 30 `COMPLETED`.

**Results** (mean ± std over the 50 outer folds; **pooled** = single r² over all
out-of-fold predictions; sorted by pooled r². ± is fold dispersion, not a SE —
folds are correlated, don't read as a CI). wandb project `baselines`.

| experiment_name | r² (mean-of-folds) | r² (pooled) | Pearson r (mf) | MAE (mf) | RMSE (mf) | run |
|---|---|---|---|---|---|---|
| `mlp-pnc-sc-age`     | **0.521 ± 0.089** | **0.521** | 0.757 ± 0.041 | 1.791 ± 0.180 | 2.325 ± 0.229 | [vpurgih9](https://wandb.ai/teampolpetta/baselines/runs/vpurgih9) |
| `enet-pnc-sc-age`    | 0.516 ± 0.056 | 0.516 | 0.722 ± 0.040 | 1.797 ± 0.084 | 2.342 ± 0.163 | [5ykfadiu](https://wandb.ai/teampolpetta/baselines/runs/5ykfadiu) |
| `xgb-pnc-sc-age`     | 0.502 ± 0.053 | 0.501 | 0.717 ± 0.040 | 1.855 ± 0.098 | 2.377 ± 0.164 | [zmy7qnvj](https://wandb.ai/teampolpetta/baselines/runs/zmy7qnvj) |
| `mlp-pnc-fc-age`     | 0.367 ± 0.053 | 0.367 | 0.617 ± 0.042 | 2.149 ± 0.107 | 2.692 ± 0.150 | [8gva3wko](https://wandb.ai/teampolpetta/baselines/runs/8gva3wko) |
| `enet-pnc-fc-age`    | 0.317 ± 0.053 | 0.317 | 0.568 ± 0.046 | 2.254 ± 0.107 | 2.797 ± 0.147 | [hftn2cfa](https://wandb.ai/teampolpetta/baselines/runs/hftn2cfa) |
| `xgb-pnc-fc-age`     | 0.315 ± 0.053 | 0.314 | 0.574 ± 0.053 | 2.270 ± 0.111 | 2.802 ± 0.151 | [ougztq7w](https://wandb.ai/teampolpetta/baselines/runs/ougztq7w) |
| `enet-pnc-glm-vwm`   | **0.244 ± 0.045** | **0.246** | 0.504 ± 0.041 | 0.474 ± 0.019 | 0.620 ± 0.031 | [40hguv2e](https://wandb.ai/teampolpetta/baselines/runs/40hguv2e) |
| `mlp-pnc-glm-vwm`    | 0.227 ± 0.056 | 0.230 | 0.497 ± 0.046 | 0.482 ± 0.022 | 0.627 ± 0.032 | [bo0sr78a](https://wandb.ai/teampolpetta/baselines/runs/bo0sr78a) |
| `xgb-pnc-glm-vwm`    | 0.224 ± 0.053 | 0.226 | 0.481 ± 0.050 | 0.477 ± 0.021 | 0.628 ± 0.032 | [3ugm39e2](https://wandb.ai/teampolpetta/baselines/runs/3ugm39e2) |
| `xgb-orbit-glm-vwm`  | 0.139 ± 0.194 | 0.143 | 0.424 ± 0.180 | 0.124 ± 0.018 | 0.152 ± 0.023 | [1uzgrvyk](https://wandb.ai/teampolpetta/baselines/runs/1uzgrvyk) |
| `xgb-pnc-fc-vwm`     | 0.117 ± 0.046 | 0.117 | 0.347 ± 0.064 | 0.518 ± 0.024 | 0.679 ± 0.034 | [iluqg2in](https://wandb.ai/teampolpetta/baselines/runs/iluqg2in) |
| `mlp-orbit-glm-vwm`  | 0.102 ± 0.223 | 0.116 | 0.394 ± 0.200 | 0.124 ± 0.017 | 0.154 ± 0.022 | [y9ezrfh9](https://wandb.ai/teampolpetta/baselines/runs/y9ezrfh9) |
| `xgb-pnc-sc-vwm`     | 0.112 ± 0.044 | 0.113 | 0.344 ± 0.059 | 0.514 ± 0.020 | 0.672 ± 0.027 | [pa4s8skp](https://wandb.ai/teampolpetta/baselines/runs/pa4s8skp) |
| `enet-pnc-sc-vwm`    | 0.109 ± 0.033 | 0.110 | 0.342 ± 0.055 | 0.515 ± 0.018 | 0.673 ± 0.025 | [4fwu06yf](https://wandb.ai/teampolpetta/baselines/runs/4fwu06yf) |
| `mlp-pnc-fc-vwm`     | 0.101 ± 0.059 | 0.102 | 0.343 ± 0.074 | 0.520 ± 0.025 | 0.685 ± 0.034 | [mqp8bgle](https://wandb.ai/teampolpetta/baselines/runs/mqp8bgle) |
| `enet-pnc-fc-vwm`    | 0.095 ± 0.045 | 0.095 | 0.322 ± 0.070 | 0.527 ± 0.022 | 0.688 ± 0.033 | [bhc0b4nt](https://wandb.ai/teampolpetta/baselines/runs/bhc0b4nt) |
| `enet-orbit-glm-vwm` | 0.036 ± 0.223 | 0.049 | 0.338 ± 0.180 | 0.130 ± 0.017 | 0.160 ± 0.021 | [9biipk76](https://wandb.ai/teampolpetta/baselines/runs/9biipk76) |
| `xgb-orbit-sc-age`   | 0.022 ± 0.120 | 0.026 | 0.214 ± 0.176 | 0.676 ± 0.055 | 0.813 ± 0.062 | [scd6zuuz](https://wandb.ai/teampolpetta/baselines/runs/scd6zuuz) |
| `mlp-pnc-sc-vwm`     | 0.012 ± 0.141 | 0.014 | 0.282 ± 0.071 | 0.548 ± 0.045 | 0.707 ± 0.051 | [n1bvqrxs](https://wandb.ai/teampolpetta/baselines/runs/n1bvqrxs) |
| `enet-orbit-sc-vwm`  | −0.007 ± 0.157 | −0.004 | 0.152 ± 0.206 | 0.149 ± 0.017 | 0.186 ± 0.025 | [do8a22nt](https://wandb.ai/teampolpetta/baselines/runs/do8a22nt) |
| `enet-orbit-sc-age`  | −0.025 ± 0.120 | −0.015 | 0.164 ± 0.132 | 0.691 ± 0.046 | 0.831 ± 0.043 | [pzbt9b2i](https://wandb.ai/teampolpetta/baselines/runs/pzbt9b2i) |
| `xgb-orbit-sc-vwm`   | −0.018 ± 0.133 | −0.019 | 0.166 ± 0.186 | 0.150 ± 0.018 | 0.188 ± 0.025 | [liljdcxn](https://wandb.ai/teampolpetta/baselines/runs/liljdcxn) |
| `mlp-orbit-sc-vwm`   | −0.037 ± 0.189 | −0.031 | 0.276 ± 0.170 | 0.146 ± 0.017 | 0.189 ± 0.026 | [wy8tz7ye](https://wandb.ai/teampolpetta/baselines/runs/wy8tz7ye) |
| `xgb-orbit-fc-vwm`   | −0.078 ± 0.182 | −0.058 | 0.122 ± 0.186 | 0.155 ± 0.014 | 0.190 ± 0.021 | [b2ygr7a8](https://wandb.ai/teampolpetta/baselines/runs/b2ygr7a8) |
| `enet-orbit-fc-age`  | −0.068 ± 0.117 | −0.065 | 0.001 ± 0.107 | 0.696 ± 0.058 | 0.836 ± 0.067 | [tynk7oo4](https://wandb.ai/teampolpetta/baselines/runs/tynk7oo4) |
| `enet-orbit-fc-vwm`  | −0.101 ± 0.176 | −0.080 | 0.048 ± 0.149 | 0.157 ± 0.011 | 0.192 ± 0.020 | [z0pgqjnn](https://wandb.ai/teampolpetta/baselines/runs/z0pgqjnn) |
| `mlp-orbit-sc-age`   | −0.098 ± 0.210 | −0.087 | 0.230 ± 0.162 | 0.706 ± 0.064 | 0.858 ± 0.075 | [y7djeyb7](https://wandb.ai/teampolpetta/baselines/runs/y7djeyb7) |
| `mlp-orbit-fc-age`   | −0.095 ± 0.162 | −0.087 | 0.156 ± 0.199 | 0.691 ± 0.059 | 0.845 ± 0.070 | [u6ewnp7u](https://wandb.ai/teampolpetta/baselines/runs/u6ewnp7u) |
| `mlp-orbit-fc-vwm`   | −0.141 ± 0.222 | −0.131 | 0.148 ± 0.222 | 0.154 ± 0.017 | 0.195 ± 0.028 | [7am0p2bn](https://wandb.ai/teampolpetta/baselines/runs/7am0p2bn) |
| `xgb-orbit-fc-age`   | −0.178 ± 0.151 | −0.170 | −0.064 ± 0.185 | 0.728 ± 0.051 | 0.877 ± 0.064 | [h0o2pnzj](https://wandb.ai/teampolpetta/baselines/runs/h0o2pnzj) |

Job IDs: enet 361213–361222, xgb-CPU(ORBIT+GLM) 361223–361228, xgb-GPU(PNC-conn)
361231–361234, mlp 361235–361244.

**Findings.**
1. **PNC age is strongly predictable from connectivity** — SC r²≈0.50–0.52, FC
   r²≈0.32–0.37, across all three models. Confirms the pipeline is correctly wired
   (labels/features/folds). Atypically, **SC→age > FC→age** here (≈0.51 vs ≈0.33),
   the reverse of the usual FC-leads-for-age literature ordering.
2. **PNC VWM is only weakly predictable from connectivity** (SC/FC r²≈0.10–0.12)…
   **but best predicted by the GLM-activation vector** — `glm-vwm` r²≈0.23–0.25
   (PNC), ~2× the connectivity ceiling, on a 400-dim input vs 79,800-dim. Directly
   answers spec Q2: the GLM contrast carries the VWM signal, the graph does not add
   much.
3. **The three estimators are interchangeable** on the signal-bearing cells
   (PNC age, PNC GLM→VWM all agree within fold noise) — the p≫n regime does **not**
   favor XGBoost over ElasticNet, as anticipated. MLP edges the others on `pnc-sc-age`.
4. **ORBIT (N≈95–130) is near-zero/negative almost everywhere** — high variance,
   underpowered; the lone positive is **GLM→VWM** (xgb pooled 0.143, mlp 0.116),
   echoing PNC's GLM>connectivity-for-VWM pattern at small N. Pooled > mean-of-folds
   on ORBIT (fold-local means are noisy at n≈25/fold), so read pooled there.
5. **vs the GNN:** classical GLM→VWM PNC (≈0.23–0.25 pooled) **matches or exceeds**
   the GNN's best GLM-node-feature VWM result (≈0.18–0.21, glm_diagonal through GCN,
   batches 2026-05-29 / 06-04) — a non-graph model on the GLM vector is not beaten
   by the GNN. See the report for the corrected resampled t-tests.

## Batch 2026-06-09 — VWM-HL GLM node-features × backbone generalization on ORBIT (GCN/GAT/GIN/Transformer)

**What.** Reproduces the [2026-06-04 PNC backbone-generalization
batch](#batch-2026-06-04--vwm-glm-node-features--backbone-generalization-gatgintransformerbraingnn)
with two substitutions: **BrainGNN → GCN** (BrainGNN sat at the no-skill floor on
PNC, so it's replaced by GCN, which is also the ADR-0013 matched-HPO *base*) and
**PNC → ORBIT**. The full 7-cell node-feature matrix is run on four backbones —
**GCN, GAT, GIN, Graph Transformer** — holding the 2026-06-04 protocol fixed and
changing only `dataset`, `labels`, and the backbone+sweeper. 28 runs, jobs
**361026–361053**, submitted 2026-06-09 to `gpunode02`, on SHA **`c6997d5`**
(branch `feature/orbit-vwmhl-backbone-matrix`; config + submit-script only, **no
training-code change**, design spec
[`2026-06-09-orbit-vwmhl-gcn-backbone-matrix-design.md`](docs/superpowers/specs/2026-06-09-orbit-vwmhl-gcn-backbone-matrix-design.md)).
The **full 28-cell matrix passed a 1-rep/2-fold/2-epoch smoke** (jobs 360996 +
360997–361024, all `COMPLETED 0:0`, wandb off) before launch — validating every
preset × backbone path on ORBIT and deploy freshness.

**Shared recipe (identical to 2026-06-04 except dataset/label/backbone+sweeper).**
`dataset=orbit model=<backbone> labels=orbit_mri_VWM_HL_p features.glm_normalize=true
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 trainer.search_space=configs/sweeper/<sweeper>.yaml
trainer.hpo_metric=val_r2 logging.project=orbitglm`, `--time=2-00:00:00`, wandb
entity `teampolpetta`.

**Matched HPO (ADR-0013, GCN dropped into BrainGNN's slot).** Optimizer held out of
HPO for every backbone:
- **GCN / GIN** → `gcn_embedding_dim.yaml` (the matched base, no personalization).
- **GAT / Transformer** → `gat_embedding_dim.yaml` / `transformer_embedding_dim.yaml`
  = base + `model.heads: choice(1,2,4,8)`.

> **⚠️ Caveat — small N.** ORBIT VWM-HL has **N≈95** (bounded by GLM-map coverage:
> 101 maps / 129 structural / 130 non-null labels), ~10× smaller than PNC's ~940.
> Mean-of-folds r² is volatile at this N (the prior `orbit_GLM_VWMHL_gcn` GCN run
> gave per-fold r² −0.43…+0.27 with Pearson steady ~0.4), so the headline will lean
> on **pooled r² + Pearson r**. The label is a **bounded proportion-correct**
> (high-load, [0.11, 0.98]), *not* PNC's unbounded d-prime — so **absolute r² is not
> comparable to PNC**; only the *within-batch* contrasts (diagonal-vs-scalar, across
> backbones) transfer. Nondeterminism unaddressed (single noisy draw per cell, same
> as 2026-06-04).

**Submission manifest** (results backfilled per backbone once the jobs finish).
Per-cell suffix → features preset: `glmdiag`→`glm_diagonal`,
`id-glmscalar`→`identity_glm_scalar`, `id-glmdiag`→`identity_glm_diagonal`,
`scprof-glmscalar`→`scprofile_glm_scalar`, `scprof-glmdiag`→`scprofile_glm_diagonal`,
`lappe-glmscalar`→`laplacian_pe_glm_scalar`, `lappe-glmdiag`→`laplacian_pe_glm_diagonal`.

| Job | experiment_name | backbone | sweeper | R² (mean-of-folds) | R² (pooled) |
|---|---|---|---|---|---|
| 361026 | `gcn-orbit-sc-vwmhl-glmdiag`          | gcn | gcn_embedding_dim | 0.081 ± 0.235 | 0.092 |
| 361027 | `gcn-orbit-sc-vwmhl-id-glmscalar`     | gcn | gcn_embedding_dim | −0.104 ± 0.425 | −0.091 |
| 361028 | `gcn-orbit-sc-vwmhl-id-glmdiag`       | gcn | gcn_embedding_dim | 0.103 ± 0.235 | 0.115 |
| 361029 | `gcn-orbit-sc-vwmhl-scprof-glmscalar` | gcn | gcn_embedding_dim | −0.312 ± 0.935 | −0.285 |
| 361030 | `gcn-orbit-sc-vwmhl-scprof-glmdiag`   | gcn | gcn_embedding_dim | −0.441 ± 1.542 | −0.441 |
| 361031 | `gcn-orbit-sc-vwmhl-lappe-glmscalar`  | gcn | gcn_embedding_dim | −0.787 ± 2.511 | −0.672 |
| 361032 | `gcn-orbit-sc-vwmhl-lappe-glmdiag`    | gcn | gcn_embedding_dim | −0.052 ± 0.512 | −0.044 |
| 361033 | `gat-orbit-sc-vwmhl-glmdiag`          | gat | gat_embedding_dim | 0.100 ± 0.215 | 0.113 |
| 361034 | `gat-orbit-sc-vwmhl-id-glmscalar`     | gat | gat_embedding_dim | −0.089 ± 0.253 | −0.076 |
| 361035 | `gat-orbit-sc-vwmhl-id-glmdiag`       | gat | gat_embedding_dim | 0.136 ± 0.211 | 0.146 |
| 361036 | `gat-orbit-sc-vwmhl-scprof-glmscalar` | gat | gat_embedding_dim | −0.420 ± 0.952 | −0.412 |
| 361037 | `gat-orbit-sc-vwmhl-scprof-glmdiag`   | gat | gat_embedding_dim | −0.330 ± 0.491 | −0.320 |
| 361038 | `gat-orbit-sc-vwmhl-lappe-glmscalar`  | gat | gat_embedding_dim | −0.252 ± 0.736 | −0.261 |
| 361039 | `gat-orbit-sc-vwmhl-lappe-glmdiag`    | gat | gat_embedding_dim | 0.080 ± 0.358 | 0.099 |
| 361040 | `gin-orbit-sc-vwmhl-glmdiag`          | gin | gcn_embedding_dim | 0.121 ± 0.230 | 0.130 |
| 361041 | `gin-orbit-sc-vwmhl-id-glmscalar`     | gin | gcn_embedding_dim | −0.510 ± 1.417 | −0.410 |
| 361042 | `gin-orbit-sc-vwmhl-id-glmdiag`       | gin | gcn_embedding_dim | 0.099 ± 0.253 | 0.115 |
| 361043 | `gin-orbit-sc-vwmhl-scprof-glmscalar` | gin | gcn_embedding_dim | −0.305 ± 0.400 | −0.274 |
| 361044 | `gin-orbit-sc-vwmhl-scprof-glmdiag`   | gin | gcn_embedding_dim | −0.317 ± 0.535 | −0.279 |
| 361045 | `gin-orbit-sc-vwmhl-lappe-glmscalar`  | gin | gcn_embedding_dim | −0.436 ± 0.870 | −0.421 |
| 361046 | `gin-orbit-sc-vwmhl-lappe-glmdiag`    | gin | gcn_embedding_dim | 0.029 ± 0.415 | 0.025 |
| 361047 | `transformer-orbit-sc-vwmhl-glmdiag`          | transformer | transformer_embedding_dim | 0.092 ± 0.213 | 0.101 |
| 361048 | `transformer-orbit-sc-vwmhl-id-glmscalar`     | transformer | transformer_embedding_dim | −0.020 ± 0.354 | −0.009 |
| 361049 | `transformer-orbit-sc-vwmhl-id-glmdiag`       | transformer | transformer_embedding_dim | 0.121 ± 0.217 | 0.126 |
| 361050 | `transformer-orbit-sc-vwmhl-scprof-glmscalar` | transformer | transformer_embedding_dim | −0.212 ± 0.431 | −0.206 |
| 361051 | `transformer-orbit-sc-vwmhl-scprof-glmdiag`   | transformer | transformer_embedding_dim | −0.065 ± 0.209 | −0.060 |
| 361052 | `transformer-orbit-sc-vwmhl-lappe-glmscalar`  | transformer | transformer_embedding_dim | −0.726 ± 3.292 | −0.772 |
| 361053 | `transformer-orbit-sc-vwmhl-lappe-glmdiag`    | transformer | transformer_embedding_dim | −0.036 ± 0.517 | −0.020 |

**Results — full metrics, per backbone** (sorted by mean-of-folds r², the tuning
metric; N=50 outer folds = 10 reps × 5 folds; pooled = one r² over all **940**
out-of-fold predictions vs the global mean → **N=94** effective subjects). All 28
`COMPLETED 0:0`. `±` is dispersion across folds, **not** a standard error (folds are
correlated, and at N≈94 each fold's r² is computed on only ~9–19 test subjects —
hence the huge ± on the scalar cells). Recovered from wandb (entity `teampolpetta`,
project `orbitglm`) + per-run `checkpoints/<name>-<jobid>/` (ADR-0012); each pooled
run's recomputed mean-of-folds r² matched its training-log value (no drift).

**GCN** (`gcn_embedding_dim` base sweeper):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `id-glmdiag`       | 361028 | **0.103 ± 0.235** | 0.115 | 0.422 ± 0.178 | 0.124 ± 0.019 | 0.154 ± 0.024 | [m647wkrt](https://wandb.ai/teampolpetta/orbitglm/runs/m647wkrt) |
| `glmdiag`          | 361026 | 0.081 ± 0.235 | 0.092 | 0.414 ± 0.184 | 0.124 ± 0.017 | 0.156 ± 0.022 | [vw3xjerq](https://wandb.ai/teampolpetta/orbitglm/runs/vw3xjerq) |
| `lappe-glmdiag`    | 361032 | −0.052 ± 0.512 | −0.044 | 0.359 ± 0.243 | 0.132 ± 0.033 | 0.165 ± 0.038 | [xvax98r5](https://wandb.ai/teampolpetta/orbitglm/runs/xvax98r5) |
| `id-glmscalar`     | 361027 | −0.104 ± 0.425 | −0.091 | 0.274 ± 0.185 | 0.135 ± 0.026 | 0.170 ± 0.031 | [4ra2zk12](https://wandb.ai/teampolpetta/orbitglm/runs/4ra2zk12) |
| `scprof-glmscalar` | 361029 | −0.312 ± 0.935 | −0.285 | 0.133 ± 0.241 | 0.147 ± 0.044 | 0.182 ± 0.048 | [kz46fswd](https://wandb.ai/teampolpetta/orbitglm/runs/kz46fswd) |
| `scprof-glmdiag`   | 361030 | −0.441 ± 1.542 | −0.441 | 0.179 ± 0.231 | 0.152 ± 0.068 | 0.187 ± 0.069 | [rivz44qw](https://wandb.ai/teampolpetta/orbitglm/runs/rivz44qw) |
| `lappe-glmscalar`  | 361031 | −0.787 ± 2.511 | −0.672 | 0.021 ± 0.217 | 0.162 ± 0.064 | 0.200 ± 0.078 | [svy4wzro](https://wandb.ai/teampolpetta/orbitglm/runs/svy4wzro) |

**GAT** (`gat_embedding_dim` sweeper):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `id-glmdiag`       | 361035 | **0.136 ± 0.211** | 0.146 | 0.457 ± 0.194 | 0.123 ± 0.018 | 0.151 ± 0.022 | [hf3wpfoz](https://wandb.ai/teampolpetta/orbitglm/runs/hf3wpfoz) |
| `glmdiag`          | 361033 | 0.100 ± 0.215 | 0.113 | 0.399 ± 0.186 | 0.124 ± 0.018 | 0.155 ± 0.021 | [y8ay02to](https://wandb.ai/teampolpetta/orbitglm/runs/y8ay02to) |
| `lappe-glmdiag`    | 361039 | 0.080 ± 0.358 | 0.099 | 0.422 ± 0.204 | 0.123 ± 0.022 | 0.155 ± 0.028 | [9rzeu5we](https://wandb.ai/teampolpetta/orbitglm/runs/9rzeu5we) |
| `id-glmscalar`     | 361034 | −0.089 ± 0.253 | −0.076 | 0.251 ± 0.200 | 0.136 ± 0.019 | 0.170 ± 0.024 | [80w0cgc5](https://wandb.ai/teampolpetta/orbitglm/runs/80w0cgc5) |
| `lappe-glmscalar`  | 361038 | −0.252 ± 0.736 | −0.261 | 0.048 ± 0.192 | 0.144 ± 0.036 | 0.180 ± 0.047 | [2rlwc6k7](https://wandb.ai/teampolpetta/orbitglm/runs/2rlwc6k7) |
| `scprof-glmdiag`   | 361037 | −0.330 ± 0.491 | −0.320 | 0.085 ± 0.196 | 0.150 ± 0.026 | 0.187 ± 0.035 | [x6m8goyz](https://wandb.ai/teampolpetta/orbitglm/runs/x6m8goyz) |
| `scprof-glmscalar` | 361036 | −0.420 ± 0.952 | −0.412 | 0.036 ± 0.239 | 0.150 ± 0.026 | 0.191 ± 0.049 | [zh6448xm](https://wandb.ai/teampolpetta/orbitglm/runs/zh6448xm) |

**GIN** (`gcn_embedding_dim` base sweeper, no personalization):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `glmdiag`          | 361040 | **0.121 ± 0.230** | 0.130 | 0.425 ± 0.203 | 0.123 ± 0.018 | 0.153 ± 0.024 | [m2ywnf3u](https://wandb.ai/teampolpetta/orbitglm/runs/m2ywnf3u) |
| `id-glmdiag`       | 361042 | 0.099 ± 0.253 | 0.115 | 0.391 ± 0.220 | 0.123 ± 0.019 | 0.154 ± 0.023 | [pt1epl2k](https://wandb.ai/teampolpetta/orbitglm/runs/pt1epl2k) |
| `lappe-glmdiag`    | 361046 | 0.029 ± 0.415 | 0.025 | 0.384 ± 0.193 | 0.128 ± 0.027 | 0.160 ± 0.034 | [q7rd16c8](https://wandb.ai/teampolpetta/orbitglm/runs/q7rd16c8) |
| `scprof-glmscalar` | 361043 | −0.305 ± 0.400 | −0.274 | 0.088 ± 0.226 | 0.146 ± 0.026 | 0.185 ± 0.027 | [4xlsf2r1](https://wandb.ai/teampolpetta/orbitglm/runs/4xlsf2r1) |
| `scprof-glmdiag`   | 361044 | −0.317 ± 0.535 | −0.279 | 0.188 ± 0.229 | 0.150 ± 0.031 | 0.184 ± 0.033 | [fes2kdh5](https://wandb.ai/teampolpetta/orbitglm/runs/fes2kdh5) |
| `lappe-glmscalar`  | 361045 | −0.436 ± 0.870 | −0.421 | 0.035 ± 0.248 | 0.152 ± 0.036 | 0.191 ± 0.049 | [n97h5az6](https://wandb.ai/teampolpetta/orbitglm/runs/n97h5az6) |
| `id-glmscalar`     | 361041 | −0.510 ± 1.417 | −0.410 | 0.111 ± 0.206 | 0.152 ± 0.028 | 0.192 ± 0.046 | [mxfurbr8](https://wandb.ai/teampolpetta/orbitglm/runs/mxfurbr8) |

**Graph Transformer** (`transformer_embedding_dim` sweeper):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `id-glmdiag`       | 361049 | **0.121 ± 0.217** | 0.126 | 0.427 ± 0.202 | 0.123 ± 0.020 | 0.153 ± 0.024 | [95aedokn](https://wandb.ai/teampolpetta/orbitglm/runs/95aedokn) |
| `glmdiag`          | 361047 | 0.092 ± 0.213 | 0.101 | 0.421 ± 0.172 | 0.122 ± 0.018 | 0.155 ± 0.024 | [e228hsa1](https://wandb.ai/teampolpetta/orbitglm/runs/e228hsa1) |
| `id-glmscalar`     | 361048 | −0.020 ± 0.354 | −0.009 | 0.336 ± 0.216 | 0.131 ± 0.025 | 0.164 ± 0.028 | [82di6njp](https://wandb.ai/teampolpetta/orbitglm/runs/82di6njp) |
| `lappe-glmdiag`    | 361053 | −0.036 ± 0.517 | −0.020 | 0.389 ± 0.180 | 0.130 ± 0.030 | 0.164 ± 0.035 | [a308ubj3](https://wandb.ai/teampolpetta/orbitglm/runs/a308ubj3) |
| `scprof-glmdiag`   | 361051 | −0.065 ± 0.209 | −0.060 | 0.198 ± 0.206 | 0.134 ± 0.017 | 0.169 ± 0.023 | [d7781naf](https://wandb.ai/teampolpetta/orbitglm/runs/d7781naf) |
| `scprof-glmscalar` | 361050 | −0.212 ± 0.431 | −0.206 | 0.104 ± 0.266 | 0.144 ± 0.028 | 0.179 ± 0.033 | [7hdzc4df](https://wandb.ai/teampolpetta/orbitglm/runs/7hdzc4df) |
| `lappe-glmscalar`  | 361052 | −0.726 ± 3.292 | −0.772 | 0.123 ± 0.193 | 0.160 ± 0.105 | 0.194 ± 0.104 | [s4aprc35](https://wandb.ai/teampolpetta/orbitglm/runs/s4aprc35) |

**Corrected significance (ADR-0008, within backbone).** Bouckaert-Frank corrected
resampled paired t-test over per-outer-fold r² (`scripts/compare_models.py` on all 7
cells per backbone; `q` = Benjamini-Hochberg across the 21 within-backbone pairs). The
three **diagonal-vs-scalar, carrier-matched** key pairs (Δ = diagonal − scalar
mean-of-folds r²):

| backbone | identity carrier (id) | scprofile carrier (scprof) | laplacian-PE carrier (lappe) |
|---|---|---|---|
| GCN         | Δ+0.207, p=0.418, q=0.890 (n.s.) | Δ−0.129, p=0.890, q=0.890 (n.s.) | Δ+0.735, p=0.582, q=0.890 (n.s.) |
| GAT         | Δ+0.225, p=0.146, q=0.732 (n.s.) | Δ+0.090, p=0.873, q=0.917 (n.s.) | Δ+0.332, p=0.473, q=0.765 (n.s.) |
| GIN         | Δ+0.609, p=0.400, q=0.764 (n.s.) | Δ−0.012, p=0.971, q=0.971 (n.s.) | Δ+0.465, p=0.379, q=0.764 (n.s.) |
| Transformer | Δ+0.141, p=0.468, q=0.921 (n.s.) | Δ+0.147, p=0.502, q=0.921 (n.s.) | Δ+0.690, p=0.688, q=0.921 (n.s.) |

**0 of 21 pairs reach BH-significance in *any* backbone** (min q 0.69–0.92) — the
corrected per-fold test is underpowered at N≈94 (each fold's r² is on ~9–19 test
subjects; mean-of-folds r² std reaches 3.3 on a scalar cell). Contrast PNC (N≈940),
where lappe diagonal-vs-scalar hit q=0.013 (GAT) / q=0.001 (GIN). The ORBIT result is
a **null from low power, not counter-evidence.**

**Headline — does the PNC effect generalize to ORBIT VWM-HL?**
1. **The diagonal/identity ordering replicates, compressed.** In all four backbones
   the top-2 cells by both r² flavours are the two diagonal/identity cells
   (`id-glmdiag`, `glmdiag`): pooled r² **0.09–0.15**, Pearson **0.37–0.42** —
   ~half the PNC plateau (≈0.18–0.22 / ≈0.48), consistent with the smaller, bounded,
   differently-defined target and 10× smaller N.
2. **Scalar carriers collapse — the robust part.** Every `*-glmscalar` cell, every
   backbone, has **negative pooled r²** and **Pearson ≈ 0**. The carrier-matched
   diagonal−scalar pooled gap is positive in 11/12 cells (identity & laplacian-PE
   carriers large; scprofile shows **no** diagonal advantage, as on PNC). Signal =
   per-node distinctness, not scalar magnitude.
3. **No significance, but cross-backbone consistency.** The descriptive pattern is
   identical across 4 independent backbones; the evidence is that consistency, since
   the corrected test can't resolve it at N≈94. No backbone *fails* the task (unlike
   BrainGNN on PNC); cross-backbone absolute r² is confounded by attention `heads`
   DOF (reported, not ranked).

**Full report:** [`reports/2026-06-09-orbit-vwmhl-gcn-backbone-generalization.md`](reports/2026-06-09-orbit-vwmhl-gcn-backbone-generalization.md).

**Command (example, gcn-orbit-sc-vwmhl-id-glmdiag).** `cluster-submit --node gpunode02
slurm/train.sh -J gcn-orbit-sc-vwmhl-id-glmdiag --time=2-00:00:00
"--export=ALL,RUN_ARGS=experiment_name=gcn-orbit-sc-vwmhl-id-glmdiag
features=identity_glm_diagonal dataset=orbit model=gcn labels=orbit_mri_VWM_HL_p
features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5
trainer.inner_hpo_trials=20 trainer.epochs=300
trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml
trainer.hpo_metric=val_r2 logging.project=orbitglm"`. All 28 cells generated by
[`slurm/submit_orbit_vwmhl_matrix.sh`](slurm/submit_orbit_vwmhl_matrix.sh)
(`NODE=gpunode02 bash slurm/submit_orbit_vwmhl_matrix.sh`); plan
[`2026-06-09-orbit-vwmhl-gcn-backbone-matrix.md`](docs/superpowers/plans/2026-06-09-orbit-vwmhl-gcn-backbone-matrix.md).

---

## Batch 2026-06-08 — identity→VWM LayerNorm vs BatchNorm (the `model.norm` knob)

**What.** Follow-up to the [2026-06-04 decomposition](#batch-2026-06-04--identityvwm-r-decomposition-reproduce-the-lost-01--nested-reproducibility):
does swapping the per-layer norm change the identity→VWM floor? Shipped a
first-class **`model.norm`** knob (`batch`|`layer`|`none`, default `batch`,
byte-identical to all prior results) via a `build_norm` factory wired into
gcn/gat/gin (branch `feature/gnn-norm-layernorm`, deploy **`6253c1a`**;
26 tests in `tests/test_backbone_norm.py`). E4 re-runs the **exact E3 protocol**
with **`model.norm=layer`** (plain `nn.LayerNorm(hidden_dim)`) at seeds
42/100/200. Comparator = the existing E3 **BatchNorm** runs (360854/55/56) — not
re-run. Full analysis: **[report §9](reports/2026-06-04-identity-vwm-r2-decomposition.md#9-e4--layernorm-vs-batchnorm-at-the-floor-the-modelnorm-knob)**.

**Headline — no rescue.** LayerNorm lands at the same **≈0 floor**: mean pooled
R² **−0.026** (LayerNorm) vs **−0.018** (BatchNorm, E3) — a Δ of −0.008 that is
an order of magnitude smaller than the cross-seed spread (~0.06) and the
pipeline's same-seed run-to-run swing (±0.017, §4). Pooled MAE/RMSE are
indistinguishable too. The norm choice is **immaterial at this no-signal floor**;
it does not manufacture a signal that pure node identity does not carry.
*Hardware caveat:* E4 ran on **gpunode02/rtx6000**, E3 on **gpunode01/rtx2080** —
uncontrolled, but subsumed by the run-to-run noise already established on
identical hardware. `transformer` backbone keeps its own hardcoded LayerNorm
(not yet on the knob — follow-up).

**E4 — cluster.** `features=identity`, GCN, PNC SC → `VWM_overall_dprime`, 10 reps
× 5 outer folds, 20-trial inner HPO over `gcn_embedding_dim`, `hpo_metric=val_r2`,
**+ `model.norm=layer`**. All `COMPLETED 0:0`, ~10 h each, gpunode02/rtx6000,
wandb `orbitglm/teampolpetta`. N=9,730 OOF predictions. `±` is dispersion across
the 50 outer folds, **not** a standard error. Sorted by the tuning metric
(mean-of-folds r²):

| experiment_name | Job | seed | R² (mean-of-folds) | R² (pooled, N=9730) | R² pooled per-rep ± std | Pearson r (mof) | MAE (mof) | RMSE (mof) | run |
|---|---|---|---|---|---|---|---|---|---|
| `gcn-pnc-sc-vwm-identity-layernorm-seed200` | 360869 | 200 | −0.025 ± 0.100 | −0.022 | −0.022 ± 0.033 | 0.203 ± 0.140 | 0.557 ± 0.028 | 0.720 ± 0.037 | [xa4ih1f5](https://wandb.ai/teampolpetta/orbitglm/runs/xa4ih1f5) |
| `gcn-pnc-sc-vwm-identity-layernorm-seed100` | 360868 | 100 | −0.025 ± 0.100 | −0.025 | −0.025 ± 0.045 | 0.208 ± 0.125 | 0.559 ± 0.036 | 0.722 ± 0.042 | [95rs4cyw](https://wandb.ai/teampolpetta/orbitglm/runs/95rs4cyw) |
| `gcn-pnc-sc-vwm-identity-layernorm-seed42` | 360867 | 42 | −0.035 ± 0.143 | −0.031 | −0.031 ± 0.070 | 0.193 ± 0.142 | 0.560 ± 0.040 | 0.723 ± 0.047 | [77ss9cjr](https://wandb.ai/teampolpetta/orbitglm/runs/77ss9cjr) |
| *BatchNorm comparator (E3 mean of 3 seeds)* | *360854/5/6* | *42/100/200* | *−0.021* | *−0.018* | — | — | — | — | *see Batch 2026-06-04* |

### gcn-pnc-sc-vwm-identity-layernorm-seed42
- **Date:** 2026-06-08
- **Description:** identity→VWM at the exact E3 nested-CV+HPO protocol but with
  `model.norm=layer` (LayerNorm instead of BatchNorm), seed 42. Tests whether the
  norm choice rescues the ≈0 floor. Exercises the new `model.norm` knob.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity model.norm=layer trainer.seed=42` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `6253c1a`
- **Job ID:** 360867 (`-J gcn-pnc-sc-vwm-identity-layernorm-seed42`), gpunode02/rtx6000.
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-identity-layernorm-seed42` — https://wandb.ai/teampolpetta/orbitglm/runs/77ss9cjr
- **Command:** `cluster-submit --node gpunode02 --gpu rtx6000 slurm/train.sh -J gcn-pnc-sc-vwm-identity-layernorm-seed42 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity-layernorm-seed42 features=identity dataset=pnc model=gcn labels=pnc_VWMdprime model.norm=layer logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=42"`
- **Results:** pearson_r 0.193 ± 0.142, r2 −0.035 ± 0.143 (mean-of-folds), r2 −0.031 (pooled, N=9730), mae 0.560 ± 0.040, rmse 0.723 ± 0.047 (50 outer folds). **Floor ≈ 0; LayerNorm does not rescue it (vs E3 BatchNorm seed42 −0.011).** [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/77ss9cjr)

### gcn-pnc-sc-vwm-identity-layernorm-seed100
- **Date:** 2026-06-08
- **Description:** As `…-layernorm-seed42` but `trainer.seed=100` — seed sensitivity of the LayerNorm arm.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity model.norm=layer trainer.seed=100` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `6253c1a`
- **Job ID:** 360868 (`-J gcn-pnc-sc-vwm-identity-layernorm-seed100`), gpunode02/rtx6000.
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-identity-layernorm-seed100` — https://wandb.ai/teampolpetta/orbitglm/runs/95rs4cyw
- **Command:** `cluster-submit --node gpunode02 --gpu rtx6000 slurm/train.sh -J gcn-pnc-sc-vwm-identity-layernorm-seed100 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity-layernorm-seed100 features=identity dataset=pnc model=gcn labels=pnc_VWMdprime model.norm=layer logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=100"`
- **Results:** pearson_r 0.208 ± 0.125, r2 −0.025 ± 0.100 (mean-of-folds), r2 −0.025 (pooled, N=9730), mae 0.559 ± 0.036, rmse 0.722 ± 0.042 (50 outer folds). **Floor ≈ 0.** [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/95rs4cyw)

### gcn-pnc-sc-vwm-identity-layernorm-seed200
- **Date:** 2026-06-08
- **Description:** As `…-layernorm-seed42` but `trainer.seed=200` — seed sensitivity of the LayerNorm arm.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity model.norm=layer trainer.seed=200` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `6253c1a`
- **Job ID:** 360869 (`-J gcn-pnc-sc-vwm-identity-layernorm-seed200`), gpunode02/rtx6000.
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-identity-layernorm-seed200` — https://wandb.ai/teampolpetta/orbitglm/runs/xa4ih1f5
- **Command:** `cluster-submit --node gpunode02 --gpu rtx6000 slurm/train.sh -J gcn-pnc-sc-vwm-identity-layernorm-seed200 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity-layernorm-seed200 features=identity dataset=pnc model=gcn labels=pnc_VWMdprime model.norm=layer logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=200"`
- **Results:** pearson_r 0.203 ± 0.140, r2 −0.025 ± 0.100 (mean-of-folds), r2 −0.022 (pooled, N=9730), mae 0.557 ± 0.028, rmse 0.720 ± 0.037 (50 outer folds). **Floor ≈ 0.** [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/xa4ih1f5)

---

## Batch 2026-06-04 — identity→VWM R² decomposition (reproduce the lost 0.1 + nested reproducibility)

**What.** A three-experiment decomposition of a remembered "identity (one-hot)
node features → `VWM_overall_dprime` at R²≈0.10" (from a **separate, now-lost
codebase**; a flat 5-fold *pooled* test R²) against this repo's nearest run, job
**360785** (nested CV + HPO, mean-of-folds **−0.028 ± 0.181**). Two questions:
*is the 0.1 real and lost, or never there?* and *how reproducible is −0.028?*
Full analysis + mechanism: **[`reports/2026-06-04-identity-vwm-r2-decomposition.md`](reports/2026-06-04-identity-vwm-r2-decomposition.md)**.

**Headline.** The 0.10 is **not reproduced** — the identity→VWM generalization
floor is **≈ 0** in every protocol. The nested runner's sub-zero readings are a
refit/HPO-noise artifact, and the specific −0.028 is **not run-to-run
reproducible** (a same-seed re-run moved it to −0.011). Pooled ≈ mean-of-folds
in all 13 runs, so aggregation is not the hidden 0.1; the most likely origin is
inner-validation optimism (`val_r2`≈+0.1 not surviving to test) and/or a single
favorable draw of this noise-dominated pipeline. Aligns with the
[2026-06-03 reproducibility audit](#batch-2026-06-03--vwm-glm-node-features-re-run-reproducibility-audit).

**E3 — cluster (the 360785 protocol re-run).** Re-ran 360785's exact config
(`features=identity`, GCN, PNC SC → `VWM_overall_dprime`, 10 reps × 5 outer
folds, 20-trial inner HPO over `gcn_embedding_dim`, `hpo_metric=val_r2`) at three
seeds, deploy SHA **`c36b47c`** (main; includes **ADR-0012** → per-fold
predictions retained → pooled recoverable). All `COMPLETED 0:0`, ~16 h each,
gpunode01/rtx2080, wandb `orbitglm/teampolpetta`. N=9,730 OOF predictions
(10 reps × 973). `±` is dispersion across the 50 outer folds, **not** a standard
error (the folds are correlated). Sorted by the tuning metric (mean-of-folds r²):

| experiment_name | Job | seed | R² (mean-of-folds) | R² (pooled, N=9730) | R² pooled per-rep ± std | Pearson r (mof) | MAE (mof) | RMSE (mof) | run |
|---|---|---|---|---|---|---|---|---|---|
| `gcn-pnc-sc-vwm-identity-repro-seed100` | 360855 | 100 | +0.008 ± 0.177 | +0.008 | +0.008 ± 0.079 | 0.291 ± 0.096 | 0.549 ± 0.054 | 0.708 ± 0.061 | [wnypa1fk](https://wandb.ai/teampolpetta/orbitglm/runs/wnypa1fk) |
| `gcn-pnc-sc-vwm-identity-repro-seed42` | 360854 | 42 | −0.011 ± 0.167 | −0.010 | −0.010 ± 0.075 | 0.288 ± 0.100 | 0.553 ± 0.051 | 0.715 ± 0.059 | [yircuja2](https://wandb.ai/teampolpetta/orbitglm/runs/yircuja2) |
| `gcn-pnc-sc-vwm-identity-repro-seed200` | 360856 | 200 | −0.060 ± 0.257 | −0.054 | −0.054 ± 0.102 | 0.269 ± 0.107 | 0.568 ± 0.071 | 0.729 ± 0.075 | [isu722xd](https://wandb.ai/teampolpetta/orbitglm/runs/isu722xd) |
| `gcn-pnc-sc-vwm-identity` (orig 360785, seed 42) | 360785 | 42 | −0.028 ± 0.181 | n/a¹ | n/a | 0.260 ± 0.115 | 0.560 ± 0.053 | 0.720 ± 0.059 | [3oyjdbbg](https://wandb.ai/teampolpetta/orbitglm/runs/3oyjdbbg) |

¹ Pre-ADR-0012 (predictions overwritten); only mean-of-folds survives.
**Same seed (42), byte-identical training code** (`git diff c2e973b c36b47c -- src/`
touches only the ADR-0012 checkpoint path + new `run_identity.py`), yet
−0.028 → −0.011 — the pipeline is nondeterministic run-to-run. Across seeds the
headline wanders −0.060…+0.008; the "≈0 with large per-fold scatter" picture is
the only robust statement.

**E1/E2 — local companions (gitignored artifacts in `tmp/r2decomp/`).** N=973,
seeds 42–46, `dataset.root=…/DATA2/…/PNC`. Numbers live here + in the report
(not in git):
- **E1** (faithful reconstruction — `runner=flat_cv`, flat 5-fold, epochs=300,
  best-val restore — the *same quantity* the 0.1 was): **pooled R² +0.009 ± 0.015**
  (mean-of-folds +0.008 ± 0.015), range −0.014…+0.027. **0.10 not reproduced;
  floor ≈ 0.** Not a bug: train loss → ≈0.13 while val R² goes negative; best-val
  restore lands at the null model.
- **E2** (nested runner, no HPO: `inner_hpo_trials=0`, 1 rep, fixed-epoch refit):
  **pooled R² −0.189 ± 0.174** (mean-of-folds −0.190 ± 0.174), range −0.483…−0.027,
  ~11× E1's seed-variance. `refit_epochs` span 1–135 (median 4; 19/25 folds ≤5,
  underfit) → the nested negativity is a **protocol artifact**, not anti-signal.

### gcn-pnc-sc-vwm-identity-repro-seed42
- **Date:** 2026-06-04
- **Description:** Reproducibility re-run of job 360785's exact nested-CV+HPO
  config at the default seed (42), on ADR-0012 code so per-fold predictions (and
  pooled r²) are retained. Part of the identity→VWM R² decomposition.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity trainer.seed=42` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `c36b47c`
- **Job ID:** 360854 (`-J gcn-pnc-sc-vwm-identity-repro-seed42`), gpunode01/rtx2080.
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-identity-repro-seed42` — https://wandb.ai/teampolpetta/orbitglm/runs/yircuja2
- **Command:** `cluster-submit --node gpunode01 --gpu rtx2080 slurm/train.sh -J gcn-pnc-sc-vwm-identity-repro-seed42 --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity-repro-seed42 features=identity dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=42"`
- **Results:** pearson_r 0.288 ± 0.100, r2 −0.011 ± 0.167 (mean-of-folds), r2 −0.010 (pooled, N=9730), mae 0.553 ± 0.051, rmse 0.715 ± 0.059 (50 outer folds). **Floor ≈ 0; vs orig 360785 (same seed) −0.028 → −0.011.** [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/yircuja2)

### gcn-pnc-sc-vwm-identity-repro-seed100
- **Date:** 2026-06-04
- **Description:** As `…-repro-seed42` but `trainer.seed=100` — seed sensitivity of the nested headline.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity trainer.seed=100` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `c36b47c`
- **Job ID:** 360855 (`-J gcn-pnc-sc-vwm-identity-repro-seed100`), gpunode01/rtx2080.
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-identity-repro-seed100` — https://wandb.ai/teampolpetta/orbitglm/runs/wnypa1fk
- **Command:** `cluster-submit --node gpunode01 --gpu rtx2080 slurm/train.sh -J gcn-pnc-sc-vwm-identity-repro-seed100 --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity-repro-seed100 features=identity dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=100"`
- **Results:** pearson_r 0.291 ± 0.096, r2 +0.008 ± 0.177 (mean-of-folds), r2 +0.008 (pooled, N=9730), mae 0.549 ± 0.054, rmse 0.708 ± 0.061 (50 outer folds). **Floor ≈ 0.** [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/wnypa1fk)

### gcn-pnc-sc-vwm-identity-repro-seed200
- **Date:** 2026-06-04
- **Description:** As `…-repro-seed42` but `trainer.seed=200` — seed sensitivity of the nested headline.
- **Dataset / target:** PNC (SC, schaefer400) / `VWM_overall_dprime`.
- **Changed parameters:** `features=identity trainer.seed=200` (+ shared recipe).
- **Commit SHA (DEPLOY_SHA):** `c36b47c`
- **Job ID:** 360856 (`-J gcn-pnc-sc-vwm-identity-repro-seed200`), gpunode01/rtx2080.
- **wandb run:** orbitglm/teampolpetta — run `gcn-pnc-sc-vwm-identity-repro-seed200` — https://wandb.ai/teampolpetta/orbitglm/runs/isu722xd
- **Command:** `cluster-submit --node gpunode01 --gpu rtx2080 slurm/train.sh -J gcn-pnc-sc-vwm-identity-repro-seed200 --time=2-00:00:00 "--export=ALL,RUN_ARGS=experiment_name=gcn-pnc-sc-vwm-identity-repro-seed200 features=identity dataset=pnc model=gcn labels=pnc_VWMdprime logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=200"`
- **Results:** pearson_r 0.269 ± 0.107, r2 −0.060 ± 0.257 (mean-of-folds), r2 −0.054 (pooled, N=9730), mae 0.568 ± 0.071, rmse 0.729 ± 0.075 (50 outer folds). **Below the floor; widest fold scatter of the three.** [wandb run](https://wandb.ai/teampolpetta/orbitglm/runs/isu722xd)
## Batch 2026-06-04 — VWM GLM node-features × backbone generalization (GAT/GIN/Transformer/BrainGNN)

**What.** A robustness/generalization test of the [2026-06-03 GCN node-feature
batch](#batch-2026-06-03--vwm-glm-node-features-re-run-reproducibility-audit):
does the GCN node-feature *effect* (diagonal GLM forms ≳ scalar forms) replicate
when the **backbone** is swapped? The full 7-cell node-feature matrix is re-run on
four more architectures — **GAT, GIN, Graph Transformer, BrainGNN** — holding the
2026-06-03 protocol fixed and changing only `model=` and its matched HPO sweeper.
28 runs, jobs **360893–360920**, submitted 2026-06-04 to `gpunode02`, on SHA
**`43907c7`** (branch `feature/multi-backbone-vwm-matrix`, **ADR-0013**;
`configs/model/transformer.yaml` + 3 matched sweepers added, no training-code
change). All four backbones passed a 1-rep/2-fold/2-epoch smoke test
(`smoke-{gat,gin,transformer,braingnn}`, jobs 360889–360892) before launch.

**Shared recipe (identical to 2026-06-03 except backbone + sweeper).**
`dataset=pnc model=<backbone> labels=pnc_VWMdprime features.glm_normalize=true
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 trainer.search_space=configs/sweeper/<sweeper>.yaml
trainer.hpo_metric=val_r2 logging.project=orbitglm`, `--time=2-00:00:00`, wandb
entity `teampolpetta`. (`trainer.epochs=300` is pinned explicitly — git-confirmed
default, despite an erroneous "epochs=100" note in older docs.)

**Matched HPO (ADR-0013).** Shared architectural base (`gcn_embedding_dim.yaml`:
embedding_dim, hidden_dim, num_layers, dropout, pooling, jk_mode, head_hidden_dim,
head_num_layers) + per-model personalization, with the **optimizer held out of HPO
for every backbone**:
- **GIN** → `gcn_embedding_dim.yaml` (base, no personalization).
- **GAT / Transformer** → `gat_embedding_dim.yaml` / `transformer_embedding_dim.yaml`
  = base + `model.heads: choice(1,2,4,8)` (all hidden_dim choices divisible).
- **BrainGNN** → `braingnn_vwm_matched.yaml` = applicable base (drops `num_layers`
  [fixed 2] and `pooling`/`jk_mode` [no-ops]) + `model_params` (pool_ratio,
  roi_embed_dim, lambda_topk/unit/consist).

> **⚠️ Caveat — same as 2026-06-03.** Nondeterminism is **not** addressed (kept
> as-is for comparability), so each cell is a single noisy draw. The clean,
> primary comparison is **within each backbone** (diagonal-vs-scalar under an
> identical protocol; ADR-0008 corrected test valid there). Cross-backbone
> **absolute** r² is confounded — the attention models and BrainGNN tune extra
> degrees of freedom (`heads`, `model_params`) the generic base lacks — so do not
> read it as a "best backbone" ranking.

**Submission manifest** (results backfilled per backbone once the jobs finish).
Per-cell suffix → features preset: `glmdiag`→`glm_diagonal`,
`id-glmscalar`→`identity_glm_scalar`, `id-glmdiag`→`identity_glm_diagonal`,
`scprof-glmscalar`→`scprofile_glm_scalar`, `scprof-glmdiag`→`scprofile_glm_diagonal`,
`lappe-glmscalar`→`laplacian_pe_glm_scalar`, `lappe-glmdiag`→`laplacian_pe_glm_diagonal`.

| Job | experiment_name | backbone | sweeper | R² (mean-of-folds) | R² (pooled) |
|---|---|---|---|---|---|
| 360893 | `gat-pnc-sc-vwm-glmdiag`          | gat | gat_embedding_dim | 0.182 ± 0.101 | 0.186 |
| 360894 | `gat-pnc-sc-vwm-id-glmscalar`     | gat | gat_embedding_dim | 0.109 ± 0.172 | 0.113 |
| 360895 | `gat-pnc-sc-vwm-id-glmdiag`       | gat | gat_embedding_dim | 0.222 ± 0.089 | 0.226 |
| 360896 | `gat-pnc-sc-vwm-scprof-glmscalar` | gat | gat_embedding_dim | -0.011 ± 0.118 | -0.006 |
| 360897 | `gat-pnc-sc-vwm-scprof-glmdiag`   | gat | gat_embedding_dim | 0.019 ± 0.157 | 0.026 |
| 360898 | `gat-pnc-sc-vwm-lappe-glmscalar`  | gat | gat_embedding_dim | -0.006 ± 0.033 | -0.006 |
| 360899 | `gat-pnc-sc-vwm-lappe-glmdiag`    | gat | gat_embedding_dim | 0.183 ± 0.106 | 0.186 |
| 360900 | `gin-pnc-sc-vwm-glmdiag`          | gin | gcn_embedding_dim | 0.179 ± 0.096 | 0.183 |
| 360901 | `gin-pnc-sc-vwm-id-glmscalar`     | gin | gcn_embedding_dim | 0.018 ± 0.091 | 0.020 |
| 360902 | `gin-pnc-sc-vwm-id-glmdiag`       | gin | gcn_embedding_dim | 0.177 ± 0.103 | 0.180 |
| 360903 | `gin-pnc-sc-vwm-scprof-glmscalar` | gin | gcn_embedding_dim | -0.099 ± 0.358 | -0.088 |
| 360904 | `gin-pnc-sc-vwm-scprof-glmdiag`   | gin | gcn_embedding_dim | -0.111 ± 0.476 | -0.108 |
| 360905 | `gin-pnc-sc-vwm-lappe-glmscalar`  | gin | gcn_embedding_dim | -0.011 ± 0.023 | -0.009 |
| 360906 | `gin-pnc-sc-vwm-lappe-glmdiag`    | gin | gcn_embedding_dim | 0.193 ± 0.088 | 0.196 |
| 360907 | `transformer-pnc-sc-vwm-glmdiag`          | transformer | transformer_embedding_dim | 0.158 ± 0.132 | 0.161 |
| 360908 | `transformer-pnc-sc-vwm-id-glmscalar`     | transformer | transformer_embedding_dim | 0.128 ± 0.122 | 0.133 |
| 360909 | `transformer-pnc-sc-vwm-id-glmdiag`       | transformer | transformer_embedding_dim | 0.184 ± 0.131 | 0.190 |
| 360910 | `transformer-pnc-sc-vwm-scprof-glmscalar` | transformer | transformer_embedding_dim | 0.060 ± 0.115 | 0.060 |
| 360911 | `transformer-pnc-sc-vwm-scprof-glmdiag`   | transformer | transformer_embedding_dim | -0.068 ± 0.755 | -0.066 |
| 360912 | `transformer-pnc-sc-vwm-lappe-glmscalar`  | transformer | transformer_embedding_dim | -0.009 ± 0.048 | -0.008 |
| 360913 | `transformer-pnc-sc-vwm-lappe-glmdiag`    | transformer | transformer_embedding_dim | 0.143 ± 0.127 | 0.147 |
| 360914 | `braingnn-pnc-sc-vwm-glmdiag`          | braingnn | braingnn_vwm_matched | -0.075 ± 0.182 | -0.070 |
| 360915 | `braingnn-pnc-sc-vwm-id-glmscalar`     | braingnn | braingnn_vwm_matched | -0.066 ± 0.255 | -0.061 |
| 360916 | `braingnn-pnc-sc-vwm-id-glmdiag`       | braingnn | braingnn_vwm_matched | -0.016 ± 0.131 | -0.010 |
| 360917 | `braingnn-pnc-sc-vwm-scprof-glmscalar` | braingnn | braingnn_vwm_matched | -0.113 ± 0.165 | -0.109 |
| 360918 | `braingnn-pnc-sc-vwm-scprof-glmdiag`   | braingnn | braingnn_vwm_matched | -0.110 ± 0.129 | -0.104 |
| 360919 | `braingnn-pnc-sc-vwm-lappe-glmscalar`  | braingnn | braingnn_vwm_matched | -0.079 ± 0.163 | -0.074 |
| 360920 | `braingnn-pnc-sc-vwm-lappe-glmdiag`    | braingnn | braingnn_vwm_matched | -0.032 ± 0.122 | -0.028 |

**Results — full metrics, per backbone** (sorted by mean-of-folds r², the tuning
metric; N=50 outer folds = 10 reps × 5 folds; pooled = one r² over all 9,400
out-of-fold predictions vs the global mean). All 28 `COMPLETED 0:0`. `±` is
dispersion across folds, **not** a standard error (folds are correlated). Recovered
from wandb (entity `teampolpetta`, project `orbitglm`) + per-run
`checkpoints/<name>-<jobid>/` (ADR-0012); each pooled run's recomputed mean-of-folds
r² matched its training-log value (no drift).

**GAT** (`gat_embedding_dim` sweeper):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `id-glmdiag`       | 360895 | **0.222 ± 0.089** | 0.226 | 0.505 ± 0.095 | 0.485 ± 0.028 | 0.628 ± 0.036 | [0j9uf6jb](https://wandb.ai/teampolpetta/orbitglm/runs/0j9uf6jb) |
| `lappe-glmdiag`    | 360899 | 0.183 ± 0.106 | 0.186 | 0.485 ± 0.048 | 0.494 ± 0.037 | 0.644 ± 0.045 | [jb7uhdqk](https://wandb.ai/teampolpetta/orbitglm/runs/jb7uhdqk) |
| `glmdiag`          | 360893 | 0.182 ± 0.101 | 0.186 | 0.473 ± 0.088 | 0.496 ± 0.032 | 0.644 ± 0.040 | [x3tb7oc2](https://wandb.ai/teampolpetta/orbitglm/runs/x3tb7oc2) |
| `id-glmscalar`     | 360894 | 0.109 ± 0.172 | 0.113 | 0.431 ± 0.100 | 0.514 ± 0.048 | 0.671 ± 0.063 | [90o063jj](https://wandb.ai/teampolpetta/orbitglm/runs/90o063jj) |
| `scprof-glmdiag`   | 360897 | 0.019 ± 0.157 | 0.026 | 0.344 ± 0.110 | 0.545 ± 0.039 | 0.704 ± 0.050 | [0vlfdyw7](https://wandb.ai/teampolpetta/orbitglm/runs/0vlfdyw7) |
| `lappe-glmscalar`  | 360898 | −0.006 ± 0.033 | −0.006 | 0.125 ± 0.087 | 0.554 ± 0.024 | 0.716 ± 0.037 | [ylqd2zyc](https://wandb.ai/teampolpetta/orbitglm/runs/ylqd2zyc) |
| `scprof-glmscalar` | 360896 | −0.011 ± 0.118 | −0.006 | 0.267 ± 0.108 | 0.553 ± 0.036 | 0.716 ± 0.043 | [jvnc895f](https://wandb.ai/teampolpetta/orbitglm/runs/jvnc895f) |

**GIN** (`gcn_embedding_dim` base sweeper, no personalization):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `lappe-glmdiag`    | 360906 | **0.193 ± 0.088** | 0.196 | 0.452 ± 0.128 | 0.491 ± 0.028 | 0.640 ± 0.040 | [bew5f5ne](https://wandb.ai/teampolpetta/orbitglm/runs/bew5f5ne) |
| `glmdiag`          | 360900 | 0.179 ± 0.096 | 0.183 | 0.461 ± 0.088 | 0.494 ± 0.029 | 0.645 ± 0.040 | [9kmtqw1r](https://wandb.ai/teampolpetta/orbitglm/runs/9kmtqw1r) |
| `id-glmdiag`       | 360902 | 0.177 ± 0.103 | 0.180 | 0.486 ± 0.088 | 0.498 ± 0.033 | 0.646 ± 0.042 | [l3m1wunm](https://wandb.ai/teampolpetta/orbitglm/runs/l3m1wunm) |
| `id-glmscalar`     | 360901 | 0.018 ± 0.091 | 0.020 | 0.298 ± 0.094 | 0.545 ± 0.029 | 0.707 ± 0.042 | [clgpueh2](https://wandb.ai/teampolpetta/orbitglm/runs/clgpueh2) |
| `lappe-glmscalar`  | 360905 | −0.011 ± 0.023 | −0.009 | 0.019 ± 0.074 | 0.554 ± 0.019 | 0.718 ± 0.033 | [mcq5x01e](https://wandb.ai/teampolpetta/orbitglm/runs/mcq5x01e) |
| `scprof-glmscalar` | 360903 | −0.099 ± 0.358 | −0.088 | 0.287 ± 0.100 | 0.578 ± 0.092 | 0.740 ± 0.095 | [sjo2lhpb](https://wandb.ai/teampolpetta/orbitglm/runs/sjo2lhpb) |
| `scprof-glmdiag`   | 360904 | −0.111 ± 0.476 | −0.108 | 0.388 ± 0.130 | 0.583 ± 0.120 | 0.742 ± 0.129 | [6w914kdd](https://wandb.ai/teampolpetta/orbitglm/runs/6w914kdd) |

**Graph Transformer** (`transformer_embedding_dim` sweeper):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `id-glmdiag`       | 360909 | **0.184 ± 0.131** | 0.190 | 0.487 ± 0.101 | 0.495 ± 0.034 | 0.642 ± 0.043 | [vyrdpe85](https://wandb.ai/teampolpetta/orbitglm/runs/vyrdpe85) |
| `glmdiag`          | 360907 | 0.158 ± 0.132 | 0.161 | 0.462 ± 0.116 | 0.505 ± 0.047 | 0.653 ± 0.055 | [2brv4jxe](https://wandb.ai/teampolpetta/orbitglm/runs/2brv4jxe) |
| `lappe-glmdiag`    | 360913 | 0.143 ± 0.127 | 0.147 | 0.452 ± 0.111 | 0.508 ± 0.040 | 0.659 ± 0.049 | [76nobv2m](https://wandb.ai/teampolpetta/orbitglm/runs/76nobv2m) |
| `id-glmscalar`     | 360908 | 0.128 ± 0.122 | 0.133 | 0.391 ± 0.182 | 0.513 ± 0.037 | 0.665 ± 0.046 | [c2iswy5m](https://wandb.ai/teampolpetta/orbitglm/runs/c2iswy5m) |
| `scprof-glmscalar` | 360910 | 0.060 ± 0.115 | 0.060 | 0.324 ± 0.145 | 0.536 ± 0.044 | 0.691 ± 0.055 | [6rjs0xz3](https://wandb.ai/teampolpetta/orbitglm/runs/6rjs0xz3) |
| `lappe-glmscalar`  | 360912 | −0.009 ± 0.048 | −0.008 | 0.086 ± 0.092 | 0.556 ± 0.023 | 0.717 ± 0.036 | [yj84aisa](https://wandb.ai/teampolpetta/orbitglm/runs/yj84aisa) |
| `scprof-glmdiag`   | 360911 | −0.068 ± 0.755 | −0.066 | 0.362 ± 0.145 | 0.565 ± 0.166 | 0.719 ± 0.167 | [qmmdne40](https://wandb.ai/teampolpetta/orbitglm/runs/qmmdne40) |

**BrainGNN** (`braingnn_vwm_matched` sweeper):

| experiment_name | Job | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE | run |
|---|---|---|---|---|---|---|---|
| `id-glmdiag`       | 360916 | −0.016 ± 0.131 | −0.010 | 0.325 ± 0.071 | 0.555 ± 0.035 | 0.717 ± 0.044 | [yr925ivc](https://wandb.ai/teampolpetta/orbitglm/runs/yr925ivc) |
| `lappe-glmdiag`    | 360920 | −0.032 ± 0.122 | −0.028 | 0.286 ± 0.072 | 0.560 ± 0.039 | 0.724 ± 0.046 | [nehh4hck](https://wandb.ai/teampolpetta/orbitglm/runs/nehh4hck) |
| `id-glmscalar`     | 360915 | −0.066 ± 0.255 | −0.061 | 0.299 ± 0.075 | 0.570 ± 0.059 | 0.733 ± 0.078 | [v4r6dehx](https://wandb.ai/teampolpetta/orbitglm/runs/v4r6dehx) |
| `glmdiag`          | 360914 | −0.075 ± 0.182 | −0.070 | 0.285 ± 0.079 | 0.572 ± 0.053 | 0.737 ± 0.060 | [kh1fp2bs](https://wandb.ai/teampolpetta/orbitglm/runs/kh1fp2bs) |
| `lappe-glmscalar`  | 360919 | −0.079 ± 0.163 | −0.074 | 0.250 ± 0.094 | 0.574 ± 0.045 | 0.739 ± 0.056 | [crs863mu](https://wandb.ai/teampolpetta/orbitglm/runs/crs863mu) |
| `scprof-glmdiag`   | 360918 | −0.110 ± 0.129 | −0.104 | 0.195 ± 0.094 | 0.583 ± 0.037 | 0.750 ± 0.043 | [mf3abw9h](https://wandb.ai/teampolpetta/orbitglm/runs/mf3abw9h) |
| `scprof-glmscalar` | 360917 | −0.113 ± 0.165 | −0.109 | 0.199 ± 0.072 | 0.581 ± 0.047 | 0.751 ± 0.056 | [ckss2di3](https://wandb.ai/teampolpetta/orbitglm/runs/ckss2di3) |

**Corrected significance (ADR-0008, within backbone).** Bouckaert-Frank corrected
resampled paired t-test over per-outer-fold r² (`scripts/compare_models.py` on all 7
cells per backbone; `p_adj` = Benjamini-Hochberg across the 21 within-backbone
pairs). The three **diagonal-vs-scalar, carrier-matched** key pairs (Δ = diagonal −
scalar mean-of-folds r²):

| backbone | identity carrier (id) | scprofile carrier (scprof) | laplacian-PE carrier (lappe) |
|---|---|---|---|
| GAT         | Δ+0.112, p=0.204, q=0.390 (n.s.) | Δ+0.030, p=0.676, q=0.788 (n.s.) | **Δ+0.190, p=0.0018, q=0.013 (sig)** |
| GIN         | Δ+0.159, p=0.019, q=0.066 (n.s.) | Δ−0.013, p=0.968, q=0.969 (n.s.) | **Δ+0.204, p=5e-05, q=0.001 (sig)** |
| Transformer | Δ+0.056, p=0.447, q=0.866 (n.s.) | Δ−0.128, p=0.744, q=0.881 (n.s.) | Δ+0.152, p=0.037, q=0.258 (n.s.) |
| BrainGNN    | Δ+0.050, p=0.662, q=0.978 (n.s.) | Δ+0.003, p=0.970, q=0.978 (n.s.) | Δ+0.047, p=0.635, q=0.978 (n.s.) |

Within GAT/GIN/Transformer the three **diagonal** cells (`glmdiag`, `id-glmdiag`,
`lappe-glmdiag`) are mutually indistinguishable (all pairwise q≫0.05) — the carrier
doesn't matter once the node features are diagonal.

**Headline — does the GCN node-feature effect generalize across backbones?**
1. **The diagonal-GLM r²≈0.18–0.22 plateau replicates** in GAT (0.18–0.22), GIN
   (0.18–0.19) and Transformer (0.14–0.18), matching the GCN reference
   (id-glmdiag/lappe-glmdiag/glmdiag ≈0.19–0.22 in the [2026-06-03
   batch](#batch-2026-06-03--vwm-glm-node-features-re-run-reproducibility-audit)).
   The diagonal/one-hot node structure — not the positional carrier — sets the ceiling.
2. **"Diagonal ≳ scalar" holds *directionally* in all three message-passing/attention
   backbones, but reaches BH-significance only for the laplacian-PE carrier** (GAT
   q=0.013, GIN q=0.001; Transformer same sign, q=0.26), where the scalar form
   collapses to a ≈0 floor (Pearson 0.02–0.13) while the diagonal form holds at
   ~0.18. The identity-carrier gap is positive everywhere but n.s. after correction
   (consistent with GCN's n.s. Δ0.12); the scprofile carrier shows no diagonal
   advantage (both forms noisy, ≈0). So the effect that cleanly survives is **scalar
   carriers collapsing toward 0 while diagonal carriers hold a plateau** — i.e. the
   signal is per-node distinctness/structure, not the scalar magnitude, echoing the
   value-permutation result in the [2026-06-01
   batch](#batch-2026-06-01--glm-valueidentity-decoupling-value-permutation-ablation).
3. **BrainGNN does not learn the task** under the matched protocol: every cell
   r²∈[−0.11, −0.02] (best `id-glmdiag` −0.016). With the model at the no-skill
   floor the node-feature contrast is untestable there — a backbone-level negative,
   not evidence against the effect (cross-backbone *absolute* r² is confounded by the
   extra HPO DOF; reported, not ranked — see caveat above).

**Caveats.** Single noisy draw per cell; nondeterminism is *not* addressed (kept for
comparability — see [ADR-0013](docs/adr/0013-matched-hpo-cross-backbone.md) and the
run-to-run instability report). `±std` is fold dispersion, not a standard error.
Two scalar diagonal-cells carry pathological fold variance (`gin-scprof-glmdiag`
±0.476, `transformer-scprof-glmdiag` ±0.755) — single folds blew up; their means
are unreliable and both already sit at/below 0.

**Full report:** [`reports/2026-06-04-vwm-glm-node-features-cross-backbone-generalization.md`](reports/2026-06-04-vwm-glm-node-features-cross-backbone-generalization.md).

**Command (example, gat-pnc-sc-vwm-id-glmdiag).** `cluster-submit --node gpunode02
slurm/train.sh -J gat-pnc-sc-vwm-id-glmdiag --time=2-00:00:00
"--export=ALL,RUN_ARGS=experiment_name=gat-pnc-sc-vwm-id-glmdiag
features=identity_glm_diagonal dataset=pnc model=gat labels=pnc_VWMdprime
features.glm_normalize=true trainer.n_repetitions=10 trainer.n_outer_folds=5
trainer.inner_hpo_trials=20 trainer.epochs=300
trainer.search_space=configs/sweeper/gat_embedding_dim.yaml
trainer.hpo_metric=val_r2 logging.project=orbitglm"`

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
