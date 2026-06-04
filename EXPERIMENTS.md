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
| 360893 | `gat-pnc-sc-vwm-glmdiag`          | gat | gat_embedding_dim | TBD | TBD |
| 360894 | `gat-pnc-sc-vwm-id-glmscalar`     | gat | gat_embedding_dim | TBD | TBD |
| 360895 | `gat-pnc-sc-vwm-id-glmdiag`       | gat | gat_embedding_dim | TBD | TBD |
| 360896 | `gat-pnc-sc-vwm-scprof-glmscalar` | gat | gat_embedding_dim | TBD | TBD |
| 360897 | `gat-pnc-sc-vwm-scprof-glmdiag`   | gat | gat_embedding_dim | TBD | TBD |
| 360898 | `gat-pnc-sc-vwm-lappe-glmscalar`  | gat | gat_embedding_dim | TBD | TBD |
| 360899 | `gat-pnc-sc-vwm-lappe-glmdiag`    | gat | gat_embedding_dim | TBD | TBD |
| 360900 | `gin-pnc-sc-vwm-glmdiag`          | gin | gcn_embedding_dim | TBD | TBD |
| 360901 | `gin-pnc-sc-vwm-id-glmscalar`     | gin | gcn_embedding_dim | TBD | TBD |
| 360902 | `gin-pnc-sc-vwm-id-glmdiag`       | gin | gcn_embedding_dim | TBD | TBD |
| 360903 | `gin-pnc-sc-vwm-scprof-glmscalar` | gin | gcn_embedding_dim | TBD | TBD |
| 360904 | `gin-pnc-sc-vwm-scprof-glmdiag`   | gin | gcn_embedding_dim | TBD | TBD |
| 360905 | `gin-pnc-sc-vwm-lappe-glmscalar`  | gin | gcn_embedding_dim | TBD | TBD |
| 360906 | `gin-pnc-sc-vwm-lappe-glmdiag`    | gin | gcn_embedding_dim | TBD | TBD |
| 360907 | `transformer-pnc-sc-vwm-glmdiag`          | transformer | transformer_embedding_dim | TBD | TBD |
| 360908 | `transformer-pnc-sc-vwm-id-glmscalar`     | transformer | transformer_embedding_dim | TBD | TBD |
| 360909 | `transformer-pnc-sc-vwm-id-glmdiag`       | transformer | transformer_embedding_dim | TBD | TBD |
| 360910 | `transformer-pnc-sc-vwm-scprof-glmscalar` | transformer | transformer_embedding_dim | TBD | TBD |
| 360911 | `transformer-pnc-sc-vwm-scprof-glmdiag`   | transformer | transformer_embedding_dim | TBD | TBD |
| 360912 | `transformer-pnc-sc-vwm-lappe-glmscalar`  | transformer | transformer_embedding_dim | TBD | TBD |
| 360913 | `transformer-pnc-sc-vwm-lappe-glmdiag`    | transformer | transformer_embedding_dim | TBD | TBD |
| 360914 | `braingnn-pnc-sc-vwm-glmdiag`          | braingnn | braingnn_vwm_matched | TBD | TBD |
| 360915 | `braingnn-pnc-sc-vwm-id-glmscalar`     | braingnn | braingnn_vwm_matched | TBD | TBD |
| 360916 | `braingnn-pnc-sc-vwm-id-glmdiag`       | braingnn | braingnn_vwm_matched | TBD | TBD |
| 360917 | `braingnn-pnc-sc-vwm-scprof-glmscalar` | braingnn | braingnn_vwm_matched | TBD | TBD |
| 360918 | `braingnn-pnc-sc-vwm-scprof-glmdiag`   | braingnn | braingnn_vwm_matched | TBD | TBD |
| 360919 | `braingnn-pnc-sc-vwm-lappe-glmscalar`  | braingnn | braingnn_vwm_matched | TBD | TBD |
| 360920 | `braingnn-pnc-sc-vwm-lappe-glmdiag`    | braingnn | braingnn_vwm_matched | TBD | TBD |

**Reporting plan (when complete).** Per cell: mean-of-folds r²±std, pooled r²,
Pearson/MAE/RMSE (backfill from wandb + per-run `checkpoints/<name>-<jobid>/`).
Per backbone: ADR-0008 corrected paired t-test on the diagonal-vs-scalar key
pairs. Headline = whether the node-feature effect replicates within each
architecture.

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
