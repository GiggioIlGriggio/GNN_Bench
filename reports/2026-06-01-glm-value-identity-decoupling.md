# GLM value↔identity decoupling: a value-permutation ablation

**Date:** 2026-06-03
**Subject experiments:** the 2026-06-01 "GLM value↔identity decoupling" batch
(GCN on PNC structural connectivity, target `VWM_overall_dprime`) — see
`EXPERIMENTS.md` → *Batch 2026-06-01*.
**Thesis tested:** `glm-value-identity-decoupling-permutation`.
**Jobs:** 360785 (`identity`), 360783 (`permsubj`), 360784 (`permfixed`), deploy
SHA `c2e973b` (branch `feature/glm-value-permute`); arm 2 reuses job 360744
(`glm_diagonal`, deploy SHA `a1899eb`) since `glm_value_permute=none` is a no-op.

---

## TL;DR

The `glm_diagonal` node feature puts one scalar GLM activation on each node's own
one-hot cell. This batch asks *what part of that carries the VWM signal* by
scrambling **which value lands on which node**, three ways, against the no-value
floor. The pattern is a clean dichotomy — **{correct binding ≈ a fixed-but-wrong
permutation} ≫ {a per-subject permutation ≈ pure one-hot} ≈ 0**:

| arm | what it keeps | mean-of-folds R² |
|---|---|---|
| `permfixed` — one shared wrong π | distinct, **cross-subject-consistent** per-node values | **0.194 ± 0.093** |
| `glmdiag` — correct π=e | the biologically-correct binding | 0.176 ± 0.112 |
| `permsubj` — fresh π per subject | distinctness only; values inconsistent across subjects | 0.006 ± 0.082 |
| `identity` — no GLM | pure one-hot distinctness | −0.028 ± 0.181 |

Read literally this says the signal lives in **cross-subject-consistent per-node
value structure** — *any* fixed value→node code works (a GCN inverts an arbitrary
but consistent relabeling), the biological correctness of the binding does not
matter, and the one-hot distinctness prior **alone** carries nothing. That
**refutes H1 as stated** ("the diagonal's signal is the one-hot distinctness prior,
not the value/binding"): distinctness alone (the `identity` arm) is ≈0.

**Two hard caveats keep this provisional, not a finding.** (1) The batch's own
**pre-registered degeneracy guard fired** — `identity` scored R²≈0, under which the
locked rule says the matrix is uninformative pending a GCN/protocol re-examination.
(2) This is a **single draw on a pipeline that is not run-to-run reproducible**
(see [`reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md`](2026-05-29-vwm-glm-node-features-run-to-run-instability.md)),
and the pre-registered ADR-0008 corrected test is **not computable** here because
the pre-ADR-0012 shared-checkpoint overwrite destroyed the per-fold predictions of
every arm except `permfixed`.

---

## 1. What was tested

`glm_diagonal` gives each of the 400 nodes a one-hot identity vector with a single
GLM activation placed on its own diagonal cell — a distinct node tag (the one-hot)
fused with a value (the GLM magnitude). The new additive knob `features.glm_value_permute`
(`none|per_subject|fixed`, seeded by `glm_permute_seed`) permutes the GLM **values
across nodes before placement**, leaving the one-hot support intact. This cleanly
separates two things the standalone diagonal confounds:

- **the one-hot distinctness prior** — each node is individually addressable;
- **the value↔node binding** — *which* GLM magnitude sits on *which* node, and
  whether that mapping is consistent across subjects.

Four arms, all 400-wide, used standalone (no carrier), `glm_normalize=true`, shared
recipe `dataset=pnc model=gcn labels=pnc_VWMdprime`, 10×5 nested CV, 20-trial inner
HPO over `configs/sweeper/gcn_embedding_dim.yaml`, `hpo_metric=val_r2`:

| arm | preset | value binding | isolates |
|---|---|---|---|
| `identity` | `identity` | none (constant 1.0) | distinctness prior alone |
| `glmdiag` (reuse 360744) | `glm_diagonal` | correct (π=e) | distinctness + correct values |
| `permsubj` | `glm_diagonal_permsubj` | fresh random π **per subject** | distinctness, values inconsistent across subjects |
| `permfixed` | `glm_diagonal_permfixed` | one shared fixed π (wrong but stable) | distinctness + consistent-but-wrong values |

`identity` is a genuinely new baseline: the only prior identity run (360724) was on
`age`, not VWM. Arm 2 is reused unchanged because `glm_value_permute=none` is a
byte-identical no-op over the 2026-05-29 `glm_diagonal` run.

## 2. Results

Both R² flavours. Mean-of-folds averages 50 fold-local R²s (each against its own
~188-sample test mean); pooled is one R² over all 9,400 out-of-fold predictions
against the global mean. Sorted by mean-of-folds R² (the tuning metric). N=50
outer folds = 10 reps × 5 folds. `±` is dispersion across folds, **not** a standard
error (the 50 folds are correlated — 10 reps reuse the same subjects; within a rep
the 5 folds share training data).

| arm (job) | R² (mean-of-folds) | R² (pooled) | Pearson r | MAE | RMSE |
|---|---|---|---|---|---|
| `permfixed` (360784) | **0.194 ± 0.093** | 0.197 | 0.488 ± 0.073 | 0.493 ± 0.032 | 0.640 ± 0.039 |
| `glmdiag` (360744, reuse) | 0.176 ± 0.112 | n/a¹ | 0.487 ± 0.057 | 0.499 ± 0.039 | 0.647 ± 0.044 |
| `permsubj` (360783) | 0.006 ± 0.082 | n/a¹ | 0.246 ± 0.088 | 0.551 ± 0.027 | 0.711 ± 0.035 |
| `identity` (360785) | −0.028 ± 0.181 | n/a¹ | 0.260 ± 0.115 | 0.560 ± 0.053 | 0.720 ± 0.059 |

¹ Pooled R² is recoverable only for `permfixed` — see §4.

## 3. The dichotomy under the pre-registered primary rule

The locked primary rule: a ΔR² counts as real only if it exceeds the **larger**
arm's fold std. Applied to all six pairs:

| pair | ΔR² | larger fold std | real? |
|---|---|---|---|
| `permfixed` − `identity` | 0.222 | 0.181 | **yes** |
| `glmdiag` − `identity` | 0.204 | 0.181 | **yes** |
| `permfixed` − `permsubj` | 0.188 | 0.093 | **yes** |
| `glmdiag` − `permsubj` | 0.170 | 0.112 | **yes** |
| `permfixed` − `glmdiag` | 0.018 | 0.112 | no |
| `permsubj` − `identity` | 0.034 | 0.181 | no |

So **{permfixed, glmdiag}** each separate from **{permsubj, identity}**, while the
two members of each group are indistinguishable. The dichotomy is exactly along the
"is there a consistent per-node value code?" axis.

### Mechanism (why this grouping)

- **`permfixed` ≈ `glmdiag`.** One shared permutation means node *i* always carries
  node π(*i*)'s GLM value, for **every** subject. That is just a relabeling of a
  consistent code; the GCN learns the relabeled node→value→outcome mapping as
  easily as the correct one. *A consistent-but-biologically-wrong binding is as good
  as the correct binding.*
- **`permsubj` ≈ `identity` ≈ 0.** A fresh π per subject means node *i*'s column
  mixes different nodes' GLM values across subjects — there is no stable per-node
  value to learn, so the model falls back to the one-hot support alone, which (the
  `identity` arm shows) carries no VWM signal. *Destroying cross-subject consistency
  is equivalent to deleting the values.*

### What this says about the hypotheses

- **H1 ("the diagonal's signal is the one-hot distinctness prior, not the value")
  is refuted.** If it held, `identity` (pure distinctness) would track `glmdiag`;
  instead it is ≈0 while `glmdiag` is 0.18. The values matter.
- **But the values do not need to be the *correct* ones** — `permfixed` matches
  `glmdiag`. What matters is a **consistent per-node value code**, not its biological
  identity. The truth sits between "distinctness alone" and "correct binding."

## 4. What we cannot conclude (and why)

**The degeneracy guard fired.** The batch pre-registered: *"if arm 1 (`identity`)
scores R²≈0 on VWM the matrix is uninformative — re-examine GCN/protocol before
reading the GLM arms."* `identity` scored −0.028 ≈ 0, so by the locked rule the
ranking above is provisional. Mitigating point: the GLM arms are **not** globally
degenerate (`permfixed` 0.194, `glmdiag` 0.176 on the identical pipeline), so the
`identity`≈0 result is best read as a real *"pure one-hot carries no VWM signal"*
floor rather than a broken run — indeed it is the cleanest single piece of evidence
against H1. Still, the pre-registration is honored: treat §3 as a hypothesis.

**Single draw on a non-reproducible pipeline.** Training is nondeterministic on GPU
(`torch.use_deterministic_algorithms(True)` is unset → PyG scatter atomics) and the
single-holdout inner HPO is noise-dominated; the companion instability report shows
mean-of-folds R² shifting by up to 0.083 between byte-identical re-runs and rankings
reshuffling. The `permfixed` − `glmdiag` gap (0.018) is well inside that run-to-run
noise, so "permfixed ≈ glmdiag" is robust but their *ordering* is not. The larger
separations (0.17–0.22) exceed the observed ~0.08 run-to-run swing, so the dichotomy
itself is more likely to survive a re-run than any within-group order.

**The corrected test is not computable for this batch.** These three arms ran on the
pre-ADR-0012 code, which writes every run's per-fold predictions to a single shared
`checkpoints/nested_cv_result.json`. The arms ran concurrently and overwrote each
other; only the last to finish (`permfixed`, 2026-06-02 12:02) survives — confirmed
by matching its stored mean-of-folds R²=0.1937 to the run log. So the pre-registered
secondary rule (ADR-0008 corrected resampled t-test, which needs per-fold predictions
for both arms) cannot be run for any pair, and pooled R² exists only for `permfixed`.

## 5. Recommendations

1. **Re-run the batch on the ADR-0012 per-run-checkpoint code (now on `main`).**
   This is the single highest-value next step: it retains every arm's per-fold
   predictions, making the pre-registered corrected test and all-arm pooled R²
   computable, and gives a second draw to test the dichotomy against the known
   ~0.08 run-to-run noise.
2. **Fix the selection noise before over-reading any ordering** — replace the single
   75/25 inner holdout with full inner k-fold HPO (see the instability report §7).
   Until then, report the dichotomy, not the within-group ranks.
3. **If the dichotomy survives the re-run, it is the real finding:** the diagonal's
   VWM signal is a *learnable consistent per-node value code*, invariant to the
   specific value→node assignment, and absent from the one-hot support alone. That
   would also predict `permfixed` ≈ `glmdiag` ≫ `permsubj` is reproducible while the
   `permfixed`/`glmdiag` order is a coin flip.

## 6. Provenance / how to reproduce

- Mean-of-folds metrics are the `Nested CV complete — mean=… std=…` log lines for
  jobs 360783/360784/360785 (fetched via
  `.claude/skills/backfill-experiment-results/scripts/fetch-experiment-results.sh`).
- Pooled R² for `permfixed`: the surviving shared `checkpoints/nested_cv_result.json`
  (mtime 2026-06-02 12:02, run name `gcn-pnc-sc-vwm-glmdiag-permfixed`) →
  `.claude/skills/backfill-experiment-results/scripts/compute-pooled-metrics.py`
  (`pooled_r2=0.1973`, `meanfold_r2=0.1937` matching the log).
- wandb runs: `gcn-pnc-sc-vwm-permfixed` [2hds44mj], `permsubj` [1biyzrr7],
  `identity` [3oyjdbbg] under `teampolpetta/orbitglm`; `glmdiag` reuse [97893t1o].
- The permutation knob: `features.glm_value_permute` / `glm_permute_seed`, presets
  `glm_diagonal_permsubj` / `glm_diagonal_permfixed` (commit `c2e973b`).
