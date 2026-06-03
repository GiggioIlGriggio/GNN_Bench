# Run-to-run instability of the 2026-05-29 VWM GLM node-feature batch

**Date:** 2026-06-03
**Subject experiments:** the 2026-05-29 "VWM GLM node-feature encodings" 7-run
matrix (GCN on PNC structural connectivity, target `VWM_overall_dprime`) — see
`EXPERIMENTS.md` → *Batch 2026-05-29*.
**Original batch:** Slurm jobs 360744–360750, deploy SHA `a1899eb`.
**Re-run batch:** Slurm jobs 360772–360778, branch `feature/per-run-checkpoint-dir`,
deploy SHA `7cef2f6` (= `c80b62a` + ADR-0012 per-run checkpoint dir; no training-code change).

---

## TL;DR

Re-running the identical batch (same code, same seeds, same data splits) does **not**
reproduce the reported numbers. Mean-of-folds R² shifts by up to **0.083** between two
runs of the same experiment. The run-to-run noise is **comparable to or larger than the
effect sizes the batch interprets**, and the rankings reshuffle. Two compounding causes:
(1) training is **nondeterministic on GPU** — `torch.use_deterministic_algorithms(True)`
is never set, so PyG's scatter/gather atomic-adds are not reproducible; (2) the
single-holdout inner HPO is **noise-dominated** and flips the selected hyperparameters in
**46/50 folds** on that noise. Separately, the repo's own ADR-0008 corrected paired
t-test finds **no significant difference** among the top conditions even within a single
favorable draw. **The 2026-05-29 batch's fine-grained conclusions are not statistically
supported and not run-to-run robust.** The near-zero "scalar collapse" results are the
exception — they reproduce qualitatively.

---

## 1. What was tested

The original batch was re-submitted verbatim (same `RUN_ARGS`, same 10×5 nested-CV
protocol, same 20-trial inner HPO over `configs/sweeper/gcn_embedding_dim.yaml`,
`features.glm_normalize=true`) after fixing the shared-checkpoint overwrite (ADR-0012),
so each run now retains its per-fold predictions. The re-run ran on byte-identical
training code: `git diff a1899eb c80b62a -- src/` is empty; the only commits in between
are docs + the merge, and the ADR-0012 commit touches only the checkpoint path and
`run_name`, not the training computation. The default `trainer.epochs` was 100 in **both**
runs (the "300 epochs" note in `EXPERIMENTS.md` is incorrect — the default has been 100
since the initial commit).

## 2. Finding 1 — the pipeline is not reproducible run-to-run

Mean-of-folds R² (the tuning metric), original vs re-run, **same code and seeds**:

| experiment | orig R² | re-run R² | Δ |
|---|---|---|---|
| `glmdiag` (standalone) | +0.176 | +0.194 | +0.018 |
| `id-glmscalar` | +0.112 | +0.098 | −0.014 |
| `id-glmdiag` (orig "winner") | +0.213 | +0.218 | +0.005 |
| `scprof-glmscalar` | −0.018 | −0.060 | −0.042 |
| **`scprof-glmdiag`** | **+0.103** | **+0.020** | **−0.083** |
| `lappe-glmscalar` | −0.027 | −0.037 | −0.010 |
| **`lappe-glmdiag`** | **+0.176** | **+0.213** | **+0.037** |

The instability is uneven: the original winner (`id-glmdiag`) is stable (Δ=+0.005), but
`scprof-glmdiag` swings −0.083 and `lappe-glmdiag` +0.037. The rankings reshuffle — in the
re-run `lappe-glmdiag` (0.213) ties `id-glmdiag` (0.218) and **beats** standalone
`glmdiag` (0.194), directly contradicting the original finding that "laplacian_pe is a
wash" and "identity is the only carrier that improves on the standalone baseline."

## 3. Mechanism

**Direct evidence it is nondeterminism, not a code or data difference.** For the
`lappe-glmdiag` pair (the one experiment with a surviving original JSON, job 360750):

- `outer_seeds` match; **all 50 folds have identical `y_true`** → identical subjects,
  labels, splits, and features. The data path is fully deterministic.
- Yet only **4/50 folds select the same hyperparameters.** The inner HPO is seeded
  identically, so identical selection would require identical per-trial val scores;
  they differ → **training itself is nondeterministic.**
- In the 4 folds that *did* land on identical hyperparameters, predictions **still
  differ** (`|Δy_pred|` 0.16–0.29). Example: `rep3 fold4`, same config/split/seed, fold
  R² **0.083 → 0.243**. This isolates the cause to nondeterministic training, not
  selection.

**Why.** `src/utils/seed.py` sets `torch.backends.cudnn.deterministic = True` but **not**
`torch.use_deterministic_algorithms(True)`. The cuDNN flag only governs convolution
kernels; it does nothing for the **scatter/gather atomic-adds** that implement GNN
message passing and pooling in PyTorch-Geometric, which are nondeterministic on GPU. Every
training run is therefore a slightly different draw.

**Amplification (the real disease).** That small training noise feeds the inner HPO, which
uses a **single 75/25 holdout** (one 188-sample validation set decides every trial; see
`_inner_split`, `nested_cross_validation.py:688`). The trial-to-trial val-score gaps are
smaller than the val set's own noise, so the noise flips the winning config in **46/50
folds**. Different configs → different models → the R² swings in §2. This is the empirically
observed counterpart of the "50/50 distinct configs selected" instability already noted in
`EXPERIMENTS.md`.

## 4. Finding 2 — no significant differences even within one draw

The repo's ADR-0008 corrected resampled paired t-test (`src/training/statistical_tests.py`,
Nadeau–Bengio / Bouckaert–Frank; variance inflated by `n_test/n_train = 1/(k−1) = 0.25`)
on the re-run batch's per-fold R² (n=50, k=5):

| comparison | Δ R² | p (corrected) | verdict |
|---|---|---|---|
| `id-glmdiag` vs `glmdiag` | +0.024 | 0.73 | ns |
| `id-glmdiag` vs `lappe-glmdiag` | +0.004 | 0.93 | ns |
| `lappe-glmdiag` vs `glmdiag` | +0.020 | 0.75 | ns |
| `id-glmdiag` vs `id-glmscalar` (diag vs scalar) | +0.120 | 0.11 | ns |
| `scprof-glmdiag` vs `scprof-glmscalar` (diag vs scalar) | +0.079 | 0.71 | ns |

None reach p<0.05 — not even diagonal-vs-scalar, the batch's central claim. (p-values use a
normal approximation because scipy is absent in the analysis venv; the Student-t with df=49
only *widens* them, so the "all ns" verdict is conservative.) This is the correct way to
compare models here; the `±std` overlap quoted in `EXPERIMENTS.md` is dispersion across
folds, **not** a significance test, and must not be read as one (the 50 fold scores are
correlated — 10 reps reuse the same 940 subjects and each rep's 5 folds share 3/5 of their
training data — so no √50, no CI from the std).

## 5. What survives and what does not

- **Not supported:** "Winner: identity_glm_diagonal"; "identity is the only carrier that
  improves on standalone glmdiag"; "laplacian_pe is a wash"; "glm_diagonal beats
  glm_scalar for every carrier (significantly)." The rankings move across re-runs and the
  corrected test is `ns`.
- **Qualitatively robust:** the **scalar-form weakness / collapse** for the `sc_row` and
  `laplacian_pe` carriers. `lappe-glmscalar` is dead in both draws (R²≈−0.03, pearson≈0),
  and the scalar carriers sit near or below zero in both. These are large, sign-stable
  effects rather than fine margins, so they survive the noise — though they are still not
  formally tested here.

## 6. Both R² flavours for the re-run batch (reference)

Internally-consistent numbers (both from the same run), for the record. "Mean-of-folds"
averages 50 fold-local R²s (each against its own ~188-sample test mean); "pooled" is a
single R² over all 9,400 out-of-fold predictions against the global mean. They agree
closely here. Pooled is **not** available for the 2026-05-29 originals — their predictions
were overwritten before ADR-0012 (only `lappe-glmdiag`/360750 survived, pooled R² 0.179).

| experiment (re-run job) | R² mean-of-folds | R² pooled (N=9400) | pearson (mof) |
|---|---|---|---|
| `id-glmdiag` (360774) | 0.218 ± 0.093 | 0.222 | 0.499 |
| `lappe-glmdiag` (360778) | 0.213 ± 0.081 | 0.217 | 0.508 |
| `glmdiag` (360772) | 0.194 ± 0.100 | 0.193 | 0.490 |
| `id-glmscalar` (360773) | 0.098 ± 0.144 | 0.104 | 0.400 |
| `scprof-glmdiag` (360776) | 0.020 ± 0.284 | 0.017 | 0.442 |
| `lappe-glmscalar` (360777) | −0.037 ± 0.096 | −0.037 | −0.029 |
| `scprof-glmscalar` (360775) | −0.060 ± 0.302 | −0.058 | 0.309 |

Note the huge fold-std on the `scprof` rows (±0.28–0.30) — those conditions are unstable
fold-to-fold as well as run-to-run.

## 7. Recommendations

1. **Make runs reproducible.** Set `torch.use_deterministic_algorithms(True)` (and export
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`) in `seed_everything`. *Caveat:* this only freezes one
   arbitrary draw of a noise-dominated selection — reproducible ≠ correct.
2. **Fix the selection (the root cause).** Replace the single 75/25 inner holdout with
   **full inner k-fold** HPO (average each config's val score over all k inner folds before
   selecting). `_inner_split` already builds a 4-way `StratifiedKFold` and discards 3 folds;
   enabling it is a small change. This makes selection robust to training noise — the actual
   disease behind the 46/50 config flips. Cost: ~k× the inner compute.
3. **Until then, do not report single-run rankings as findings.** Either report run-to-run
   spread (re-run each condition over several seeds and quote mean ± run-to-run std) or, at
   minimum, gate any claim on the ADR-0008 corrected test rather than `±std` overlap.
4. **Correct `EXPERIMENTS.md`:** the "300 epochs" note is wrong (default is 100), and the
   2026-05-29 findings block should carry a pointer to this report.

## 8. Provenance / how to reproduce

- Per-run artifacts (ADR-0012): `checkpoints/<experiment_name>-<jobid>/nested_cv_result.json`.
  Re-run JSONs fetched to `/tmp/rerun/<jobid>.json`; original surviving JSON
  (`lappe-glmdiag`/360750) at `/tmp/nested_cv_result.json`.
- Both R² flavours: `.claude/skills/backfill-experiment-results/scripts/compute-pooled-metrics.py <json>`.
- Diagnostics in §3 (split match, hparam match, same-config prediction drift) and the
  corrected test in §4 were computed directly from the per-fold `y_true`/`y_pred` arrays in
  those JSONs (n=50, k=5, `n_test/n_train`=0.25).
- The corrected-test logic mirrors `src/training/statistical_tests.py`; `scripts/compare_models.py`
  runs it directly on the 7 re-run JSONs once a torch-complete environment is available.
