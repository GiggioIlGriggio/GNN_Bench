# Where did the identity→VWM R²≈0.1 go? A three-experiment decomposition

**Date:** 2026-06-04
**Subject question:** a remembered "identity (one-hot) node features predict
`VWM_overall_dprime` at R²≈0.10" — from a **separate, now-lost codebase** (a
flat 5-fold pooled test R²) — could not be reproduced in this repo, where the
nearest run (job 360785, nested CV + HPO) reports **−0.028 ± 0.181**
mean-of-folds. This report decomposes the gap: *is the 0.1 real and lost, or was
it never there?* and *how reproducible is the −0.028 itself?*
**Branch:** `feature/vwm-identity-r2-decomposition` (this PR).
**Experiments:** E1/E2 local (`tmp/r2decomp/`, gitignored); E3 cluster jobs
**360854 / 360855 / 360856**, deploy SHA `c36b47c` (main; includes ADR-0012).
Original comparator: job **360785**, deploy SHA `c2e973b`.

---

## TL;DR

The **0.10 is not reproduced** and there is no evidence it was ever a
generalization R² on this data. Across three independent attempts the
identity→VWM signal sits at a hard **floor of ≈ 0**:

- **E1 (the faithful reconstruction — flat 5-fold, pooled test R², the *same
  quantity* the 0.1 was):** pooled R² = **+0.009 ± 0.015** over seeds 42–46
  (range −0.014…+0.027). Not a bug: the model trains to near-zero train loss
  while validation R² goes negative; best-val restore lands the model at the
  true identity floor ≈ 0. **One-hot node identity carries no generalizable VWM
  signal here.**
- **E2 (the nested runner in its degenerate no-HPO mode):** pooled R² =
  **−0.189 ± 0.174** (range −0.483…−0.027), ~11× E1's seed-variance. The cause
  is mechanical: an effectively-unmonitored refit whose depth swings 1–135
  epochs (median 4) — under- or over-fitting per fold. The negativity is a
  **protocol artifact**, not anti-signal.
- **E3 (the exact 360785 protocol — full nested CV + 20-trial HPO, re-run on the
  cluster at 3 seeds):** mean-of-folds R² wanders **−0.060 … +0.009** across
  seeds; pooled **−0.054 … +0.008**. The **same-seed** re-run (seed 42) of 360785
  moved its headline from **−0.028 → −0.011** despite byte-identical training
  code — confirming this pipeline is **not run-to-run reproducible** (the
  GPU-nondeterminism + noise-dominated inner HPO documented in
  [`reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md`](2026-05-29-vwm-glm-node-features-run-to-run-instability.md)).
  The precise number is not seed-reproducible; the **"≈0 with large per-fold
  scatter" picture is robust.**

In every experiment **pooled R² ≈ mean-of-folds R²** (within ≤0.006), so the
aggregation choice is *not* what hides the 0.1. The most likely origin of the
remembered 0.10 is the lost codebase's *inner-validation* optimism (HPO selects
on a val_r2 of ≈+0.1 that does not survive to held-out test; cf.
`val_test_optimism_gap`) and/or a single favorable draw of this noise-dominated
pipeline — not a reproducible identity→VWM effect.

---

## 1. The question and the three reconstructions

The "0.1" was, by the user's recollection, a **flat 5-fold cross-validation,
pooled-over-all-folds test R²** (one R² over every out-of-fold prediction
against the global mean), produced by a codebase that no longer exists. This
repo never ran that exact protocol for identity→VWM; its nearest artifact (job
360785) is a 10×5 **nested** CV with a 20-trial inner HPO, reporting
**mean-of-folds** R² = −0.028. Three confounds therefore separate the two
numbers: **(a)** flat vs nested protocol, **(b)** pooled vs mean-of-folds
aggregation, **(c)** run-to-run / seed noise. The three experiments isolate each:

| Exp | Where | Protocol | Purpose | N (preds) |
|---|---|---|---|---|
| **E1** | local | flat 5-fold, epochs=300, best-val restore | reproduce the 0.1 *as the same quantity* | 973 |
| **E2** | local | nested runner, no HPO (`inner_hpo_trials=0`, 1 rep, fixed-epoch refit) | isolate the protocol's own contribution | 973 |
| **E3** | cluster | full nested CV (10 reps × 5 folds, 20-trial HPO) — **360785's exact config** | reproducibility of the −0.028 itself | 9,730 |

E1 was run through the new `runner=flat_cv` route added on this branch
(Deliverable A); its artifact is `nested_cv_result.json`-compatible, so the same
`scripts/pooled_vs_meanfolds.py` recompute applies to all three.

## 2. E1 — the faithful reconstruction does not find 0.1 (floor ≈ 0)

Flat 5-fold CV, `features=identity`, GCN, PNC SC → `VWM_overall_dprime`,
`epochs=300` with best-validation-epoch restore, seeds 42–46 (N=973):

| seed | R² mean-of-folds | R² pooled (N=973) |
|---|---|---|
| 42 | −0.0152 ± 0.0797 | −0.0136 |
| 43 | +0.0155 ± 0.0760 | +0.0129 |
| 44 | +0.0151 ± 0.1177 | +0.0161 |
| 45 | +0.0024 ± 0.0342 | +0.0024 |
| 46 | +0.0245 ± 0.0538 | +0.0272 |
| **summary** | **+0.008 ± 0.015** | **+0.009 ± 0.015** |

The pooled R² — **the same quantity the remembered 0.10 was** — is
+0.009 ± 0.015, range −0.014…+0.027. **The 0.10 is not reproduced; the gap is
~0.09 R², an order of magnitude.**

This is a genuine floor, not a training failure: across seeds the model drives
**train loss → ≈0.13** (near-zero) while **validation R² goes negative**, the
signature of memorizing the one-hot inputs with no transferable structure. The
best-validation-epoch restore therefore checkpoints the model at the point where
it still generalizes — which for pure identity features is the **null model**,
R² ≈ 0. Pure node identity (a constant 1.0 on each of 400 distinct nodes) gives
the GCN positional one-hot distinctness but no subject-varying signal, so there
is nothing to generalize from.

## 3. E2 — the nested protocol's own negativity is a refit artifact

Running the nested runner in its degenerate, no-HPO mode (`inner_hpo_trials=0`,
`n_repetitions=1`, fixed 300-epoch refit budget) on the same data, seeds 42–46:

| seed | R² mean-of-folds | R² pooled (N=973) |
|---|---|---|
| 42 | −0.1018 ± 0.1045 | −0.1024 |
| 43 | −0.1897 ± 0.1427 | −0.1793 |
| 44 | −0.1560 ± 0.2508 | −0.1546 |
| 45 | −0.4825 ± 0.3562 | −0.4826 |
| 46 | −0.0240 ± 0.0992 | −0.0274 |
| **summary** | **−0.190 ± 0.174** | **−0.189 ± 0.174** |

This sits **below** the E1 floor and swings ~11× more across seeds (pooled std
0.174 vs 0.015). The mechanism is visible directly in `refit_epochs` (epochs
actually trained before the refit's restore fires), pooled over all 25 folds:

- **range 1–135, median 4**; **19/25 folds stop at ≤5 epochs** (severe
  underfit), **3/25 run ≥60 epochs** (overfit past the generalization point).

Without a well-tuned, monitored stopping signal the refit lands almost randomly
on the under-/over-fit spectrum, and a model that *underfits the test fold's
mean* scores R² < 0 by construction. So the nested runner's **−0.028…−0.19 band
is a protocol artifact** — the cost of fixed-budget refitting on a no-signal
target — **not** evidence that identity *anti*-predicts VWM. Note pooled ≈
mean-of-folds in every row: the negativity is in the predictions, not the
aggregation.

## 4. E3 — the −0.028 is not reproducible; the "≈0" picture is

Re-running **360785's exact config** (full 10×5 nested CV, 20-trial inner HPO
over `gcn_embedding_dim`, `hpo_metric=val_r2`) on the cluster at three seeds,
deploy `c36b47c` (so per-fold predictions survive under ADR-0012, N=9,730 OOF
predictions each):

| Job | Seed | R² mean-of-folds | R² pooled (N=9730) | R² pooled per-rep ± std | Pearson r (mof) | MAE (mof) | RMSE (mof) |
|---|---|---|---|---|---|---|---|
| 360854 | 42 | −0.0106 ± 0.167 | −0.0096 | −0.0096 ± 0.075 | 0.288 ± 0.100 | 0.553 ± 0.051 | 0.715 ± 0.059 |
| 360855 | 100 | +0.0085 ± 0.177 | +0.0083 | +0.0083 ± 0.079 | 0.291 ± 0.096 | 0.549 ± 0.054 | 0.708 ± 0.061 |
| 360856 | 200 | −0.0599 ± 0.257 | −0.0540 | −0.0540 ± 0.102 | 0.269 ± 0.107 | 0.568 ± 0.071 | 0.729 ± 0.075 |
| *360785* | *42* | *−0.028 ± 0.181* | *n/a¹* | *n/a* | *0.260 ± 0.115* | *0.560 ± 0.053* | *0.720 ± 0.059* |

¹ Original predictions overwritten pre-ADR-0012 (shared `checkpoints/` path);
only mean-of-folds survives in `EXPERIMENTS.md`.

Two reproducibility readings, both negative for "the −0.028 is a stable number":

- **Across seeds**, mean-of-folds spans −0.060…+0.009 (pooled −0.054…+0.008) — a
  ~0.07 R² band straddling zero. None is meaningfully different from the others
  or from the E1 floor.
- **At the *same* seed**, 360785 (seed 42, deploy `c2e973b`) and 360854 (seed 42,
  deploy `c36b47c`) differ **−0.028 → −0.011** (Δ +0.017) even though the only
  source change between those deploys is the ADR-0012 checkpoint **path**
  (`git diff c2e973b c36b47c -- src/` touches only `run_identity.py` (new) and
  one path line in `nested_cross_validation.py`; the training computation is
  byte-identical). Same seed + same code + different result ⇒ the pipeline is
  **nondeterministic run-to-run**, exactly the failure dissected in the
  2026-05-29 instability report (PyG scatter/gather atomic-adds are not seeded by
  `torch.use_deterministic_algorithms`, and the single-holdout inner HPO
  amplifies that noise into different selected configs).

Usefully, the **per-rep pooled std** (0.075–0.102) — pooled R² recomputed within
each of the 10 reps — is *larger* than the cross-seed spread of the headline
(~0.06). The run-to-run irreproducibility of the reported number is therefore
dominated by ordinary rep-level Monte-Carlo noise, not by anything systematic
across seeds. The qualitative conclusion — **≈0 with large per-fold scatter** —
is the only robust statement.

## 5. Reconciliation — what the remembered "0.1" most likely was

Putting the confounds back together:

- **Protocol (flat vs nested):** E1 (flat, the matching protocol) gives +0.009;
  E2 (nested, no HPO) gives −0.19. Switching to the nested protocol moves the
  number *down*, not up — it cannot manufacture a +0.10. ✗
- **Aggregation (pooled vs mean-of-folds):** identical to ≤0.006 in all 13 runs
  across E1–E3. Not the hidden 0.10. ✗
- **Seed/run noise:** spans ≈0.07 R² but centered on zero; the best single draw
  observed anywhere (E3 fold-level reaches +0.19, seed-100 mean +0.008) still
  never approaches +0.10 at the run level. A lucky draw is an unlikely full
  explanation. ✗ (as a sole cause)

The remaining, and most likely, origin is **validation optimism**: the inner HPO
selects hyperparameters on a `val_r2` that is positive (≈+0.1 during live HPO
monitoring of E3; cf. `val_test_optimism_gap`, where nested val R² overstates
held-out test R² by ~0.13), and a pipeline that *reported its selection/val
metric rather than a held-out pooled test R²* would surface ≈+0.1. If the lost
codebase quoted a validation or selection-time R² (or a single favorable
flat-CV draw of this same noise-dominated estimator), that reconciles cleanly
with everything here without any identity→VWM signal existing. This aligns with
the broader finding for this pipeline (`vwm_glm_rerun_pooled`): **its headline
R²s are not run-to-run reproducible and val-time numbers do not survive to
test.**

## 6. What survives / what does not

- **Not supported:** "identity (one-hot) node features predict VWM at R²≈0.1."
  No protocol, aggregation, or seed in E1–E3 reaches it; the faithful
  reconstruction lands at +0.009 ± 0.015.
- **Robust:** the identity→VWM **generalization floor is ≈ 0** (E1 +0.009; E3
  −0.02…+0.01 pooled). Pure node distinctness carries no transferable
  `VWM_overall_dprime` signal under GCN on PNC SC. This corroborates the
  2026-06-01 value-permutation batch's degeneracy-guard firing (arm 1 `identity`
  ≈ 0).
- **Robust (negative):** the nested runner's sub-zero readings (−0.028…−0.19) are
  a **refit/HPO-noise artifact**, and the specific value is **not** run-to-run
  reproducible — so −0.028 should never be cited as *the* identity→VWM number.

## 7. Provenance / how to reproduce

- **E3 artifacts:** `cluster-fetch checkpoints/gcn-pnc-sc-vwm-identity-repro-seed<S>-<JOB>/nested_cv_result.json`
  (paths are **relative to** `cluster.project_root`), fetched to
  `tmp/r2decomp/e3/seed{42,100,200}.json`. Jobs 360854/360855/360856, all
  `COMPLETED 0:0`, ~16 h each, gpunode01/rtx2080, wandb `orbitglm/teampolpetta`.
- **E1/E2 artifacts:** `tmp/r2decomp/e1/seed{42..46}/` and `…/e2/…` (gitignored —
  numbers live here and in `EXPERIMENTS.md`). Regenerate the tables with
  `for f in $(find tmp/r2decomp -name nested_cv_result.json|sort); do PYTHONPATH=$(pwd) .venv/bin/python scripts/pooled_vs_meanfolds.py "$f"; done`.
  All local runs need `dataset.root=/media/compa/DATA2/Compa/DATA_DERIVATIVES/PNC`.
- **Both R² flavours** for any run: `scripts/pooled_vs_meanfolds.py <json>`
  (prints mean-of-folds, pooled-all-folds, per-rep pooled, per-fold, and pooled
  MAE/RMSE/pearson from the per-fold `y_true`/`y_pred` arrays).
- **Nondeterminism mechanism** (§4): see
  [`reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md`](2026-05-29-vwm-glm-node-features-run-to-run-instability.md)
  §3 (GPU scatter/gather non-determinism; single-holdout inner HPO amplification)
  and its recommendations (`torch.use_deterministic_algorithms(True)`; full inner
  k-fold HPO).

## 8. Note on test suite (out of scope for this PR)

Two pre-existing failures on this branch —
`tests/test_training.py::TestNestedCrossValidator::test_fast_preset_smoke` and
`test_hpo_preset_smoke` — assert the old `<td>/nested_cv_result.json` location
(`assert saved.exists()`), but ADR-0012 (PR #30, present in this branch's base
`14ce6b2`) now writes under `<td>/<run_name>/`. The failures stem entirely from
that base change, are unrelated to this branch's work, and are left untouched
(stale assertions, not regressions). The **18** tests added by Deliverable A
(5 in `tests/test_cross_validation_artifact.py` + 13 in
`tests/test_run_experiment_routing.py`) all pass.
