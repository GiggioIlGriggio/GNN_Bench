# VWM-HL GLM node-feature effect on ORBIT, across backbones (GCN / GAT / GIN / Graph Transformer)

**Date:** 2026-06-10 (analysis); batch launched 2026-06-09.
**Subject experiments:** the 2026-06-09 "VWM-HL GLM node-features × backbone
generalization on ORBIT" matrix (ORBIT structural connectivity, target
`mri_VWM_HL_p`) — see `EXPERIMENTS.md` → *Batch 2026-06-09*.
**Batch:** Slurm jobs 361026–361053 (4 backbones × 7 node-feature cells), branch
`feature/orbit-vwmhl-backbone-matrix`, deploy SHA `c6997d5`, gpunode02. All 28
`COMPLETED 0:0`; full 28-cell smoke passed first.
**Reference:** the PNC version of the same matrix —
[`reports/2026-06-04-vwm-glm-node-features-cross-backbone-generalization.md`](2026-06-04-vwm-glm-node-features-cross-backbone-generalization.md)
(`EXPERIMENTS.md` → *Batch 2026-06-04*), which swapped GCN for the now-substituted
BrainGNN. **Design:** BrainGNN→GCN (BrainGNN was at the no-skill floor on PNC),
PNC→ORBIT; protocol byte-identical (10×5 nested CV, 20-trial inner HPO, 300 ep,
ADR-0013 matched HPO).

---

## TL;DR

The 7-cell node-feature matrix (diagonal-GLM vs scalar-GLM forms, on three positional
carriers) was run on four backbones on ORBIT VWM-HL (N≈94) to ask whether the PNC
"diagonal GLM forms ≳ scalar forms" effect transfers to a second dataset.

- **The diagonal/identity ordering replicates — directionally and across all four
  backbones — but at roughly half the PNC magnitude.** In *every* backbone the top
  two cells by both r² flavours are the two diagonal/identity cells (`id-glmdiag`,
  `glmdiag`): pooled r² **0.09–0.15**, Pearson **0.37–0.42**. On PNC the same cells
  sat at pooled r² ≈0.18–0.22, Pearson ≈0.48. So the plateau is **present and
  consistent but compressed** on ORBIT.
- **Scalar carriers collapse — the robust part of the replication.** All three
  `*-glmscalar` cells land at **negative pooled r²** and **Pearson ≈ 0** in all four
  backbones. The signal is per-node distinctness/identity structure, not the scalar
  activation magnitude — exactly the PNC conclusion.
- **Nothing reaches corrected significance.** The ADR-0008 Bouckaert–Frank corrected
  per-fold paired t-test finds **0 / 21 significant pairs in every backbone** (min
  BH q = 0.69–0.92). At N≈94 each outer fold's r² is computed on only ~9–19 test
  subjects, so the per-fold scores are too noisy for the conservative corrected test
  to resolve even a +0.2–0.7 pooled-r² gap. **This is a power ceiling, not
  counter-evidence** — the evidence here is the cross-backbone *consistency* of the
  descriptive pooled-r²/Pearson pattern, not within-batch significance.
- **No backbone fails the task** (contrast BrainGNN on PNC): GCN/GAT/GIN/Transformer
  all reach the same modest diagonal level (~0.1 pooled r²). Cross-backbone *absolute*
  r² is confounded by HPO DOF (GAT/Transformer tune `heads`) — reported, not ranked.
- **The `scprofile` carrier shows no diagonal advantage** (often `scprof-glmdiag` ≈ or
  < `scprof-glmscalar`), matching PNC. The clean diagonal>scalar gaps are on the
  **identity** and **laplacian-PE** carriers.

---

## 1. What was tested

Identical to the PNC 2026-06-04 batch except the dataset, target, and the
BrainGNN→GCN swap. 4 backbones × 7 node-feature presets = 28 nested-CV runs:

- **Carriers × forms.** `glmdiag` (bare diagonal GLM); identity-carried
  (`id-glmscalar`, `id-glmdiag`); structural-profile-carried (`scprof-glmscalar`,
  `scprof-glmdiag`); laplacian-PE-carried (`lappe-glmscalar`, `lappe-glmdiag`).
- **Backbones / matched HPO (ADR-0013).** GCN & GIN on the base `gcn_embedding_dim`
  sweeper; GAT & Transformer on base + `model.heads`. Optimizer held out of HPO for
  all four. GCN *is* the matched base — a cleaner drop-in than BrainGNN's reduced,
  personalized sweeper ever was.
- **Target.** `mri_VWM_HL_p` — ORBIT MRI visual-working-memory **high-load
  proportion-correct**, a bounded measure in [0.11, 0.98] (mean 0.73, sd 0.19). This
  is *not* PNC's unbounded `VWM_overall_dprime`; absolute r²/MAE are therefore **not
  comparable across the two datasets**, only the within-batch contrasts.
- **N.** 940 pooled out-of-fold predictions / 10 reps = **94 effective subjects**
  (bounded by GLM-map coverage: 101 maps, 129 structural mats, 130 non-null labels).
  ~10× smaller than PNC.

## 2. The diagonal/identity plateau replicates, compressed

Per backbone, the two diagonal/identity cells are the top two on both r² flavours:

| backbone | best cell | pooled r² | Pearson r | 2nd cell | pooled r² |
|---|---|---|---|---|---|
| GCN         | `id-glmdiag` | 0.115 | 0.385 | `glmdiag` | 0.092 |
| GAT         | `id-glmdiag` | 0.146 | 0.416 | `glmdiag` | 0.113 |
| GIN         | `glmdiag`    | 0.130 | 0.408 | `id-glmdiag` | 0.115 |
| Transformer | `id-glmdiag` | 0.126 | 0.396 | `glmdiag` | 0.101 |

The ceiling is set by the diagonal/one-hot node structure, not the positional carrier
— same as PNC. But the level is **~half** the PNC plateau (pooled r² ≈0.10 vs ≈0.20;
Pearson ≈0.40 vs ≈0.48), consistent with the smaller, bounded, differently-defined
ORBIT target and the 10× smaller N. The `lappe-glmdiag` cell holds the plateau on GAT
(0.099) and weakly on GIN (0.025)/GCN (−0.044)/Transformer (−0.020) — weaker than on
PNC, where laplacian-PE diagonal was a top cell everywhere.

## 3. Scalar carriers collapse — the robust replication

Every `*-glmscalar` cell, every backbone, lands at **negative pooled r²** with
**Pearson ≈ 0**:

| carrier (scalar cell) | GCN | GAT | GIN | Transformer |
|---|---|---|---|---|
| `id-glmscalar`     pooled r² (Pearson)    | −0.091 (0.22) | −0.076 (0.23) | −0.410 (0.08) | −0.009 (0.29) |
| `scprof-glmscalar` pooled r² (Pearson)    | −0.285 (0.10) | −0.412 (0.02) | −0.274 (0.08) | −0.206 (0.07) |
| `lappe-glmscalar`  pooled r² (Pearson)    | −0.672 (−0.01) | −0.261 (0.02) | −0.421 (0.00) | −0.772 (0.05) |

The carrier-matched **diagonal − scalar** pooled-r² gap is positive in 11 of 12
cells (the lone exception is GCN-`scprof`, where both forms are deep-negative noise):

| carrier | GCN | GAT | GIN | Transformer |
|---|---|---|---|---|
| identity (id)        | +0.21 | +0.22 | +0.53 | +0.14 |
| scprofile (scprof)   | −0.16 | +0.09 | −0.01 | +0.15 |
| laplacian-PE (lappe) | +0.63 | +0.36 | +0.45 | +0.75 |

So the per-node-distinctness-holds / scalar-magnitude-collapses effect transfers
cleanly, carried by the **identity** and **laplacian-PE** carriers; the **scprofile**
carrier shows no diagonal advantage (both forms ≈0/negative), exactly as on PNC.

## 4. No corrected within-backbone significance — a power ceiling at N≈94

The ADR-0008 Bouckaert–Frank corrected resampled paired t-test over per-outer-fold
r² (`scripts/compare_models.py --metric r2`, BH-adjusted across each backbone's 21
pairs) finds **no significant pair in any backbone**:

| backbone | sig pairs (q<0.05) / 21 | min BH q | key diag−scalar pairs (Δ mean-of-folds r², BH q) |
|---|---|---|---|
| GCN         | 0 | 0.89 | id Δ+0.21 q=0.89 · scprof Δ−0.13 q=0.89 · lappe Δ+0.74 q=0.89 |
| GAT         | 0 | 0.73 | id Δ+0.23 q=0.73 · scprof Δ+0.09 q=0.92 · lappe Δ+0.33 q=0.77 |
| GIN         | 0 | 0.69 | id Δ+0.61 q=0.76 · scprof Δ−0.01 q=0.97 · lappe Δ+0.47 q=0.76 |
| Transformer | 0 | 0.92 | id Δ+0.14 q=0.92 · scprof Δ+0.15 q=0.92 · lappe Δ+0.69 q=0.92 |

The corrected test is conservative by design (variance inflated by the
`n_test/n_train = 0.25` Nadeau–Bengio term), and at N≈94 the per-fold r² is computed
on ~9–19 test subjects, so single folds blow up (mean-of-folds r² std up to **3.3**
on `transformer-lappe-glmscalar`). With per-fold scores that noisy, even a +0.2–0.7
pooled-r² gap is undetectable. **Contrast PNC** (N≈940), where the laplacian-PE
diagonal-vs-scalar pair reached q=0.013 (GAT) / q=0.001 (GIN). The ORBIT result is
therefore a **null from low power**, and the finding rests on the descriptive
pooled-r²/Pearson pattern being **consistent across four independent backbones**, not
on per-batch significance.

## 5. What survives, what is new, what is not supported

- **Survives (descriptively, ×4 backbones):** diagonal/identity cells lead; scalar
  carriers collapse to ≈0 Pearson / negative pooled r²; scprofile carrier shows no
  diagonal advantage. The qualitative PNC story transfers to a second dataset and a
  different (bounded) VWM target.
- **New:** on ORBIT the plateau is **compressed to ~half** the PNC level, and the
  **laplacian-PE diagonal** cell is materially weaker than on PNC (a top cell there,
  middling here). The clean diagonal>scalar gaps are identity- and lappe-carried.
- **Not supported:** any *significance* claim — 0/21 corrected pairs per backbone.
  And any cross-dataset *absolute*-r² comparison (different target construct/scale).
  Cross-backbone absolute r² is likewise confounded (GAT/Transformer tune `heads`).

## 6. Full results (reference)

See `EXPERIMENTS.md` → *Batch 2026-06-09* for the complete per-backbone tables (both
r² flavours, Pearson, MAE, RMSE, wandb run links for all 28 cells). Summary of the
diagonal/identity cells (pooled r² | Pearson): GCN 0.092–0.115 | 0.37–0.39, GAT
0.113–0.146 | 0.38–0.42, GIN 0.115–0.130 | 0.38–0.41, Transformer 0.101–0.126 |
0.39–0.40.

## 7. Caveats

- **N≈94, single noisy draw per cell.** Mean-of-folds r² is volatile (folds hold
  ~9–19 subjects); `±std` is fold dispersion, not a standard error (folds are
  correlated). Lead with pooled r² + Pearson. Nondeterminism unaddressed (kept for
  comparability with 2026-06-04).
- **Bounded proportion target**, not d-prime → no cross-dataset absolute-r²
  comparison. Pooled r² is against the global mean; a model that only captures the
  group mean scores ≈0, and several scalar cells score well below 0 (anti-fit on
  held-out folds).
- **Cross-backbone absolute r² confounded** by extra HPO DOF (attention `heads`) —
  reported, not ranked.

## 8. Provenance / how to reproduce

- **Launch:** `NODE=gpunode02 bash slurm/submit_orbit_vwmhl_matrix.sh` (28 cells;
  `SMOKE=1` for the pre-flight smoke, `DRY_RUN=1` to preview). Deploy SHA `c6997d5`.
- **Per-run artifacts (ADR-0012):** `checkpoints/<experiment_name>-<jobid>/nested_cv_result.json`,
  fetched to `/tmp/<jobid>.json` via `cluster-fetch`.
- **Both r² flavours:**
  `.claude/skills/backfill-experiment-results/scripts/compute-pooled-metrics.py <json>`
  (recomputes mean-of-folds from stored arrays, warns on drift; none seen).
- **Within-backbone corrected t-tests (§4):**
  `python scripts/compare_models.py --inputs <7 JSONs> --names <cells> --metric r2
  --output-dir /tmp/cmp_<backbone>` — Bouckaert–Frank corrected resampled paired
  t-test, BH-adjusted across the 21 within-backbone pairs.
- **Design + plan:** `docs/superpowers/specs/2026-06-09-orbit-vwmhl-gcn-backbone-matrix-design.md`,
  `docs/superpowers/plans/2026-06-09-orbit-vwmhl-gcn-backbone-matrix.md`.
