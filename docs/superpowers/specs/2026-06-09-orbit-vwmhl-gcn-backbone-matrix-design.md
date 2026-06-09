# Design — VWM-HL GLM node-features × backbone generalization on ORBIT (GCN/GAT/GIN/Transformer)

**Date:** 2026-06-09
**Status:** Approved (brainstorming) — pending implementation plan
**Branch:** `feature/orbit-vwmhl-backbone-matrix`
**Lineage:** reproduces [Batch 2026-06-04 — VWM GLM node-features × backbone
generalization](../../../EXPERIMENTS.md) (ADR-0013, SHA `43907c7`) with two
substitutions: **BrainGNN → GCN** (BrainGNN is at the no-skill floor / "not
working") and **PNC → ORBIT**.

## 1. Goal

Re-run the 7-cell node-feature matrix across four backbones on the **ORBIT**
dataset, asking the same *within-batch* question the original PNC batch asked:

- Does the **diagonal-GLM ≳ scalar-GLM** node-feature effect replicate when the
  backbone is swapped?
- Does the **diagonal-GLM r² plateau** (~0.18 on PNC) appear on ORBIT VWM-HL?

This establishes the ORBIT GCN reference *and* its cross-backbone generalization
in one batch (on PNC those were two separate batches: 2026-06-03 GCN reference +
2026-06-04 other backbones).

**Non-goal:** cross-dataset absolute-r² comparison. ORBIT VWM-HL is a bounded
proportion-correct measure, PNC was unbounded d-prime — absolute r²/MAE are **not
comparable across datasets**. Only the within-batch contrasts transfer.

## 2. What changes vs the original batch

This is **config + submit + analysis only** — no training-code change, exactly
like ADR-0013.

| Axis | Original (2026-06-04) | This batch |
|---|---|---|
| Dataset | `dataset=pnc` (N ≈ 940) | `dataset=orbit` (N ≈ 95, GLM-bounded) |
| Label | `labels=pnc_VWMdprime` (d-prime, unbounded) | `labels=orbit_mri_VWM_HL_p` (proportion, [0.11, 0.98]) |
| Backbones | GAT, GIN, Transformer, **BrainGNN** | GAT, GIN, Transformer, **GCN** |
| Sweeper for the swapped slot | `braingnn_vwm_matched` | `gcn_embedding_dim` (the matched base itself) |
| Protocol | 10×5, 20 inner trials, 300 ep | **identical** (byte-for-byte) |

## 3. Matrix (28 cells)

4 backbones × 7 node-feature presets. Suffix → preset mapping is **identical** to
the PNC batch:

| suffix | features preset |
|---|---|
| `glmdiag` | `glm_diagonal` |
| `id-glmscalar` | `identity_glm_scalar` |
| `id-glmdiag` | `identity_glm_diagonal` |
| `scprof-glmscalar` | `scprofile_glm_scalar` |
| `scprof-glmdiag` | `scprofile_glm_diagonal` |
| `lappe-glmscalar` | `laplacian_pe_glm_scalar` |
| `lappe-glmdiag` | `laplacian_pe_glm_diagonal` |

## 4. Matched HPO (ADR-0013, GCN dropped into BrainGNN's slot)

Shared architectural base, optimizer held out of HPO for every backbone:

| backbone | sweeper | personalization |
|---|---|---|
| GCN | `gcn_embedding_dim` | none — GCN **is** the matched base |
| GIN | `gcn_embedding_dim` | none (base) |
| GAT | `gat_embedding_dim` | base + `model.heads: choice(1,2,4,8)` |
| Transformer | `transformer_embedding_dim` | base + `model.heads` |

GCN slotting in is *cleaner* than BrainGNN ever was: BrainGNN needed a reduced,
personalized `braingnn_vwm_matched` sweeper (dropped `num_layers`/`pooling`/
`jk_mode`, added `model_params`); GCN uses the unmodified base, so all four
backbones now share the same base knobs with only GAT/Transformer adding `heads`.

## 5. Shared recipe (per cell)

Identical to 2026-06-04 except `dataset`, `labels`, backbone, sweeper:

```
dataset=orbit model=<backbone> labels=orbit_mri_VWM_HL_p
features=<preset> features.glm_normalize=true
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
trainer.epochs=300 trainer.search_space=configs/sweeper/<sweeper>.yaml
trainer.hpo_metric=val_r2 logging.project=orbitglm
```

- **wandb:** project `orbitglm`, entity `teampolpetta` (unchanged).
- **Naming:** `<backbone>-orbit-sc-vwmhl-<suffix>`, e.g.
  `gcn-orbit-sc-vwmhl-id-glmdiag`.
- **Slurm:** `--time=2-00:00:00`; node by GPU availability at submit time.
  Expected far faster than PNC (graphs are 400-node, N ≈ 95).
- **Cluster data:** ORBIT is pre-uploaded at
  `/data/bdip_ssd/al5165/GNNBenchV2/data/ORBIT` (13 GB); `slurm/train.sh` already
  pins `dataset.root` for `dataset=orbit` via its `RUN_ARGS` case block.

## 6. Smoke gate (mandatory, before the 28)

The prior `checkpoints/orbit_GLM_VWMHL_gcn/` run only proves `glm_diagonal` works
on ORBIT. The identity-concat, scprofile, and laplacian-PE presets on ORBIT are
**untested**. Run a 1-rep / 2-fold / 2-epoch smoke per backbone on at least
`id-glmdiag` and `lappe-glmdiag` (the two presets most likely to surface a
node-dimension or contrast-path mismatch). Launch the 28 only after all smokes
pass `COMPLETED`.

## 7. Submission

A generated, committed `slurm/submit_orbit_vwmhl_matrix.sh` that loops the 28
`cluster-submit` calls — **one Slurm job per cell**. Rationale:

- One job per cell → ADR-0012 per-run checkpoint dirs
  (`checkpoints/<experiment_name>-<jobid>/`) → clean `backfill-experiment-results`.
- Each cell is independently re-runnable if one fails.
- A single committed script is reproducible (vs 28 hand-typed commands, which is
  what the original batch literally did).

The script also emits the 4 smoke jobs (gated behind a `--smoke` flag or a
separate small block) so the gate is reproducible too.

## 8. Analysis & write-up

Mirror the original batch exactly:

1. **Per-backbone results tables** — mean-of-folds **and** pooled r² + Pearson r +
   MAE + RMSE, recovered via `backfill-experiment-results` (wandb + per-run
   checkpoints).
2. **Within-backbone significance** — ADR-0008 Bouckaert-Frank corrected resampled
   paired t-test over per-outer-fold r² (`scripts/compare_models.py`), BH-adjusted
   across the within-backbone pairs. Key pairs: diagonal-vs-scalar, carrier-matched
   (id / scprof / lappe).
3. **New `## Batch 2026-06-09` section in `EXPERIMENTS.md`** + a full report under
   `reports/2026-06-09-orbit-vwmhl-gcn-backbone-generalization.md`.

## 9. Caveats (baked into the write-up)

- **N ≈ 95** (bounded by GLM-map coverage: 101 maps, 129 structural mats, 130
  non-null labels → intersection ≈ 95). ~10× smaller than PNC.
- **Mean-of-folds r² is volatile** at this N. The prior ORBIT GCN glm_diagonal run
  gave per-fold r² of −0.43, +0.27, −0.27, −0.005, +0.04 (mean ≈ −0.08) while
  Pearson r stayed positive (~0.4) every fold. → lead the narrative with **pooled
  r² + Pearson r**; report mean-of-folds but flag it noisy.
- **Single-holdout inner HPO is noise-dominated** — already true at N=940 per the
  repo's own caveat; worse at N=95. (User chose byte-identical protocol over
  small-N hardening for maximal fidelity — accepted.)
- **No cross-dataset r² comparison** — bounded proportion vs d-prime.
- **Single noisy draw per cell** — nondeterminism unaddressed (kept for
  comparability, same as the original; see the run-to-run instability report).

## 10. Scope guard (YAGNI)

No training-code changes. No new feature presets. No protocol redesign. No new
sweeper configs (all four already exist). Pure config-selection + submit-script +
analysis, exactly like ADR-0013. If the smoke gate surfaces a genuine code bug in
an untested ORBIT preset path, that fix is a separate, minimal change — not part
of this batch's scope.
