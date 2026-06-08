# Cross-backbone generalization of the VWM GLM node-feature effect (GAT / GIN / Graph Transformer / BrainGNN)

**Date:** 2026-06-08 (analysis); batch launched 2026-06-04.
**Subject experiments:** the 2026-06-04 "VWM GLM node-features × backbone
generalization" matrix (PNC structural connectivity, target `VWM_overall_dprime`) —
see `EXPERIMENTS.md` → *Batch 2026-06-04*.
**Batch:** Slurm jobs 360893–360920 (4 backbones × 7 node-feature cells), branch
`feature/multi-backbone-vwm-matrix`, deploy SHA `43907c7`, **ADR-0013**, gpunode02.
All 28 `COMPLETED 0:0`.
**Reference:** the GCN re-run of the same 7-cell matrix —
`EXPERIMENTS.md` → *Batch 2026-06-03* and
[`reports/2026-05-29-vwm-glm-node-features-run-to-run-instability.md`](2026-05-29-vwm-glm-node-features-run-to-run-instability.md).

---

## TL;DR

The GCN node-feature matrix was re-run on four more backbones to ask: does the
"diagonal GLM forms ≳ scalar forms" effect depend on the GCN backbone, or generalize?

- **The diagonal-GLM `r²≈0.18–0.22` plateau replicates** in GAT, GIN, and Graph
  Transformer, matching the GCN reference. Within each of these three backbones the
  three diagonal cells (`glmdiag`, `id-glmdiag`, `lappe-glmdiag`) are **statistically
  indistinguishable** (all pairwise corrected `q ≫ 0.05`). The diagonal / one-hot node
  structure — not the positional carrier — sets the ceiling.
- **"Diagonal ≳ scalar" holds *directionally* in all three but reaches significance
  (BH-corrected) only for the laplacian-PE carrier** (GAT `q=0.013`, GIN `q=0.001`;
  Transformer same sign, `q=0.26`). That significance is driven by the **scalar form
  collapsing to a ≈0 floor**, not by an unusually strong diagonal gain. The
  identity-carrier diagonal-vs-scalar gap is positive everywhere but n.s. after
  correction (consistent with GCN's n.s. `Δ0.12`); the scprofile carrier shows no
  diagonal advantage.
- **BrainGNN does not learn the task** under the matched protocol: every cell
  `r² ∈ [−0.11, −0.02]` (best `id-glmdiag` −0.016). The node-feature contrast is
  untestable there — a backbone-level negative, not evidence against the effect.

**Net:** the effect generalizes across message-passing / attention backbones. The
robust, reproducible-across-backbones signal is **diagonal carriers holding a plateau
while scalar carriers collapse toward 0** — i.e. the signal is per-node
distinctness / structure, not the scalar magnitude. This echoes the 2026-06-01
value-permutation result. Each cell is a **single noisy draw** (nondeterminism kept —
the 2026-05-29 instability applies), so read the plateau and the collapse, not the
fine ordering.

---

## 1. What was tested

The full 7-cell node-feature matrix (`glm_diagonal` standalone, plus
`identity / scprofile / laplacian_pe × glm_scalar / glm_diagonal`) was re-run on
**four backbones** — GAT, GIN, Graph Transformer, BrainGNN — holding the 2026-06-03
protocol fixed (PNC SC → `VWM_overall_dprime`, 10×5 nested CV, 20-trial inner HPO,
`features.glm_normalize=true`, `trainer.epochs=300`, `logging.project=orbitglm`) and
changing **only** `model=` and its matched HPO sweeper. No training-code change; the
PR adds `configs/model/transformer.yaml` and three sweepers.

**Matched HPO (ADR-0013).** A shared architectural base (`gcn_embedding_dim.yaml`:
embedding_dim, hidden_dim, num_layers, dropout, pooling, jk_mode, head dims) plus
per-model personalization, with the **optimizer held out of HPO for every backbone**:
GIN uses the base unchanged; GAT and Transformer add `model.heads ∈ {1,2,4,8}`;
BrainGNN uses the applicable base (drops `num_layers` [fixed 2] and `pooling`/`jk_mode`
[no-ops]) plus its `model_params` (pool_ratio, roi_embed_dim, lambda_topk/unit/consist).
All four passed a 1-rep / 2-fold / 2-epoch smoke test (jobs 360889–360892) before launch.

This design makes the **within-backbone** diagonal-vs-scalar comparison clean (identical
protocol, only the node features change). Cross-backbone **absolute** r² is confounded —
the attention models and BrainGNN tune extra degrees of freedom the generic base lacks —
so it is reported but not used to rank backbones.

## 2. The diagonal-GLM plateau replicates across backbones

Mean-of-folds r² for the three **diagonal** cells, by backbone (GCN = 2026-06-03 re-run):

| diagonal cell | GCN | GAT | GIN | Transformer | BrainGNN |
|---|---|---|---|---|---|
| `glmdiag`       | 0.194 | 0.182 | 0.179 | 0.158 | −0.075 |
| `id-glmdiag`    | 0.218 | 0.222 | 0.177 | 0.184 | −0.016 |
| `lappe-glmdiag` | 0.213 | 0.183 | 0.193 | 0.143 | −0.032 |

In GAT, GIN and Transformer the diagonal cells land in a tight `0.14–0.22` band that
matches GCN. The **within-backbone** corrected t-test finds no diagonal cell
significantly better than another in any of these three backbones (every pairwise
`q ≫ 0.05`): which positional carrier sits underneath the diagonal GLM does not matter
once the node features are diagonal. The top diagonal cell differs by backbone
(`id-glmdiag` for GAT/Transformer, `lappe-glmdiag` for GIN) but those differences are
within the single-draw noise documented for this pipeline.

## 3. Diagonal vs scalar — significant only where the scalar carrier is degenerate

The carrier-matched **diagonal − scalar** contrast (`scripts/compare_models.py`,
Bouckaert–Frank corrected resampled paired t-test over per-fold r²; `q` = BH-adjusted
across the 21 within-backbone pairs):

| backbone | identity (id) | scprofile (scprof) | laplacian-PE (lappe) |
|---|---|---|---|
| GAT         | Δ+0.112, p=0.204, q=0.390 (n.s.) | Δ+0.030, p=0.676, q=0.788 (n.s.) | **Δ+0.190, p=0.0018, q=0.013 (sig)** |
| GIN         | Δ+0.159, p=0.019, q=0.066 (n.s.) | Δ−0.013, p=0.968, q=0.969 (n.s.) | **Δ+0.204, p=5e-05, q=0.001 (sig)** |
| Transformer | Δ+0.056, p=0.447, q=0.866 (n.s.) | Δ−0.128, p=0.744, q=0.881 (n.s.) | Δ+0.152, p=0.037, q=0.258 (n.s.) |
| BrainGNN    | Δ+0.050, p=0.662, q=0.978 (n.s.) | Δ+0.003, p=0.970, q=0.978 (n.s.) | Δ+0.047, p=0.635, q=0.978 (n.s.) |

The only BH-significant diagonal-vs-scalar pairs are the **laplacian-PE** ones in GAT
and GIN. That is best read together with the scalar-cell r²:

| scalar cell | GCN | GAT | GIN | Transformer | BrainGNN |
|---|---|---|---|---|---|
| `id-glmscalar`    | 0.098 | 0.109 | 0.018 | 0.128 | −0.066 |
| `scprof-glmscalar`| −0.060 | −0.011 | −0.099 | 0.060 | −0.113 |
| `lappe-glmscalar` | −0.037 | −0.006 | −0.011 | −0.009 | −0.079 |

`lappe-glmscalar` is a **dead ≈0 floor** in every backbone (Pearson 0.02–0.13): the
Laplacian-PE positional code carries no VWM signal on its own, and a scalar GLM weight
on it adds nothing. The significant lappe diagonal-vs-scalar gap is therefore the
diagonal plateau (~0.18) measured against that floor — a real but somewhat trivial
contrast (diagonal node features beat a dead baseline). The **identity** carrier is the
one scalar form that retains signal (`id-glmscalar` 0.10–0.13 in GCN/GAT/Transformer),
which is exactly why its diagonal-vs-scalar gap is smaller and never clears correction.
The `scprofile` scalar is noisy and near/below 0, and its diagonal counterpart is not
reliably better (Δ flips sign across backbones).

So the within-backbone story is consistent with GCN, where **no** diagonal-vs-scalar
pair was significant: adding three more backbones does not turn "diagonal beats scalar"
into a clean, carrier-general effect. What it does confirm is the **asymmetry** — scalar
carriers (except identity) collapse toward 0 while diagonal carriers hold the plateau.

## 4. BrainGNN does not learn the task

Every BrainGNN cell scores `r² ∈ [−0.11, −0.02]`; the best, `id-glmdiag`, is −0.016
(pooled −0.010). Pearson tops out at 0.33. With the model sitting at the no-skill floor,
the node-feature contrast is **untestable** there — every within-BrainGNN pair is n.s.
with `q ≈ 0.98`. This is a backbone-level failure under the matched protocol, not
evidence against the node-feature effect. Plausible causes: BrainGNN's ROI-pooling
inductive bias plus the reduced matched search space (fixed 2 layers, no pooling/jk
choice) is a poor fit for PNC SC → VWM at this sample size. Because cross-backbone
absolute r² is confounded by the differing HPO degrees of freedom (§1), this is
reported descriptively and **not** used to rank BrainGNN against the others.

## 5. What survives, what is new, what is not supported

- **Survives (replicates from GCN):** the diagonal-GLM `r²≈0.18–0.22` plateau; the
  mutual indistinguishability of the three diagonal cells within a backbone; the
  scalar-carrier collapse for `laplacian_pe` and `scprofile`.
- **New vs GCN:** the `laplacian_pe` diagonal-vs-scalar contrast, which was only
  "qualitatively robust" (untested) in the GCN report, reaches BH-significance in GAT
  and GIN — though driven by the dead scalar floor, not a larger diagonal gain.
- **Not supported:** any claim that diagonal **significantly** beats scalar for the
  *informative* (identity) carrier (positive but n.s. after correction in every
  backbone, as in GCN); any ranking of backbones by absolute r² (confounded); BrainGNN
  as informative about node features (it failed the task).

## 6. Full results (reference) — both R² flavours, all 28 cells

Mean-of-folds (average of 50 fold-local r²s, each against its own ~188-sample test
mean) and pooled (one r² over all 9,400 out-of-fold predictions vs the global mean);
they agree closely. `±` is dispersion across folds, **not** a standard error. Sorted by
mean-of-folds r² within each backbone. Recovered from wandb (entity `teampolpetta`,
project `orbitglm`) + per-run `checkpoints/<name>-<jobid>/` (ADR-0012); each pooled
run's recomputed mean-of-folds r² matched its training-log value (no drift).

| backbone | cell (job) | r² mean-of-folds | r² pooled | Pearson (mof) |
|---|---|---|---|---|
| GAT | `id-glmdiag` (360895) | 0.222 ± 0.089 | 0.226 | 0.505 |
| GAT | `lappe-glmdiag` (360899) | 0.183 ± 0.106 | 0.186 | 0.485 |
| GAT | `glmdiag` (360893) | 0.182 ± 0.101 | 0.186 | 0.473 |
| GAT | `id-glmscalar` (360894) | 0.109 ± 0.172 | 0.113 | 0.431 |
| GAT | `scprof-glmdiag` (360897) | 0.019 ± 0.157 | 0.026 | 0.344 |
| GAT | `lappe-glmscalar` (360898) | −0.006 ± 0.033 | −0.006 | 0.125 |
| GAT | `scprof-glmscalar` (360896) | −0.011 ± 0.118 | −0.006 | 0.267 |
| GIN | `lappe-glmdiag` (360906) | 0.193 ± 0.088 | 0.196 | 0.452 |
| GIN | `glmdiag` (360900) | 0.179 ± 0.096 | 0.183 | 0.461 |
| GIN | `id-glmdiag` (360902) | 0.177 ± 0.103 | 0.180 | 0.486 |
| GIN | `id-glmscalar` (360901) | 0.018 ± 0.091 | 0.020 | 0.298 |
| GIN | `lappe-glmscalar` (360905) | −0.011 ± 0.023 | −0.009 | 0.019 |
| GIN | `scprof-glmscalar` (360903) | −0.099 ± 0.358 | −0.088 | 0.287 |
| GIN | `scprof-glmdiag` (360904) | −0.111 ± 0.476 | −0.108 | 0.388 |
| Transformer | `id-glmdiag` (360909) | 0.184 ± 0.131 | 0.190 | 0.487 |
| Transformer | `glmdiag` (360907) | 0.158 ± 0.132 | 0.161 | 0.462 |
| Transformer | `lappe-glmdiag` (360913) | 0.143 ± 0.127 | 0.147 | 0.452 |
| Transformer | `id-glmscalar` (360908) | 0.128 ± 0.122 | 0.133 | 0.391 |
| Transformer | `scprof-glmscalar` (360910) | 0.060 ± 0.115 | 0.060 | 0.324 |
| Transformer | `lappe-glmscalar` (360912) | −0.009 ± 0.048 | −0.008 | 0.086 |
| Transformer | `scprof-glmdiag` (360911) | −0.068 ± 0.755 | −0.066 | 0.362 |
| BrainGNN | `id-glmdiag` (360916) | −0.016 ± 0.131 | −0.010 | 0.325 |
| BrainGNN | `lappe-glmdiag` (360920) | −0.032 ± 0.122 | −0.028 | 0.286 |
| BrainGNN | `id-glmscalar` (360915) | −0.066 ± 0.255 | −0.061 | 0.299 |
| BrainGNN | `glmdiag` (360914) | −0.075 ± 0.182 | −0.070 | 0.285 |
| BrainGNN | `lappe-glmscalar` (360919) | −0.079 ± 0.163 | −0.074 | 0.250 |
| BrainGNN | `scprof-glmdiag` (360918) | −0.110 ± 0.129 | −0.104 | 0.195 |
| BrainGNN | `scprof-glmscalar` (360917) | −0.113 ± 0.165 | −0.109 | 0.199 |

## 7. Caveats

1. **Single noisy draw per cell.** Nondeterminism is *not* addressed (kept for
   comparability with GCN); the run-to-run instability quantified in
   [`reports/2026-05-29-...`](2026-05-29-vwm-glm-node-features-run-to-run-instability.md)
   (mean-of-folds r² swinging up to 0.083 between identical re-runs; inner-HPO config
   flips in 46/50 folds) applies in full here. Read the plateau and the collapse, not
   the fine ordering.
2. **Pathological fold variance** on two diagonal-scalar-carrier cells:
   `gin-scprof-glmdiag` (±0.476) and `transformer-scprof-glmdiag` (±0.755) — individual
   folds blew up. Their means are unreliable; both already sit at/below 0, so they do
   not affect any conclusion.
3. **`±std` is fold dispersion, not a standard error** — the 50 fold scores are
   correlated (10 reps reuse the same subjects; each rep's 5 folds share training data),
   so no √N and no CI. Model-vs-model significance comes only from the ADR-0008 corrected
   test (§3).
4. **Cross-backbone absolute r² is confounded** by the differing matched-HPO degrees of
   freedom (heads, model_params); within-backbone is the only clean comparison.

## 8. Provenance / how to reproduce

- Per-run artifacts (ADR-0012): `checkpoints/<experiment_name>-<jobid>/nested_cv_result.json`,
  fetched to `/tmp/<jobid>.json` via `cluster-fetch`.
- Both r² flavours:
  `.claude/skills/backfill-experiment-results/scripts/compute-pooled-metrics.py <json>`
  (recomputes mean-of-folds from the stored arrays and warns on drift; none seen).
- Within-backbone corrected t-tests (§2, §3):
  `python scripts/compare_models.py --inputs <7 JSONs> --names <cells> --metric r2
  --output-dir /tmp/cmp_<backbone>` — Bouckaert–Frank corrected resampled paired t-test
  with `n_test/n_train = 1/(k−1) = 0.25`, BH-adjusted across the 21 within-backbone pairs.
- Mean-of-folds + wandb run IDs:
  `.claude/skills/backfill-experiment-results/scripts/fetch-experiment-results.sh <jobids>`.
- Design rationale: [`docs/adr/0013-matched-hpo-cross-backbone.md`](../docs/adr/0013-matched-hpo-cross-backbone.md).
