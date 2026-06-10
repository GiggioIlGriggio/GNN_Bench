# Design — Classical ML connectivity / GLM baselines

**Date:** 2026-06-10
**Status:** approved (design), pending implementation plan
**Author:** brainstorming session (Claude + user)

## 1. Motivation

The GNN batches (PRs #32, #34) measured how node-initial features × backbone
affect SC→working-memory regression. We now need **classical (non-graph) ML
baselines** that learn directly from connectivity, to answer:

1. *Can a non-graph model on raw connectivity match the GNN?* — the standard
   connectome-predictive-modeling sanity check the GNN results currently lack.
2. *How much of the VWM score is predictable from the GLM activations alone*
   (400 values, one per node, **no graph**)? — directly contextualizes the GLM
   node-feature GNN experiments.

**Binding constraints (user):** reuse the GNN's existing data preprocessing
wherever possible so baseline-vs-GNN is apples-to-apples, and do not rewrite code
that already exists. See memory `feedback_reuse_gnn_pipeline`.

## 2. Scope & run matrix

- **Estimators:** XGBoost, ElasticNet (new sklearn path); MLP (existing torch
  model, reused).
- **Connectivity baselines:** input ∈ {SC, FC} × target ∈ {age, VWM} ×
  dataset ∈ {PNC, ORBIT} × estimator ∈ {3} = **24 runs**.
- **GLM-activation baseline:** input = `glm_scalar` (400-vec) × target = {VWM} ×
  dataset ∈ {PNC, ORBIT} × estimator ∈ {3} = **6 runs**.
- **Total = 30 runs.**

Label configs (existing): PNC age `pnc_default` (`age_at_cnb`), PNC VWM
`pnc_VWMdprime`; ORBIT age `default` (`sdi_AGE`), ORBIT VWM
`orbit_mri_VWM_HL_p`. Modality is chosen by dataset config: SC = `pnc`/`orbit`,
FC = `pnc_fc`/`orbit_fc`. GLM baseline uses `features=glm_scalar`
(`contrast-2back_vs_0back`, shared with the prior GNN GLM runs on both datasets).

The GLM-activation baseline is **VWM-only** (the GLM contrast is a WM contrast;
age is not the question there). Connectivity baselines cover both targets.

## 3. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Connectivity feature vector | The GNN's **top-20% thresholded weighted** upper-triangle (`N(N-1)/2 ≈ 79,800`), zeros where edges dropped per subject | Fairest head-to-head vs GNN; reuses the exact thresholded `Data` the GNN sees |
| MLP implementation | Reuse existing torch `model=mlp` through existing `NestedCrossValidator` + `Trainer` | Zero new model code; same Trainer as the GNN ⇒ most directly comparable; reuses `configs/sweeper/mlp.yaml` |
| XGBoost + ElasticNet | New `SklearnNestedCrossValidator` (approach **B**) reusing leaf components | sklearn estimators cannot run through the epoch-coupled torch path; B keeps the GNN runner untouched and guarantees identical folds |
| Protocol | 10 reps × 5 outer folds × 20 inner Optuna trials, maximize `r2` | Mirrors GNN batches; output feeds `scripts/compare_models.py` |
| Feature standardization | Per-fold `StandardScaler` fit on **train only**, applied to all sklearn estimators | Required for ElasticNet; harmless for XGBoost; leakage-safe |
| wandb project | **`baselines`** (new project) | Isolate baselines from GNN sweeps |
| Compute | Cluster: sklearn on **CPU nodes**, torch MLP on GPU nodes | sklearn is CPU-only and must not consume GPU allocation. CPU-node support has **landed** in the cluster-helper skill (user, 2026-06-10) — see §10 |

## 4. Components — reuse vs refactor vs new

**Reused unchanged:** dataset loaders (PNC/ORBIT, `modality=sc|fc`),
`feature_builder` + top-20% thresholding, `FoldBarrier` (label norm + GLM-column
norm), `src/training/metrics.py:compute_metrics`, `NestedCVResult`/`FoldResult`,
`WandbLogger.log_nested_*`, `TrainerConfig` knobs
(`resolved_outer_seeds`, `effective_n_outer_folds`, `stratify_bins`,
`inner_hpo_trials`, `hpo_metric`), `configs/sweeper/mlp.yaml`,
`scripts/compare_models.py`.

**Two small extractions (behavior-preserving refactor — one impl shared by both
paths):**
- `src/training/splits.py` ← the binning + outer/inner split logic currently
  inside `NestedCrossValidator` (`_stratify_bins`, the outer `StratifiedKFold`
  enumeration keyed by `resolved_outer_seeds()`, `_inner_split` with
  `inner_seed = outer_seed*1000 + fold`). Both runners call it ⇒ **byte-identical
  folds** to the GNN.
- `src/models/flatten.py` ← the adjacency vectorization currently in
  `MLPBrainModel._build_adjacency_vector`, plus a node-feature flatten. MLP +
  sklearn build the **same** vectors.

**New:**
- `src/training/sklearn_nested_cv.py` — `SklearnNestedCrossValidator`: a simple,
  no-epoch fold loop (see §5).
- `src/models/sklearn_baselines.py` — registry mapping `xgboost`, `elasticnet`
  → `(estimator_builder(params) -> sklearn estimator, search_space adapter)`.
- `configs/model/xgboost.yaml`, `configs/model/elasticnet.yaml` — carry `name`,
  `kind: sklearn`, base HPs, and `input: adjacency | node_features`.
- `configs/sweeper/xgboost.yaml`, `configs/sweeper/elasticnet.yaml` — HP search
  spaces (§6), reusing the existing `search_space` YAML schema parsed by
  `src/training/search_space.py`.

**Modified (small, additive):**
- `scripts/run_experiment.py:select_runner()` — route `model.kind == "sklearn"`
  → `SklearnNestedCrossValidator`. One branch; GNN path untouched.
- `MLPBrainModel` — call `flatten.py` helpers (identical behavior).
- `NestedCrossValidator` — call `splits.py` helpers (identical behavior).

## 5. Data flow — one sklearn fold

1. Hydra builds cfg → `dataset = get_dataset(...)` → `List[Data]` (feature-built,
   thresholded); `labels: np.ndarray`.
2. `SklearnNestedCrossValidator.run`: `bins = splits.stratify_bins(labels)`;
   for each `outer_seed` in `resolved_outer_seeds()`: outer `StratifiedKFold`;
   per fold: `splits.inner_split(...)`.
3. Per outer fold:
   - `FoldBarrier.fit(train_graphs, train_labels)` — reuse GNN's GLM-column
     normalization + label normalization.
   - Apply `barrier.transform_graphs` then flatten train/val/test to `X` via
     `flatten.py` (adjacency-weighted upper-tri for connectivity, or node-feature
     flatten for GLM).
   - `StandardScaler` fit on `X_train`, applied to all splits.
   - Inner HPO: in-process Optuna study (same pattern as
     `NestedCrossValidator._run_inner_hpo`) over the sweeper search space;
     objective = fit `estimator(params)` on inner-train, predict inner-val,
     return `r2` (maximize).
   - Refit best params on outer-trainval (one shot — no epochs); predict
     outer-test; `barrier.inverse_transform_labels`; `compute_metrics`.
   - Emit `FoldResult`; `logger.log_nested_outer_test` / `log_nested_best_hparams`.
4. Aggregate → `NestedCVResult.save()` (same JSON schema as the GNN runs).

## 6. Estimators & search spaces

- **ElasticNet** (`sklearn.linear_model.ElasticNet`): `alpha` log-uniform
  `[1e-3, 10]`, `l1_ratio` uniform `[0.05, 0.95]`, `max_iter=5000`. On
  standardized features.
- **XGBoost** (`xgboost.XGBRegressor`, `tree_method=hist`, `n_jobs=<cpus>`):
  `n_estimators` `[200, 1000]`, `max_depth` `[2, 6]`, `learning_rate` log
  `[0.01, 0.3]`, `subsample` `[0.6, 1.0]`, `colsample_bytree` `[0.3, 1.0]`,
  `min_child_weight` `[1, 10]`, `reg_lambda` log `[1e-2, 10]`.
- **MLP** (existing): `model=mlp`, `configs/sweeper/mlp.yaml`;
  `mlp_input=adjacency` (connectivity) or `node_features` (GLM),
  `mlp_adjacency_type=weighted`.

## 7. Execution plan

- Submit via `cluster-submit` like the GNN batches, but sklearn jobs target a
  **CPU node** (`--node <cpunode>`, see §10); torch MLP jobs target a GPU node.
- **Dependency:** add `xgboost` to requirements and `cluster-push-container`
  (sklearn already present). Confirm the local `.venv` for any local smoke (see
  memory `llm_venv_pyg_companion_libs`).
- `logging.project=baselines`.
- Naming: `<est>-<dataset>-<sc|fc|glm>-<age|vwm>` — e.g. `xgb-pnc-sc-vwm`,
  `enet-orbit-fc-age`, `mlp-pnc-glm-vwm`.
- **Two-stage validation before the full launch:**
  1. **Pipeline smoke** (1 rep / 2 fold / 2 trial, wandb off) over every
     estimator × input × dataset path — proves it runs end-to-end and the deploy
     is fresh. Fast, throwaway (`smoke-*` names).
  2. **r² sanity check** (the user's requirement): a few **real-ish** cells run
     long enough to produce a trustworthy r², then compared against expectations
     (§8a) to catch *silent correctness bugs* (label misalignment, wrong feature
     vector, leakage) that a "it ran" smoke would miss. Only after this passes do
     we launch the full 30-run matrix.
- Backfill results into `EXPERIMENTS.md` + a `reports/` report; run
  `compare_models.py` for baseline-vs-GNN stats.

## 8. Testing

- **Unit:** `flatten.py` dims (`tri_size = N(N-1)/2`; node-feature flatten
  `N*node_feat_dim`); `splits.py` produces **byte-identical fold indices** to the
  current `NestedCrossValidator` (regression test against the pre-refactor
  behavior).
- **Integration:** a tiny sklearn run (1 rep / 2 fold / 2 trial) on a small/
  synthetic dataset completes and writes a valid `NestedCVResult` JSON loadable
  by `NestedCVResult.load`.
- **Determinism:** assert the sklearn runner and the GNN runner enumerate the
  same `(rep, fold)` index sets for the same `labels` + seeds.

## 8a. r² sanity expectations (pre-launch gate)

Before the full matrix, run a small set of **anchor cells** at a real-ish budget
(e.g. 3 reps / 5 folds / ~10 trials) and confirm the pooled r² is roughly where
domain knowledge says it should be. The point is to catch silent wiring bugs, not
to chase exact numbers. Gate criteria:

| Anchor cell | Expectation | Fails if |
|---|---|---|
| **FC → age, PNC** (ElasticNet **and** XGBoost) | **Clearly positive**, substantial (functional connectivity predicts age strongly; literature r² ≳ 0.3–0.5, ElasticNet often strong) | r² ≈ 0 or negative ⇒ label misalignment / wrong feature vector / leakage-inverted |
| **SC → age, PNC** | Positive, typically lower than FC→age | strongly negative |
| **GLM → VWM, PNC** (ElasticNet) | Mildly positive, in the neighbourhood of the GNN GLM cells (GNN diagonal pooled r² ~0.18–0.22; a linear/tree model on the 400-vec should land *somewhere positive*, ~0.05–0.2) | strongly negative |
| **Any ORBIT cell** | High variance at N≈95; judge on **pooled r² + Pearson**, not mean-of-folds | use as direction-only, not a hard gate |

Anchor primarily on **FC→age (PNC)**: it is the most predictable target and the
cleanest "is the pipeline correct" signal. If age is recovered well but VWM is
near zero, that's a *scientific* result, not a bug. If even age fails, **stop and
debug** before launching anything.

## 9. Risks / notes

- **p ≫ n:** ~79,800 features vs N≈940 (PNC) / N≈95 (ORBIT). ElasticNet/XGBoost
  handle it; expect ElasticNet to dominate on ORBIT. Report leans on pooled r²
  + Pearson on small-N ORBIT (same caveat as the GNN batches).
- ORBIT VWM-HL label is bounded proportion-correct, not PNC's unbounded d-prime
  ⇒ absolute r² not comparable across datasets; only within-dataset
  baseline-vs-GNN contrasts transfer.
- StandardScaler on a ~79,800-dim sparse-ish vector is fine in memory at this N.

## 10. CPU-node support (landed)

sklearn jobs run on CPU nodes and must not consume GPU allocation. CPU-node
support has been **added to the cluster-helper skill** (user, 2026-06-10; the
exploration/integration originally handed off via
`/tmp/handoff-cluster-cpu-nodes.md`). The plan should confirm the current CPU
node/partition/QoS names from the updated skill (`cluster-gpus`/`cluster-cpus`
listing) before submitting. The torch MLP runs continue to target GPU nodes.
