# Onboarding Guide — `brain-gnn-codebase`

> A modular platform for benchmarking Graph Neural Networks on brain connectomes.
> Generated from the project knowledge graph (commit `0d8698b`, analyzed 2026-06-13) ·
> 232 files · 11 architecture layers · 14-step guided tour.

## 1. Project Overview

**GNN Brain Connectivity Benchmarking Platform** — a modular, scaffolded platform for
benchmarking Graph Neural Networks on structural (SC) and functional (FC) brain
connectomes.

| | |
|---|---|
| **Primary language** | Python |
| **Other languages** | YAML (Hydra configs), Markdown (docs), Shell (Slurm), CSV, Dockerfile, TOML |
| **Core frameworks** | PyTorch · PyTorch Geometric · Hydra · Optuna · WandB · Pydantic |
| **Also uses** | scikit-learn · XGBoost · Docker · pytest |

**What it does:** Loads ORBIT/PNC connectomes → attaches pluggable node/edge features →
trains a model zoo (GCN/GAT/GIN/Graph Transformer + a BrainGNN adapter over vendored
upstream layers) → evaluates with leakage-safe nested cross-validation + Optuna HPO →
compares against classical-ML baselines (XGBoost/ElasticNet/MLP) with corrected
significance tests → logs everything to WandB.

The whole platform is **config-driven**: one entry point (`scripts/run_experiment.py`)
composes a Hydra config tree, validates it through Pydantic, and dispatches to one of
five runners (sklearn baseline, sweep, fine-tuning, flat CV, nested CV).

**Quick start:**

```bash
python scripts/run_experiment.py                          # default (GCN on ORBIT)
python scripts/run_experiment.py model=gat trainer.lr=0.0005
```

## 2. Architecture Layers (11)

| Layer | Role | Anchor files |
|---|---|---|
| **Experiment Orchestration & Scripts** | The Hydra entry point + analysis/plotting CLIs | `scripts/run_experiment.py`, `scripts/compare_models.py` |
| **Hydra Config Groups** | 55 declarative YAML configs (dataset/model/features/labels/trainer/sweeper/…) | `configs/experiment.yaml`, `configs/model/*.yaml` |
| **Configuration Schemas** | Pydantic dataclasses validating every config group | `src/configs/*_config.py` |
| **Data & Features** | Connectome loaders, registry, feature/label builders | `src/datasets/base_dataset.py`, `feature_builder.py` |
| **Model Zoo** | Backbones, fusion, heads, registry, vendored BrainGNN | `src/models/unimodal.py`, `backbones/`, `vendor/braingnn/` |
| **Training & Cross-Validation** | Trainer, nested/flat CV, leakage barrier, normalizers, stats | `src/training/nested_cross_validation.py`, `fold_barrier.py` |
| **Fine-tuning, Explainability & Sweeps** | Higher-level flows over the training core | `src/finetuning/finetuner.py`, `src/sweeps/hydra_sweep.py`, `src/gnn_explainer/explainer.py` |
| **Utilities & Logging** | Typed WandB logger, IO/seed/adapters | `src/logging/wandb_logger.py`, `src/utils/io.py` |
| **Tests** | pytest suites mirroring each subsystem | `tests/test_training.py`, `test_models.py` |
| **Documentation** | README/CONTEXT + 13 ADRs + plans/specs + dated reports | `CONTEXT.md`, `docs/adr/` |
| **Infrastructure & Ops** | Docker image + Slurm job scripts | `Dockerfile`, `slurm/*.sh`, `.cluster-helper.yaml` |

## 3. Key Concepts (the design decisions to internalize)

These are the patterns you'll see everywhere, each anchored to an ADR (`docs/adr/`):

- **Hydra + Pydantic split (ADR-0001):** Hydra composes layered YAML into one config
  object; Pydantic validates it at load time so bad runs fail loudly, not silently hours
  later. The two systems are kept deliberately separate.
- **The graph contract (ADR-0002):** Every loader emits a PyG `Data` object with pinned
  `x/edge_index/edge_attr/y` shapes + `subject_id`, enforced by `validate_graph_contract()`
  — so models need no defensive shape checks.
- **Registry pattern (ADR-0005):** Datasets, models, and composite labels register
  themselves by name via decorators. Add a new dataset/backbone by writing one class +
  tagging it `@register_*` — no edits to dispatch code.
- **No data leakage (ADR-0004) → FoldBarrier (ADR-0009):** All normalization (label,
  GLM-feature, composite) is fit on training subjects only, per fold. The `FoldBarrier`
  unifies these behind one stateful object persisted as `barrier.pt`. **This is the single
  most important correctness invariant in the repo** — small-N brain data makes leakage
  catastrophic.
- **Nested cross-validation (ADR-0008):** Repeated stratified outer test folds, each
  wrapping inner Optuna HPO, refit-on-trainval, scored on the untouched fold — with
  Nadeau–Bengio corrected paired t-tests + Benjamini-Hochberg FDR. The canonical benchmark
  protocol.
- **Stratified K-fold for continuous labels (ADR-0003):** Regression targets are
  quantile-binned (`pd.qcut`) so each fold's distribution matches the dataset.
- **Matched HPO across backbones (ADR-0013):** A shared architectural search base + per-model
  personalisation (optimiser held out) keeps within-backbone node-feature comparisons clean.
- **Vendored BrainGNN (ADR-0010):** Rather than maintain a drifting hand-rewrite, the
  published BrainGNN conv/loss layers are vendored verbatim
  (`src/models/vendor/braingnn/`, see its `PROVENANCE.md`) behind a thin adapter — fidelity
  to Li et al. 2021 over local cleverness.
- **Pooled vs. mean-of-folds R²:** A recurring distinction in `metrics.py` that several
  experiment reports hinge on — be aware these are two different aggregation conventions.

## 4. Guided Tour (recommended reading order)

1. **Project Overview** — `README.md` → `CONTEXT.md` (domain glossary + full architecture map)
2. **The Entry Point** — `scripts/run_experiment.py` + `configs/experiment.yaml` (how one script runs hundreds of variants)
3. **Typed Config Schemas** — `src/configs/*_config.py` (the vocabulary the whole platform speaks; highest fan-in)
4. **Datasets & Registry** — `base_dataset.py`, `registry.py`, `orbit_dataset.py`, `pnc_dataset.py`
5. **Features & Labels** — `feature_builder.py`, `label_builder.py`, `composite_registry.py`
6. **Model Assembly** — `base_model.py`, `unimodal.py`, `registry.py` (the encode→decode seam)
7. **Backbone Zoo** — `base_backbone.py` + `gcn/gat/gin/transformer.py` (swap only the conv operator)
8. **Heads & Fusion** — `regression_head.py`, `concat/gated/attention_fusion.py`
9. **BrainGNN: the special case** — `braingnn_model.py` + ADR-0010
10. **Training Loop & Leakage Barrier** ⭐ — `trainer.py`, `cross_validation.py`, `fold_barrier.py`
11. **Nested CV & Metrics** — `nested_cross_validation.py`, `splits.py`, `metrics.py`
12. **Sweeps / Fine-tuning / Explainability** — `hydra_sweep.py`, `finetuner.py`, `explainer.py`
13. **Classical-ML Baselines & Comparison** — `flatten.py`, `sklearn_baselines.py`, `sklearn_nested_cv.py`, `compare_models.py`
14. **Reproducible Runs** — `Dockerfile`, `slurm/*.sh`, `.cluster-helper.yaml`

## 5. File Map (key files by layer)

**Experiment Orchestration & Scripts**
- `scripts/run_experiment.py` — Hydra entry point; wires datasets/features/models/loggers and dispatches to a runner (sklearn / sweep / finetune / flat CV / nested CV).
- `scripts/compare_models.py` — loads `NestedCVResult` artifacts and runs corrected resampled paired t-tests → pairwise p-value matrices + markdown report.
- `scripts/pooled_vs_meanfolds.py` — recomputes pooled-vs-mean-of-folds R² from a saved CV result to expose the aggregation gap.

**Configuration Schemas** (`src/configs/`)
- `trainer_config.py` — optimizer/scheduler, early stopping, label normalization, nested-CV fold/seed layout, checkpointing.
- `model_config.py` — parameterizes every model (backbone, dims, layers, pooling, norm, fusion, MLP/sklearn variants).
- `feature_config.py` — declarative node/edge feature selection (Laplacian PE, cycle counts, GLM features).
- `label_config.py` — prediction target + composite-label columns/method/params.

**Data & Features** (`src/datasets/`)
- `base_dataset.py` — `BrainGraphDataset` ABC, `RawGraphData`, `validate_graph_contract`.
- `feature_builder.py` — declarative `_node_feat_*`/`_edge_feat_*` dispatch from `FeatureConfig`.
- `label_builder.py` — target extraction + stateful fit/transform composite labels.
- `orbit_dataset.py` / `pnc_dataset.py` — concrete loaders (`.mat`/`.npy`/`.csv` + GLM maps).
- `registry.py` / `composite_registry.py` — decorator-based dataset & composite-label registries.

**Model Zoo** (`src/models/`)
- `base_model.py` — encode→decode contract + auxiliary-loss hook.
- `unimodal.py` — assembles backbone + pooling + head.
- `registry.py` — `register_model` / `get_model` factory.
- `backbones/base_backbone.py` + `gcn/gat/gin/transformer.py` — shared base + per-conv subclasses.
- `fusion/` — `concat`, `gated`, `attention` SC/FC fusion strategies.
- `heads/regression_head.py` — configurable MLP regression head.
- `braingnn_model.py` — adapter over vendored layers.
- `vendor/braingnn/` — vendored upstream conv (`MyNNConv`), message passing, inits, losses (see `PROVENANCE.md`).

**Training & Cross-Validation** (`src/training/`)
- `nested_cross_validation.py` — repeated stratified nested CV + inner Optuna HPO (ADR-0008).
- `cross_validation.py` — flat stratified K-fold orchestrator.
- `fold_barrier.py` — per-fold leakage barrier (ADR-0009); `label_normalizer.py` / `glm_normalizer.py` are the fitted pieces.
- `fold_checkpoint.py` — per-fold/per-epoch checkpoint bundles.
- `trainer.py` — pure-PyTorch loop (early stopping, best-epoch, sign-flip aug).
- `splits.py` — deterministic stratified outer/inner split helpers.
- `metrics.py` — MSE/MAE/R²/correlation + per-fold aggregation (pooled vs mean-of-folds).
- `search_space.py` — YAML sweeper spec → Optuna suggestions.
- `sklearn_nested_cv.py` / `statistical_tests.py` — epoch-free baseline CV + corrected t-tests/BH.

**Fine-tuning, Explainability & Sweeps**
- `src/finetuning/finetuner.py` — checkpoint loading, layer freezing, discriminative LR groups (ADR-0007).
- `src/sweeps/hydra_sweep.py` — Hydra+Optuna trial driver.
- `src/gnn_explainer/explainer.py` — post-training GNNExplainer over saved checkpoints (ADR-0006).

**Utilities & Logging**
- `src/logging/wandb_logger.py` + `log_schema.py` — typed WandB facade + canonical metric keys.
- `src/utils/io.py` / `seed.py` — `.mat`/`.npy`/CSV/JSON loaders; RNG seeding.
- `src/interfaces/adapters.py` — boundary adapters (unimodal↔multimodal, partial state-dict load).

**Infrastructure & Ops**
- `Dockerfile` — PyTorch 2.6 + CUDA 12.4 image with CUDA-matched PyG extensions.
- `slurm/train.sh` / `sweep.sh` / `finetune.sh` / `train_sklearn.sh` — SBATCH job scripts (driven by `$RUN_ARGS`).
- `.cluster-helper.yaml` — cluster project manifest (host, container, defaults).

## 6. Complexity Hotspots (approach carefully)

The deepest, highest-risk files — read these slowly and lean on the tests:

| File | Why it's a hotspot |
|---|---|
| `src/training/nested_cross_validation.py` | The headline evaluation engine: outer folds × inner Optuna HPO × refit × persistence (ADR-0008) |
| `src/training/fold_barrier.py` | Leakage-protection coordinator — correctness-critical, touches every fold (ADR-0009) |
| `src/training/trainer.py` | Pure-PyTorch loop: optimizer/scheduler, early stopping, best-epoch, sign-flip aug |
| `src/datasets/feature_builder.py` | Declarative feature dispatch — degree, identity, cycle counts, Laplacian PE, GLM maps |
| `src/datasets/orbit_dataset.py` / `pnc_dataset.py` | Per-subject `.mat`/`.npy`/`.csv` + GLM loading across timepoints |
| `src/models/braingnn_model.py` | ROI-aware conv + hierarchical TopK pooling + auxiliary losses over vendored layers |
| `src/finetuning/finetuner.py` | Checkpoint loading, layer freezing, discriminative LR groups, epoch sweeps |
| `src/gnn_explainer/explainer.py` | Reloads per-fold checkpoints, aggregates ROI×ROI importance across reps/folds |
| `src/training/sklearn_nested_cv.py` + `statistical_tests.py` | Epoch-free nested CV + Nadeau–Bengio corrected t-tests / BH correction |

---

## Where to go next

- **Deeper architecture + glossary:** `CONTEXT.md`
- **Why decisions were made:** `docs/adr/` (13 ADRs)
- **What experiments have been run:** `EXPERIMENTS.md` + `reports/`
- **How to run on the cluster:** `slurm/README.md` + the `/cluster-helper` skill
- **Explore the graph interactively:** `/understand-dashboard` · ask questions with `/understand-chat`
