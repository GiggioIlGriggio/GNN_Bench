# Onboarding Guide — `brain-gnn-codebase`

> A modular platform for benchmarking Graph Neural Networks on brain connectomes.
> Generated from the project knowledge graph (commit `94e0ea3`, analyzed 2026-05-29).

## 1. Project Overview

**What it is:** A modular, scaffolded platform for benchmarking Graph Neural Networks (GCN, GAT, GIN, Graph Transformer, BrainGNN) on **structural (SC)** and **functional (FC)** brain connectomes. It does regression on subject-level targets, with Hydra/Optuna sweeps, nested cross-validation, and WandB logging.

| | |
|---|---|
| **Primary language** | Python |
| **Other languages** | Dockerfile, shell, YAML, TOML, Markdown |
| **Core frameworks** | PyTorch, PyTorch Geometric, Hydra, Optuna, WandB, pydantic |
| **Also uses** | scikit-learn, pytest, Docker (Alembic/SQLAlchemy/aiohttp appear only as transitive deps) |
| **Scale** | 304 graph nodes · 676 edges · 9 architectural layers |

The design philosophy is **typed contracts + registries**: Hydra composes config, Pydantic validates it, and decorator-based registries map string names to dataset/model classes so new components drop in without touching the entry point.

---

## 2. Architecture Layers

The codebase is organized into 9 layers. Roughly, data flows **Config → Data/Features → Models → Training/CV → Experimentation**, with Utilities, Orchestration, Tests, and Docs cutting across.

| Layer | Role | Key files |
|---|---|---|
| **Configuration & Schemas** | Pydantic config dataclasses + the Hydra config tree (dataset/feature/label/model/trainer/logging/explainer/finetuning presets) | `trainer_config.py`, `model_config.py`, `feature_config.py`, `dataset_config.py` |
| **Data & Feature Pipeline** | Dataset abstractions, ORBIT/PNC loaders, feature builder, label builder, dataset registry | `base_dataset.py`, `orbit_dataset.py`, `pnc_dataset.py`, `feature_builder.py`, `label_builder.py` |
| **Model Architectures** | BrainGNN adapter, MLP/unimodal baselines, GCN/GAT/GIN/Transformer backbones, fusion modules, heads, registry | `braingnn_model.py`, `base_model.py`, `base_backbone.py`, `registry.py` |
| **Training & Evaluation** | Single-fold trainer, flat & nested CV engines, fold barrier, normalizers, checkpointing, Optuna search-space, metrics, stats tests | `trainer.py`, `cross_validation.py`, `nested_cross_validation.py`, `fold_barrier.py` |
| **Experimentation & Analysis** | HP sweeps, fine-tuning, post-training GNNExplainer, WandB logging, cohort/diagnostic scripts | `hydra_sweep.py`, `finetuner.py`, `explainer.py`, `wandb_logger.py` |
| **Shared Utilities** | I/O, seeding, boundary adapters between modules | `io.py`, `adapters.py`, `model_printer.py` |
| **Orchestration & Infrastructure** | Run entry point, Slurm submission scripts, Docker definition | `run_experiment.py`, `slurm/run_experiments.sh`, `Dockerfile` |
| **Test Suite** | Pytest suites + manual demo harnesses across datasets/features/models/training/barriers/checkpoints/explainer | `test_training.py`, `test_datasets.py`, `test_fold_barrier.py` |
| **Documentation & Decisions** | ADRs, the `CONTEXT.md` domain model, README, agent-config docs, superpowers specs/plans | `CONTEXT.md`, `docs/adr/*` |

---

## 3. Key Concepts

These are the design decisions that recur throughout the code (all backed by ADRs in `docs/adr/`):

- **Hydra + Pydantic split (ADR 0001).** Hydra composes config groups and handles CLI overrides; Pydantic enforces type-safe validation per group. The config classes (`ModelConfig`, `FeatureConfig`, `DatasetConfig`, `TrainerConfig`) are the **highest fan-in files in the whole codebase** — almost everything depends on them.
- **The graph contract (ADR 0002).** `validate_graph_contract` enforces fixed shapes/dtypes for `x`, `edge_index`, `edge_attr`, `y`, so downstream models never need defensive shape checks.
- **Registry pattern (ADR 0005).** `@register_dataset` / `@register_model` decorators map string names to classes. Add a new model or dataset by name without editing the entry point.
- **No data leakage / the Fold Barrier (ADR 0004, 0009).** The `FoldBarrier` unifies label normalization, GLM feature normalization, and composite-label construction, **fitting all of them on training subjects only**, persisted as a typed `barrier.pt` state-dict.
- **Stratified CV on continuous labels (ADR 0003).** Quantile-binned stratification makes k-fold work for regression targets.
- **Nested CV + corrected t-tests (ADR 0008).** Outer repeated stratified k-fold for unbiased estimation wraps an inner Optuna search; corrected resampled t-tests compare models fairly.
- **Vendored BrainGNN (ADR 0010).** BrainGNN's ROI-aware conv and TopK pooling layers are vendored verbatim from upstream (Li et al. 2021); `braingnn_model.py` is a thin encode/decode adapter over them. *(This supersedes an earlier hand-rewrite that drifted.)*
- **Positional & identity features (ADR 0011).** Laplacian PE (normalized-Laplacian eigenvectors) + ID-GNN-Fast cycle counts, with a train-time eigenvector sign-flip augmentation to handle LapPE sign ambiguity.
- **Post-hoc capabilities.** Fine-tuning auto-adopts the pretrained feature config (ADR 0007); GNNExplainer runs after training on saved checkpoints (ADR 0006).

---

## 4. Guided Tour

A 14-step learning path, ordered to expand the pipeline from entry point outward:

1. **Project Overview** — `README.md` + `CONTEXT.md` (the canonical domain vocabulary: ROI, SC/FC, nested CV, fold barrier, auxiliary losses).
2. **The Experiment Entry Point** — `scripts/run_experiment.py`: the Hydra entry point that composes all sub-configs and dispatches to flat/nested CV, fine-tuning, and explanation. *This file is the map of the whole pipeline.*
3. **Configuration & Schemas** — `model_config.py`, `feature_config.py`, `dataset_config.py` + ADR 0001.
4. **Datasets & the Graph Contract** — `base_dataset.py` (`RawGraphData`, abstract `BrainGraphDataset`), `orbit_dataset.py`, `pnc_dataset.py` + ADR 0002.
5. **Feature & Label Builders** — `feature_builder.py` (fan-in 15), `label_builder.py`, `composite_registry.py`.
6. **Positional & Identity Node Features** — `laplacian_pe.yaml`, `cycle_counts.yaml` + ADR 0011.
7. **Model Architectures & Backbones** — `base_model.py` (encode/decode + auxiliary_loss), `base_backbone.py`, `gat.py`, model `registry.py`, `regression_head.py`.
8. **The Registry Pattern & BrainGNN** — ADR 0005, `braingnn_model.py`, ADR 0010.
9. **The Single-Fold Trainer** — `trainer.py`, `trainer_config.py`, `metrics.py`.
10. **Leak-Free CV & the Fold Barrier** — `cross_validation.py`, `fold_barrier.py` + ADRs 0004, 0009.
11. **Nested Cross-Validation** — `nested_cross_validation.py` (fan-out 10), `search_space.py` + ADR 0008.
12. **Hyperparameter Sweeps & Logging** — `hydra_sweep.py`, `bayesian.yaml`, `wandb_logger.py`.
13. **Fine-Tuning & Explainability** — `finetuner.py` + ADR 0007, `explainer.py` + ADR 0006.
14. **Containerization & Cluster Deployment** — `Dockerfile`, `slurm/README.md`, `requirements.txt`.

---

## 5. File Map

Organized by layer. Descriptions from node summaries. (⚠️ = `complex`-rated hotspot.)

### Orchestration (start here)
- **`scripts/run_experiment.py`** ⚠️ — Hydra entry point; composes all configs, runs flat/nested CV, optional fine-tuning and GNNExplainer; threads the Laplacian PE sign-flip range into `TrainerConfig`.
- **`slurm/run_experiments.sh`** — cluster launcher; Hydra overrides reach job scripts via `$RUN_ARGS`.
- **`Dockerfile`** — builds on official PyTorch 2.6.0 / CUDA 12.4 image; installs PyG compiled extensions matched to that exact torch+CUDA (the fiddliest part to reproduce).

### Configuration & Schemas
- **`src/configs/trainer_config.py`** ⚠️ — optimizer, scheduler, early-stopping, label normalization, nested-CV, HPO, checkpointing settings + validation.
- **`src/configs/model_config.py`** — parameterizes every GNN (highest fan-in).
- **`src/configs/feature_config.py`** — node/edge feature selection.
- **`src/configs/label_config.py`**, **`explainer_config.py`** — target resolution and explainer setup.

### Data & Feature Pipeline (all complex)
- **`src/datasets/base_dataset.py`** ⚠️ — `RawGraphData` container, `validate_graph_contract`, abstract `BrainGraphDataset`.
- **`src/datasets/feature_builder.py`** ⚠️ — central `FeatureBuilder`: degree/strength/centrality, identity, connectivity profiles, GLM maps, cycle-counts, Laplacian positional encodings.
- **`src/datasets/label_builder.py`** ⚠️ — resolves labels from metadata; composite targets with fit/transform persistence for fold-safe reuse.
- **`src/datasets/orbit_dataset.py`**, **`pnc_dataset.py`** ⚠️ — concrete cohort loaders (.mat structural, .npy functional) with edge thresholding.
- **`src/datasets/composite_registry.py`** ⚠️ — composite-label registry (e.g. inverse efficiency scores).
- **`src/datasets/registry.py`** — string-name dataset registry.

### Model Architectures
- **`src/models/braingnn_model.py`** ⚠️ — thin adapter over vendored ROI-aware conv + TopK pooling; encode/decode with topk/unit/consistency auxiliary losses.
- **`src/models/base_model.py`** — abstract `BrainGNN`: fixes encode/decode contract + `auxiliary_loss` hook.
- **`src/models/backbones/base_backbone.py`** — message-passing loop (conv→norm→relu→dropout) + jumping-knowledge; GCN/GAT/GIN/Transformer subclass it.
- **`src/models/unimodal.py`**, **`fusion/attention_fusion.py`**, **`fusion/gated_fusion.py`**, **`mlp_model.py`** — baselines and modality fusion.

### Training & Evaluation (mostly complex)
- **`src/training/trainer.py`** ⚠️ — single-fold loop: optimizer/scheduler, epoch training with optional LapPE sign-flip, early stopping, predict + eval.
- **`src/training/cross_validation.py`** ⚠️ — flat k-fold: stratified splits, per-fold leak-free preprocessing via `FoldBarrier`, checkpointing, metric aggregation.
- **`src/training/nested_cross_validation.py`** ⚠️ — outer repeated stratified k-fold + inner Optuna HPO, refit on train+val, outer-test eval.
- **`src/training/fold_barrier.py`** ⚠️ — all per-fold transforms fitted on training data only; save/load support.
- **`src/training/search_space.py`** ⚠️ — parses Optuna choice/range/interval via a restricted AST interpreter; applies overrides without mutation.
- **`src/training/statistical_tests.py`** ⚠️, **`fold_checkpoint.py`** ⚠️ — corrected resampled t-tests; per-fold checkpoint management.
- **`src/training/metrics.py`**, **`glm_normalizer.py`**, **`label_normalizer.py`** — MAE/RMSE/R²/correlation + CIs; the two normalizers.

### Experimentation & Analysis
- **`src/finetuning/finetuner.py`** ⚠️ — transfer learning: loads per-fold checkpoints, head re-init, layer freezing, per-group LRs.
- **`src/gnn_explainer/explainer.py`** ⚠️ — reloads each fold's best checkpoint, explains held-out graphs, emits edge-importance matrices (nested + legacy layouts).
- **`src/logging/wandb_logger.py`** ⚠️ — WandB facade: fold metrics, dataset stats, sweep trials, nested-CV summaries, scatter plots.
- **`src/sweeps/hydra_sweep.py`**, **`scripts/compare_models.py`**, **`scripts/cohort_selection/generate_cohort.py`** ⚠️ — sweep driver, model comparison, cohort generation.

### Shared Utilities
- **`src/utils/io.py`**, **`src/interfaces/adapters.py`** — I/O + seeding; boundary adapters between modules.
- **`src/utils/model_printer.py`** ⚠️ — model summary printing.

---

## 6. Complexity Hotspots

Approach these carefully — they are the `complex`-rated files with the most logic and the highest blast radius. A new developer should read the paired ADR/test before modifying them.

| File | Why it's tricky | Read alongside |
|---|---|---|
| `scripts/run_experiment.py` | The orchestrator — touches every layer | the full tour |
| `src/training/fold_barrier.py` | Leakage prevention is a correctness invariant; train-only fitting is subtle | ADR 0004, 0009; `test_fold_barrier.py` |
| `src/training/nested_cross_validation.py` | Outer/inner loop + Optuna + refit; easy to get the evaluation protocol wrong | ADR 0008 |
| `src/training/cross_validation.py` | Stratification + per-fold preprocessing + checkpointing | ADR 0003; `test_training.py` |
| `src/training/search_space.py` | Restricted AST interpreter for HPO expressions | `test_training.py` |
| `src/datasets/feature_builder.py` | Many feature families incl. LapPE/cycle counts; column bookkeeping matters | ADR 0011 |
| `src/models/braingnn_model.py` | Thin adapter over **vendored** upstream layers — don't "fix" the vendored code | ADR 0010; `test_vendor_braingnn.py` |
| `src/datasets/base_dataset.py` | Graph contract validation underpins all models | ADR 0002; `test_datasets.py` |
| `src/finetuning/finetuner.py` | Checkpoint loading + freezing + per-group LRs + feature-config adoption | ADR 0007 |
| `src/gnn_explainer/explainer.py` | Dual checkpoint layouts; reloads per-fold best models | ADR 0006; `test_gnn_explainer.py` |

> **Note on the test suite:** nearly every test file is rated `complex`, which is healthy — the leakage barrier, checkpoints, vendored BrainGNN, and CV engines all have dedicated coverage (`tested_by` edges: 32). Run the relevant test before and after touching a hotspot.
