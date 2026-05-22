# CONTEXT.md — GNN Brain Connectivity Benchmarking Platform

This file is the canonical source of domain language and architecture for this codebase.
Engineering skills read it before exploring code. `/grill-with-docs` updates it lazily as terms get resolved.

---

## Domain overview

This platform benchmarks Graph Neural Networks for predicting **behavioural or cognitive scores** from **brain connectivity data**. Each research subject becomes one or more graphs; the GNN is trained to regress a scalar target (age, memory score, accuracy, etc.) from the graph structure and node features. The canonical evaluation is **repeated stratified nested K-fold cross-validation** with per-outer-fold Optuna HPO ([ADR-0008](docs/adr/0008-nested-cross-validation.md)); cross-model comparisons go through the Bouckaert–Frank corrected t-test + BH-FDR.

The two supported datasets are:
- **ORBIT** — longitudinal adult cohort with structural and functional MRI
- **PNC** — Philadelphia Neurodevelopmental Cohort (adolescents), functional MRI + structural MRI

---

## Glossary

Terms are defined as used in this codebase. Synonyms in parentheses are accepted; **avoid** lists the terms that would confuse a reader.

### Brain / neuroscience

| Term | Definition | Avoid |
|---|---|---|
| **ROI** (Region of Interest) | A single brain parcel / node in the graph. The number of ROIs is fixed per atlas (e.g. 400 for Schaefer-400). | "voxel", "region" (ambiguous) |
| **Atlas** | A brain parcellation scheme that assigns every voxel to an ROI. Configured via `DatasetConfig.atlas` (e.g. `schaefer400`). | "parcellation scheme" |
| **SC** (Structural Connectivity) | Edge weights derived from white-matter tractography (diffusion MRI). Stored in `.mat` files under `Structural_Mats/`. | "DTI", "white-matter connectivity" |
| **FC** (Functional Connectivity) | Edge weights derived from fMRI BOLD signal correlations. Stored as `.npy` files under `Functional_Mats/`. | "BOLD connectivity" |
| **Connectome** | The full N×N dense connectivity matrix (SC or FC) for a subject before thresholding. Not stored in this codebase — converted immediately to edge-list format. | "adjacency matrix", "connectivity matrix" |
| **Modality** | Whether a graph is built from SC, FC, or both (`"sc"`, `"fc"`, `"multimodal"`). Controlled by `DatasetConfig.modality`. | — |
| **GLM Map** | A General Linear Model activation map: one scalar value per ROI, summarising how strongly that ROI responds to a task contrast. Used as extra node features. | "activation map", "contrast map" (acceptable but be explicit about which) |
| **Contrast** | A specific GLM comparison (e.g. `"contrast-2back_vs_0back"`). One `.npy` file per subject per contrast. | — |
| **Timepoint** | A longitudinal data-collection session (e.g. `T0`, `T1`). Multiple timepoints produce multiple graphs per subject. | "visit", "session" |
| **Subject** | A research participant. Identified by a string like `"sub-001"` (ORBIT) or `"sub-0001103037"` (PNC). | "patient", "case", "sample" |

### Graph representation

| Term | Definition | Avoid |
|---|---|---|
| **Graph** | A PyTorch Geometric `Data` object representing one subject (+ timepoint). Satisfies the **graph contract**: `x [N, node_feat_dim]`, `edge_index [2, E]`, `edge_attr [E, edge_feat_dim]`, `y [1]`. | "sample", "instance" |
| **Graph contract** | The strict tensor dtype and shape requirements enforced by `validate_graph_contract()` in `src/datasets/base_dataset.py`. Every `build_graph` implementation must satisfy it before returning. | — |
| **Edge thresholding** | Reduction of the dense connectome to a sparse edge list. Three modes: `absolute` (keep edges above a value), `topk_percent` (keep the top-k% strongest edges), `none` (keep all non-zero edges). | "pruning", "sparsification" |
| **Node feature** | A scalar or vector attached to each ROI node. Built by `FeatureBuilder` from `node_features` config. Examples: degree, strength, betweenness centrality, GLM maps. | — |
| **Edge feature / weight** | A scalar attached to each edge (connectivity value). Built by `FeatureBuilder.build_edge_features`. | "edge attribute" (same thing, both fine) |
| **Subject-timepoint ID** | A subject ID prefixed with the timepoint when multiple timepoints are loaded, e.g. `"T0_sub-001"`. | — |

### Machine learning

| Term | Definition | Avoid |
|---|---|---|
| **Outer fold** | One stratified split in the *outer* loop of nested CV ([ADR-0008](docs/adr/0008-nested-cross-validation.md)). Holds out a test set; the remainder is the **outer TrainVal pool**. There are `n_repetitions × n_outer_folds` outer folds per run (default 10 × 5 = 50). | "fold" alone is ambiguous — say outer or inner |
| **Repetition** | One full pass through `n_outer_folds` outer folds with its own stratification seed. The paper protocol uses 10 repetitions; the legacy fast mode uses 1. | "rep" (accepted in logs/keys), "run" (too broad) |
| **Inner split** | A single stratified 4-way split of an outer fold's TrainVal pool into 3 train + 1 val parts (3:1 ratio). One inner split per outer fold — not a full inner CV. Used for Optuna HP selection and Trainer early stopping. | "inner fold" (we have one split, not folds), "validation split" (ambiguous with outer val) |
| **Outer TrainVal pool** | The data left over after holding out the outer Test fold. Source for both the inner split and the final refit. | "training data" (ambiguous with inner Train) |
| **Refit-on-TrainVal** | After Optuna picks winning HPs for an outer fold ([ADR-0008](docs/adr/0008-nested-cross-validation.md)), retrain one fresh model on the full outer TrainVal pool with those HPs for a fixed epoch count equal to the winning trial's `best_epoch`. The refit checkpoint is the only one evaluated on outer Test. | "retraining" (too broad) |
| **Inner-HPO trial** | One Optuna trial inside an outer fold: sample HPs, train on inner Train, score on inner Val using the HPO metric. `inner_hpo_trials = 0` skips Optuna entirely and falls back to fixed model-config HPs. | "trial" alone (ambiguous with `_trial_current` checkpoint dir) |
| **HPO metric** | The scalar metric optimised by inner-loop Optuna and used as the Trainer's early-stopping signal. Default `val_mae` (minimise); `val_r2` (maximise) is supported. | "objective" (overloaded — Hydra also uses it) |
| **Fast mode** | Preset of `NestedCrossValidator` knobs: `n_repetitions=1, n_outer_folds=5, inner_hpo_trials=0`. Reproduces the legacy single-CV behaviour. | "single CV" (was the old name) |
| **Complete mode** | Preset of `NestedCrossValidator` knobs: `n_repetitions=10, n_outer_folds=5, inner_hpo_trials=20`. Matches the paper protocol; this is the canonical benchmark configuration. | "nested mode" (more descriptive but the user-facing knob is named) |
| **Bouckaert–Frank test** | Corrected resampled paired t-test for comparing two models' per-outer-fold scores under repeated CV ([ADR-0008](docs/adr/0008-nested-cross-validation.md)). Corrects the naive paired t-test's degrees of freedom for the dependence between scores from overlapping training sets. Implemented in `src/training/statistical_tests.py`; exposed via `scripts/compare_models.py`. | "paired t-test" (the naive one is biased here) |
| **BH-adjusted p-value** | Benjamini–Hochberg false-discovery-rate adjustment applied across all pairwise model comparisons within one comparison run. Computed by `statistical_tests.benjamini_hochberg`. Synonyms: "BH-FDR correction". | "p-value correction" (too vague) |
| **Search-space DSL** | The string mini-language used in `configs/sweeper/<model>.yaml` to declare HP search ranges (`choice(...)`, `range(low, high[, step])`, `interval(low, high)`, `tag(log, ...)`). Originally Hydra-Optuna's multirun DSL; parsed by `src/training/search_space.py` so the same YAML drives both legacy `HydraSweep` and the nested-CV inner HPO. | "sweeper syntax" (too vague) |
| **Label normalisation** | Per-fold standardisation of the scalar regression target. Fit only on training subjects; applied to val and test. Prevents data leakage. | "target scaling", "output normalisation" |
| **Composite label** | A regression target derived from multiple raw score columns, either by weighted combination or PCA. Built per-fold by `LabelBuilder` to avoid leakage from PCA fitting. | "multi-component label" |
| **GLM normalisation** | Per-node z-scoring of GLM feature columns across training subjects. Applied per fold to prevent leakage. Controlled by `FeatureConfig.glm_normalize`. | — |
| **Fold barrier** | The per-fold leakage-protection coordinator. Owns the composite-label `LabelBuilder` (when configured), the `LabelNormalizer`, and the `GLMFeatureNormalizer` as a single fitted bundle for one outer fold's train pool. Exposes `fit(train_graphs, train_labels_or_components)`, `transform_graphs(graphs) → graphs` (returns new graphs; never mutates inputs), `transform_labels(...)`, and `inverse_transform_labels(...)`. Persisted as a typed state-dict (`barrier.pt`) inside the fold's [Checkpoint](#checkpoint-layout). Unifies the three leakage protections from [ADR-0004](docs/adr/0004-no-data-leakage.md) into one module instead of three transformers fit and threaded by hand. | "leakage barrier", "preprocessor", reaching into the bundled `LabelNormalizer` directly |
| **Backbone** | The GNN message-passing layers (GCN, GAT, GIN, Graph Transformer). Produces node-level embeddings. | "encoder base", "GNN layers" |
| **Fusion** | A module that combines SC and FC embeddings in multimodal mode (concat, cross-attention, gated). | "aggregation" (ambiguous) |
| **Head** | The MLP that maps the pooled graph embedding to a scalar prediction. Currently only `RegressionHead`. | "decoder", "output layer" |
| **Pooling** | Global graph pooling (mean, max, add, attention) that converts node embeddings to a graph-level embedding. | "readout" (accepted in PyG literature) |
| **ROI-TopK Pooling** | Hierarchical differentiable node selection used in BrainGNN. A learned scoring function ranks each ROI; the top-k fraction are kept and the graph is coarsened. Implemented via `torch_geometric.nn.TopKPooling`. Distinct from global pooling — it reduces the graph mid-network, not at the end. | "TopK pool", "node pooling" (ambiguous — say "ROI-TopK pooling") |
| **Auxiliary loss** | An extra loss term beyond MSE, emitted by models that implement `auxiliary_loss() → dict[str, Tensor]`. The trainer adds these to the main MSE loss and logs them separately to wandb. Currently only BrainGNN emits aux losses. | "regularisation loss" (too broad) |
| **Unit loss** | BrainGNN auxiliary loss that penalises pooling scores far from 0 or 1, encouraging binary (hard) node selection. Weight controlled by `model_params.unit_loss_weight`. | — |
| **TopK loss** | BrainGNN auxiliary loss that encourages consistent top-K ROI selection across training samples. Weight controlled by `model_params.topk_loss_weight`. | — |
| **model_params** | A free-form `dict` field in `ModelConfig` for model-specific hyperparameters (e.g. `pool_ratio`, `unit_loss_weight`). Read by the model constructor; ignored by all other models. Avoids adding model-specific fields to the shared `ModelConfig` schema. | "extra_params", "kwargs" |
| **Checkpoint** | The complete persistent record of one outer fold under `checkpoints/rep_<R>/fold_<K>/` — the refit-on-TrainVal model `state_dict` plus the fitted per-fold leakage state inside `barrier.pt` (see [Fold barrier](#glossary)) plus per-fold metadata (`best_hparams.json`, `trials.csv`, `test_predictions.npz`, `model_config.json`, `feature_config.json`, `metrics.json`). See [Checkpoint layout](#checkpoint-layout) for the full file list. Legacy `HydraSweep` runs still write the flat `checkpoints/fold_<K>/` layout. | "model weights", "snapshot" (use "epoch snapshot" for epoch-level saves) |
| **Epoch snapshot** | A model `state_dict` saved mid-training (either every N epochs or on best-val). Stored in `epoch_checkpoints/` next to the fold checkpoint. | — |
| **Allowlist** | An optional text file listing subject IDs to include. When set, subjects not in the list are silently skipped; IDs in the list that are missing on disk raise an error. | "whitelist" |

---

## Architecture map

```
scripts/run_experiment.py          # Hydra entry point — orchestrates everything (standard branch routes to NestedCrossValidator)
scripts/compare_models.py          # Cross-run statistical comparison: consumes ≥2 NestedCVResult JSONs, runs pairwise Bouckaert–Frank + BH-FDR, emits CSV + markdown
│
├── configs/                        # YAML config files (one group per module)
│   ├── sweeper/<model>.yaml        # Single source of truth for HP search spaces (Hydra-Optuna DSL); consumed by both legacy HydraSweep and the nested-CV inner HPO via search_space.py
│   └── (validated by pydantic schemas in src/configs/)
│
├── src/datasets/                   # Data loading
│   ├── base_dataset.py             # BrainGraphDataset (abstract) + RawGraphData + validate_graph_contract
│   ├── orbit_dataset.py            # ORBIT loader (@register_dataset("orbit"))
│   ├── pnc_dataset.py              # PNC loader (@register_dataset("pnc"))
│   ├── feature_builder.py          # FeatureBuilder — assembles node/edge features from config
│   ├── label_builder.py            # LabelBuilder — scalar or composite labels
│   ├── composite_registry.py       # Composite label method registry
│   └── registry.py                 # Dataset registry (@register_dataset decorator)
│
├── src/models/                     # GNN model zoo
│   ├── base_model.py               # BrainGNN (abstract): encode → decode; optional auxiliary_loss() hook
│   ├── unimodal.py                 # Unimodal GNN (backbone + pooling + head)
│   ├── mlp_model.py                # MLP baseline (no graph structure)
│   ├── braingnn_model.py           # BrainGNN (Li et al. 2021): ROI-aware GConv + ROI-TopK pooling + aux losses
│   ├── backbones/                  # GCNBackbone, GATBackbone, GINBackbone, GraphTransformerBackbone
│   ├── fusion/                     # ConcatFusion, CrossAttentionFusion, GatedFusion
│   ├── heads/                      # RegressionHead
│   └── registry.py                 # Model registry (@register_model decorator)
│
├── src/training/                   # Training and evaluation
│   ├── trainer.py                  # Trainer: fit() + evaluate() + predict()
│   ├── nested_cross_validation.py  # NestedCrossValidator (ADR-0008): canonical entry; outer repeated stratified K-fold × inner stratified 4-way split + in-process Optuna HPO + refit-on-TrainVal; produces NestedCVResult
│   ├── search_space.py             # Search-space DSL parser (ADR-0008): reads configs/sweeper/<model>.yaml and emits SearchSpec objects that drive trial.suggest_*
│   ├── statistical_tests.py        # Pure functions (ADR-0008): Bouckaert–Frank corrected resampled paired t-test + Benjamini–Hochberg FDR adjustment
│   ├── cross_validation.py         # CrossValidator: legacy stratified K-fold (still used by HydraSweep); superseded for standard runs by NestedCrossValidator
│   ├── label_normalizer.py         # LabelNormalizer: standard/robust/minmax/none, serialisable
│   ├── glm_normalizer.py           # GLMFeatureNormalizer: per-node z-score of GLM columns
│   ├── metrics.py                  # compute_metrics (MAE, RMSE, R², Pearson r), aggregate_fold_metrics
│   └── checkpoint_manager.py       # CheckpointManager: save/load fold and epoch checkpoints
│
├── src/sweeps/                     # Legacy hyperparameter sweeps (kept until retention decision; see ADR-0008 + open issue)
│   ├── base_sweep.py               # SweepRunner (abstract)
│   └── hydra_sweep.py              # HydraSweep: one Optuna trial per Hydra --multirun call (HP-selection leakage — not the canonical benchmark)
│
├── src/finetuning/                 # Transfer learning
│   └── finetuner.py                # Finetuner: load pretrained, freeze layers, run CV; epoch-sweep mode
│
├── src/gnn_explainer/              # Post-training interpretability
│   └── explainer.py                # GNNExplainerRunner: re-loads best checkpoints, outputs importance matrices
│
├── src/logging/                    # Experiment logging
│   ├── wandb_logger.py             # WandbLogger: typed methods per event
│   └── log_schema.py               # Final[str] constants for all wandb keys
│
├── src/interfaces/                 # Adapter utilities
│   └── adapters.py                 # unimodal_to_multimodal, adapt_input_features, load_partial_state_dict
│
└── src/utils/                      # Shared helpers
    ├── seed.py                     # seed_everything
    ├── io.py                       # load_mat, load_npy, load_csv, save/load JSON
    └── model_printer.py            # Rich console tables for config + model summary
```

### Data flow (standard run — nested CV, ADR-0008)

The standard branch of `scripts/run_experiment.py` routes to `NestedCrossValidator`. The protocol has two loops; per-fold leakage protections (LabelBuilder / LabelNormalizer / GLMFeatureNormalizer) are fit on the outer Train pool only — never on the outer Test fold.

```
Dataset.load_raw()
  → Dataset.get_dataset()             # list of PyG Data objects (y=placeholder for composite)
  → Dataset.get_labels()              # np.ndarray [N] for stratification
  → NestedCrossValidator.run()
    │
    │  outer loop: n_repetitions × n_outer_folds  (default 10 × 5 = 50)
    ├── StratifiedKFold.split()       # outer (train_val_idx, test_idx); seed = outer_seeds[rep]
    │     │
    │     │  inner loop: one stratified 4-way split (3:1) of outer TrainVal
    │     ├── StratifiedKFold(n=4).split()  # inner (train_idx, val_idx); seed = outer_seeds[rep]*1000 + fold
    │     │     │
    │     │     │  inner HPO: inner_hpo_trials Optuna TPE trials
    │     │     ├── search_space.parse(configs/sweeper/<model>.yaml) → [SearchSpec, ...]
    │     │     ├── for trial in study.optimize:
    │     │     │     LabelBuilder.fit_transform()  # outer-train only
    │     │     │     LabelNormalizer.fit()         # outer-train only
    │     │     │     GLMFeatureNormalizer.fit()    # outer-train only
    │     │     │     Trainer.fit(train=inner-train, val=inner-val)  # early-stop on hpo_metric
    │     │     │     → trial reports (best_epoch, val_hpo_metric)
    │     │     └── best_hparams, best_trial, best_epoch ← study.best_trial
    │     │
    │     │  refit-on-TrainVal: one fresh model on full outer TrainVal
    │     ├── apply(best_hparams)
    │     ├── Trainer.fit(train=outer-TrainVal, val=None, max_epochs=best_epoch+1)
    │     ├── Trainer.predict(test=outer-Test)      # the only outer-Test evaluation
    │     ├── CheckpointManager.save_fold_checkpoint(rep_<R>/fold_<K>/)
    │     └── write rep_<R>/fold_<K>/{best_hparams.json, trials.csv, test_predictions.npz}
    │
    └── NestedCVResult.save(nested_cv_result.json)  # aggregate mean ± std over 50 folds
```

When `inner_hpo_trials == 0` the validator skips Optuna entirely and reuses the fixed `model_params` HPs; the inner split still drives Trainer early stopping, and the same refit-on-TrainVal step runs with `best_epoch` set to the early-stopping epoch. This is the fast / legacy preset.

Cross-run comparison is a separate step:

```
≥2 nested_cv_result.json (one per model run)
  → scripts/compare_models.py
    → for each pair: statistical_tests.corrected_resampled_paired_t_test(scores_a, scores_b, n_outer_folds)
    → statistical_tests.benjamini_hochberg(p_values)          # FDR adjustment over all pairs
    → write {p_matrix.csv, p_matrix_bh.csv, summary.md}
```

### Legacy data flow (`HydraSweep` --multirun)

Kept until the retention decision in ADR-0008 / open follow-up issue closes. Each `--multirun` process runs one Optuna trial that itself does a full K-fold CV via the legacy `CrossValidator`; HP selection sees the same folds used for evaluation (HP-selection leakage). Output layout is the legacy `checkpoints/fold_<K>/`, not the nested `rep_<R>/fold_<K>/`.

### Execution modes

The entry point detects which mode to run based on CLI flags and config:

| Mode | Trigger | What happens |
|---|---|---|
| **Standard (nested CV)** | No `--multirun`, `finetuning.enabled=false` | `NestedCrossValidator` runs the two-loop protocol of ADR-0008; writes `checkpoints/rep_<R>/fold_<K>/` + top-level `nested_cv_result.json`. Knobs `n_repetitions`, `n_outer_folds`, `inner_hpo_trials` select fast vs. complete preset. |
| **Legacy sweep** | `--multirun` with a sweeper config | One Optuna trial per Hydra process; each trial runs a full legacy `CrossValidator` K-fold. Retained for fast iteration; not the canonical benchmark (HP-selection leakage). Best trial runs GNNExplainer. |
| **Finetuning** | `finetuning.enabled=true` | Loads pretrained checkpoint, freezes layers, runs legacy CV |
| **Epoch-sweep finetuning** | `finetuning.epoch_checkpoint_dir` set | Finetunes from every epoch snapshot, logs R² vs pretrain epoch |

---

## Key invariants

- **No data leakage**: label normalisation, GLM normalisation, and composite-label PCA are always fit on the outer Train pool only, then applied to inner val, refit, and outer test. Both `NestedCrossValidator` and the legacy `CrossValidator` honour this.
- **Graph contract enforced**: every `build_graph` implementation calls `validate_graph_contract()` before returning. Downstream code may assume the contract holds.
- **Per-fold fresh model**: each outer fold calls `model_factory()` to get a new untrained model — both for every Optuna trial inside an outer fold and for the refit-on-TrainVal step.
- **Refit checkpoint is the only one evaluated on outer Test**: under nested CV, no trial model is scored on outer Test. The trial's role is HP selection only; the refit-on-TrainVal model is the one persisted at `rep_<R>/fold_<K>/model_best.pt` and the only model whose predictions populate `test_predictions.npz`.
- **Wandb keys are constants**: no bare strings — all wandb metric keys are `Final[str]` values in `log_schema.py`.

---

## Checkpoint layout

The standard nested-CV branch ([ADR-0008](docs/adr/0008-nested-cross-validation.md)) writes one self-contained directory per outer fold, plus a single aggregate JSON at the run root:

```
<checkpoint_dir>/
├── nested_cv_result.json           # NestedCVResult: mean ± std over all 50 folds + per-fold scores
└── rep_<R>/                        # R = 0 .. n_repetitions-1
    └── fold_<K>/                   # K = 0 .. n_outer_folds-1
        ├── model_best.pt           # refit-on-TrainVal state_dict — the only model scored on outer Test
        ├── model_last.pt           # final-epoch state_dict (== best in refit because no inner val)
        ├── barrier.pt              # outer-TrainVal-fit FoldBarrier state-dict (torch.save, ADR-0009)
        ├── metrics.json            # outer-Test metrics for this fold (MAE, RMSE, R², Pearson r)
        ├── model_config.json       # ModelConfig.model_dump() at the winning HPs
        ├── feature_config.json     # FeatureConfig used to build this fold's graphs
        ├── best_hparams.json       # {"best_hparams": {...}, "best_trial": int, "refit_epochs": int}
        ├── trials.csv              # one row per inner-HPO trial: HPs + inner-val hpo_metric + best_epoch
        └── test_predictions.npz    # {"y_true": [...], "y_pred": [...]} on outer Test
```

When `inner_hpo_trials == 0` (fast preset) the inner Optuna loop is skipped; `best_hparams.json` records the fixed model-config HPs and `trials.csv` is absent.

### `nested_cv_result.json` schema

Produced by `NestedCVResult.save()` (see `src/training/nested_cross_validation.py`). Shape:

```json
{
  "run_name": "string",
  "model_name": "string",
  "n_repetitions": 10,
  "n_outer_folds": 5,
  "inner_hpo_trials": 20,
  "hpo_metric": "val_mae",
  "outer_seeds": [42, 43, ...],
  "fold_results": [
    {
      "rep": 0,
      "fold": 0,
      "outer_test_metrics": {"mae": ..., "rmse": ..., "r2": ..., "pearson_r": ...},
      "best_hparams": {"...": ...},
      "best_trial": 7,
      "refit_epochs": 42,
      "n_train": 120, "n_val": 40, "n_test": 40,
      "y_true": [...], "y_pred": [...]
    }
    /* … one entry per (rep, fold), 50 total under the complete preset */
  ],
  "mean_metrics": {"mae": ..., "rmse": ..., "r2": ..., "pearson_r": ...},
  "std_metrics": {"mae": ..., "rmse": ..., "r2": ..., "pearson_r": ...}
}
```

`scripts/compare_models.py` consumes this file directly — no model code needed at comparison time.

### Legacy `HydraSweep` layout

Until the retention decision in [ADR-0008](docs/adr/0008-nested-cross-validation.md) resolves, `--multirun` runs continue to write the flat layout:

```
<checkpoint_dir>/
└── fold_<K>/
    ├── model_best.pt
    ├── model_last.pt
    ├── barrier.pt
    ├── metrics.json
    ├── model_config.json
    └── feature_config.json
```

This layout has HP-selection leakage (each `--multirun` trial sees the same K folds it is scored on) and is not the canonical benchmark.

---

## Dataset directory layouts

### ORBIT

```
<root>/
├── T0/
│   ├── Structural_Mats/    # sub-XXX.mat  (scipy .mat, keys: {atlas}_<sc_type>, {atlas}_region_labels)
│   ├── Functional_Mats/    # sub-XXX/<fc_task>/fc_matrix.npy
│   ├── Tabular_Data/       # ALL tabular data.csv  (ID column: integer, e.g. 1)
│   └── GLM_Maps/
│       └── {atlas}_glm/<aggregation>/<map_type>/<contrast>/sub-XXX.npy
└── T1/
    └── (same structure)
```

Subject ID mapping: integer `ID` in tabular CSV → zero-padded `sub-001` in filenames.

### PNC

```
<root>/
└── T0/
    ├── Structural_maps/    # sub-XXXXXXXXXX_run-1_..._connectome.mat
    ├── Functional_Mats/    # sub-XXXXXXXXXX_connectivity_matrix.csv
    ├── Tabular_data/       # PNC_ALL_SCORES.csv  (SUBJID column: 12-digit, e.g. 600001103037)
    └── GLM_Maps/           # same layout as ORBIT
```

Subject ID mapping: `SUBJID` (12-digit with `6000` prefix) → `sub-XXXXXXXXXX` (last 10 digits with `sub-` prefix).

---

## Extension points

| What to add | Where | Steps |
|---|---|---|
| New dataset | `src/datasets/` | Subclass `BrainGraphDataset`, decorate with `@register_dataset("name")`, add `configs/dataset/name.yaml` |
| New backbone | `src/models/backbones/` | Subclass `GNNBackbone`, register with `@register_model("name")`, add `configs/model/name.yaml` |
| New standalone model | `src/models/` | Subclass `BrainGNN` directly (like `braingnn_model.py`), decorate with `@register_model("name")`, add `configs/model/name.yaml`; put model-specific hyperparameters in `model_params:` dict in YAML |
| New node feature | `src/datasets/feature_builder.py` | Add method `_node_feat_<name>`, enable by name in YAML |
| New fusion strategy | `src/models/fusion/` | Subclass `ModalityFusion`, register, use via `model.fusion` config |
| New label composite | `src/datasets/composite_registry.py` | Register a new composite method |
