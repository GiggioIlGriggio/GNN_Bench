# CONTEXT.md — GNN Brain Connectivity Benchmarking Platform

This file is the canonical source of domain language and architecture for this codebase.
Engineering skills read it before exploring code. `/grill-with-docs` updates it lazily as terms get resolved.

---

## Domain overview

This platform benchmarks Graph Neural Networks for predicting **behavioural or cognitive scores** from **brain connectivity data**. Each research subject becomes one or more graphs; the GNN is trained to regress a scalar target (age, memory score, accuracy, etc.) from the graph structure and node features. All experiments use **stratified K-fold cross-validation** to produce unbiased performance estimates.

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
| **Fold** | One split of the dataset in K-fold cross-validation. Each fold has a train set, a val set, and a test set. | "split" (ambiguous — could mean train/val/test) |
| **Label normalisation** | Per-fold standardisation of the scalar regression target. Fit only on training subjects; applied to val and test. Prevents data leakage. | "target scaling", "output normalisation" |
| **Composite label** | A regression target derived from multiple raw score columns, either by weighted combination or PCA. Built per-fold by `LabelBuilder` to avoid leakage from PCA fitting. | "multi-component label" |
| **GLM normalisation** | Per-node z-scoring of GLM feature columns across training subjects. Applied per fold to prevent leakage. Controlled by `FeatureConfig.glm_normalize`. | — |
| **Backbone** | The GNN message-passing layers (GCN, GAT, GIN, Graph Transformer). Produces node-level embeddings. | "encoder base", "GNN layers" |
| **Fusion** | A module that combines SC and FC embeddings in multimodal mode (concat, cross-attention, gated). | "aggregation" (ambiguous) |
| **Head** | The MLP that maps the pooled graph embedding to a scalar prediction. Currently only `RegressionHead`. | "decoder", "output layer" |
| **Pooling** | Global graph pooling (mean, max, add, attention) that converts node embeddings to a graph-level embedding. | "readout" (accepted in PyG literature) |
| **ROI-TopK Pooling** | Hierarchical differentiable node selection used in BrainGNN. A learned scoring function ranks each ROI; the top-k fraction are kept and the graph is coarsened. Implemented via `torch_geometric.nn.TopKPooling`. Distinct from global pooling — it reduces the graph mid-network, not at the end. | "TopK pool", "node pooling" (ambiguous — say "ROI-TopK pooling") |
| **Auxiliary loss** | An extra loss term beyond MSE, emitted by models that implement `auxiliary_loss() → dict[str, Tensor]`. The trainer adds these to the main MSE loss and logs them separately to wandb. Currently only BrainGNN emits aux losses. | "regularisation loss" (too broad) |
| **Unit loss** | BrainGNN auxiliary loss that penalises pooling scores far from 0 or 1, encouraging binary (hard) node selection. Weight controlled by `model_params.unit_loss_weight`. | — |
| **TopK loss** | BrainGNN auxiliary loss that encourages consistent top-K ROI selection across training samples. Weight controlled by `model_params.topk_loss_weight`. | — |
| **model_params** | A free-form `dict` field in `ModelConfig` for model-specific hyperparameters (e.g. `pool_ratio`, `unit_loss_weight`). Read by the model constructor; ignored by all other models. Avoids adding model-specific fields to the shared `ModelConfig` schema. | "extra_params", "kwargs" |
| **Checkpoint** | Saved model `state_dict` + `LabelNormalizer` + metrics for one fold. Stored under `checkpoints/fold_<N>/`. | "model weights", "snapshot" (use "epoch snapshot" for epoch-level saves) |
| **Epoch snapshot** | A model `state_dict` saved mid-training (either every N epochs or on best-val). Stored in `checkpoints/fold_<N>/epoch_checkpoints/`. | — |
| **Allowlist** | An optional text file listing subject IDs to include. When set, subjects not in the list are silently skipped; IDs in the list that are missing on disk raise an error. | "whitelist" |

---

## Architecture map

```
scripts/run_experiment.py          # Hydra entry point — orchestrates everything
│
├── configs/                        # YAML config files (one group per module)
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
│   ├── cross_validation.py         # CrossValidator: stratified K-fold, per-fold label/GLM normalisation
│   ├── label_normalizer.py         # LabelNormalizer: standard/robust/minmax/none, serialisable
│   ├── glm_normalizer.py           # GLMFeatureNormalizer: per-node z-score of GLM columns
│   ├── metrics.py                  # compute_metrics (MAE, RMSE, R², Pearson r), aggregate_fold_metrics
│   └── checkpoint_manager.py       # CheckpointManager: save/load fold and epoch checkpoints
│
├── src/sweeps/                     # Hyperparameter sweeps
│   ├── base_sweep.py               # SweepRunner (abstract)
│   └── hydra_sweep.py              # HydraSweep: one Optuna trial per Hydra multirun call
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

### Data flow (standard cross-validation run)

```
Dataset.load_raw()
  → Dataset.get_dataset()          # list of PyG Data objects (y=placeholder for composite)
  → Dataset.get_labels()           # np.ndarray [N] for stratification
  → CrossValidator.split()         # yields (train_idx, val_idx, test_idx) per fold
    → LabelBuilder.fit_transform() # per-fold composite label (if composite mode)
    → LabelNormalizer.fit()        # fit on train labels only
    → GLMFeatureNormalizer.fit()   # fit on train node features only (if GLM)
    → Trainer.fit()                # training loop with early stopping
    → Trainer.predict()            # inference on test set
    → CheckpointManager.save()     # fold checkpoint
  → CVResult (per-fold + pooled metrics)
  → GNNExplainerRunner.run()       # optional, post-training
```

### Execution modes

The entry point detects which mode to run based on CLI flags and config:

| Mode | Trigger | What happens |
|---|---|---|
| **Standard CV** | No `--multirun`, `finetuning.enabled=false` | K-fold cross-validation, saves checkpoints |
| **Sweep** | `--multirun` with a sweeper config | One Optuna trial per process; best trial runs GNNExplainer |
| **Finetuning** | `finetuning.enabled=true` | Loads pretrained checkpoint, freezes layers, runs CV |
| **Epoch-sweep finetuning** | `finetuning.epoch_checkpoint_dir` set | Finetunes from every epoch snapshot, logs R² vs pretrain epoch |

---

## Key invariants

- **No data leakage**: label normalisation, GLM normalisation, and composite-label PCA are always fit on training subjects only, then applied to val and test.
- **Graph contract enforced**: every `build_graph` implementation calls `validate_graph_contract()` before returning. Downstream code may assume the contract holds.
- **Per-fold fresh model**: `CrossValidator` calls `model_factory()` to get a new untrained model at the start of each fold.
- **Best-epoch model used for test**: after training, `result.best_model_state_dict` (selected by val metric) is loaded before evaluating on the test split.
- **Wandb keys are constants**: no bare strings — all wandb metric keys are `Final[str]` values in `log_schema.py`.

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
