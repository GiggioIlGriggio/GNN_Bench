# GNN Brain Connectivity Benchmarking Platform

A fully modular, scaffolded platform for benchmarking Graph Neural Networks on
brain connectivity data (structural and functional connectomes).

---

## Overview

This platform provides:

- **Dataset loading** — ORBIT dataset with structural connectivity (SC) matrices
  from `.mat` files and functional connectivity (FC) matrices from `.npy` files,
  with extensible registry for adding new datasets.
- **Feature engineering** — Pluggable node and edge feature builders
  (degree, strength, betweenness, clustering, …).
- **Model zoo** — GCN, GAT, GIN, Graph Transformer backbones;
  concat / cross-attention / gated fusion modules; regression head.
- **Training** — Pure-PyTorch training loop, stratified K-fold
  cross-validation, per-fold label normalisation, early stopping, checkpoint management.
- **Hyperparameter sweeps** — Hydra + Optuna Bayesian search.
- **Fine-tuning** — Load pretrained backbone, freeze layers, configure LR groups.
- **Logging** — Typed WandB logger with schema-enforced keys.
- **Longitudinal support** — Process multiple timepoints (e.g. T0, T1) in a
  single run; each subject-timepoint pair becomes a separate graph.

---

## Installation

```bash
# 1. Create environment (Python ≥ 3.10)
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### PyTorch Geometric

PyG wheel selection depends on your CUDA version. Visit
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>
and install `torch-scatter`, `torch-sparse`, `torch-cluster`,
`torch-spline-conv` before `torch-geometric`.

---

## Quick start

```bash
# Run default experiment (GCN on ORBIT, age prediction, 5-fold CV)
python scripts/run_experiment.py

# Override from CLI
python scripts/run_experiment.py model=gat trainer.lr=0.0005
```

---

## Project structure

```
project/
├── INTERFACES.md            # Full interface specification
├── DECISIONS.md             # Design-decision log
├── README.md                # This file
├── requirements.txt
│
├── configs/                 # Hydra YAML configurations
│   ├── experiment.yaml      # Top-level composed entry point
│   ├── dataset/
│   │   └── orbit.yaml
│   ├── features/
│   │   └── default.yaml
│   ├── labels/
│   │   └── default.yaml
│   ├── model/
│   │   ├── gcn.yaml
│   │   └── gat.yaml
│   ├── trainer/
│   │   └── default.yaml
│   ├── sweep/
│   │   └── bayesian.yaml
│   ├── finetuning/
│   │   └── default.yaml
│   └── logging/
│       └── wandb.yaml
│
├── scripts/
│   └── run_experiment.py    # Hydra entry point
│
├── src/
│   ├── configs/             # Pydantic v2 config schemas
│   ├── datasets/            # Dataset loading, feature/label building
│   ├── models/              # Backbones, fusion, heads, registry
│   │   ├── backbones/
│   │   ├── fusion/
│   │   └── heads/
│   ├── training/            # Trainer, CV, metrics, checkpoints
│   ├── sweeps/              # Hydra + Optuna sweep runner
│   ├── finetuning/          # Pretrained model fine-tuning
│   ├── logging/             # WandB logger with typed schema
│   ├── interfaces/          # Adapter utilities
│   └── utils/               # Seed, I/O helpers
│
└── tests/                   # Pytest stubs
    ├── test_datasets.py
    ├── test_models.py
    ├── test_training.py
    └── test_interfaces.py
```

---

## Configuration system

All configuration is managed via **Hydra** with structured YAML files.
Every YAML group has a matching **pydantic v2** schema in `src/configs/`.

| Config group   | Schema                     | YAML example               |
| -------------- | -------------------------- | -------------------------- |
| dataset        | `DatasetConfig`            | `configs/dataset/orbit.yaml` |
| features       | `FeatureConfig`            | `configs/features/default.yaml` |
| labels         | `LabelConfig`              | `configs/labels/default.yaml` |
| model          | `ModelConfig`              | `configs/model/gcn.yaml`   |
| trainer        | `TrainerConfig`            | `configs/trainer/default.yaml` |
| sweep          | `SweepConfig`              | `configs/sweep/bayesian.yaml` |
| finetuning     | `FinetuningConfig`         | `configs/finetuning/default.yaml` |
| logging        | `LoggingConfig`            | `configs/logging/wandb.yaml` |

Override any field from the command line:

```bash
python scripts/run_experiment.py dataset.atlas=schaefer200 trainer.lr=1e-3
```

---

## Adding a new dataset

1. Create `src/datasets/my_dataset.py`.
2. Subclass `BrainGraphDataset`.
3. Decorate with `@register_dataset("my_dataset")`.
4. Add `configs/dataset/my_dataset.yaml`.

See the ORBIT implementation for a complete reference.

---

## Adding a new backbone

1. Create `src/models/backbones/my_backbone.py`.
2. Subclass `GNNBackbone`.
3. Register via `@register_model("my_backbone")`.
4. Add `configs/model/my_backbone.yaml`.

---

## Supported modalities

| Modality      | Config value    | Description                       |
| ------------- | --------------- | --------------------------------- |
| Structural    | `sc`            | Single SC graph per subject       |
| Functional    | `fc`            | Single FC graph per subject       |
| Multimodal    | `multimodal`    | Both SC + FC fused via fusion module |

---

## Tests

```bash
pytest tests/ -v
```

All test functions are stubs (`raise NotImplementedError`).
Implement the TODO bodies before running.

---

## License

MIT
