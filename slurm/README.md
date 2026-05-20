# SLURM Launcher — EF Neural Substrate

This folder contains everything needed to submit experiment jobs to a SLURM cluster.

---

## Folder structure

```
slurm/
├── config.sh               ← cluster & environment settings (edit this first)
├── submit_job.sh           ← submit a single job
├── run_experiments.sh      ← batch launcher (loops over parameter grids)
├── templates/
│   ├── train.slurm         ← standard cross-validation job template
│   ├── sweep.slurm         ← Optuna hyperparameter sweep template
│   └── finetune.slurm      ← fine-tuning from checkpoint template
└── logs/                   ← stdout/stderr logs (created automatically)
```

---

## Quick-start (3 steps)

### 1. Edit `config.sh`

Set your cluster-specific paths and resources:

```bash
PROJECT_ROOT="/data/bdip_ssd/al5165/EF_neural_substrate"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"

PARTITION="rad2"
QOS="16cpu"
ACCOUNT="rad"
```

### 2. Submit a single job

```bash
# Standard training (orbit dataset, GCN model, default features & labels)
bash slurm/submit_job.sh train orbit gcn default default

# Fine-tuning from a pretrained checkpoint
bash slurm/submit_job.sh finetune orbit gcn identity default \
    from_age checkpoints

# Optuna sweep
bash slurm/submit_job.sh sweep orbit gcn default default bayesian r2
```

### 3. Submit a batch of jobs

Edit the parameter arrays in `run_experiments.sh`, then run:

```bash
# Preview without submitting (dry run)
DRY_RUN=1 bash slurm/run_experiments.sh

# Submit for real
bash slurm/run_experiments.sh
```

---

## Reference: `submit_job.sh`

```
bash slurm/submit_job.sh <mode> <dataset> <model> <features> <labels> \
                         [mode-specific arg(s)] [extra_overrides...]
```

| Argument | Options |
|---|---|
| `mode` | `train` · `sweep` · `finetune` |
| `dataset` | `orbit` · `pnc` |
| `model` | `gcn` · `gat` · `gin` · `mlp` |
| `features` | `default` · `glm_scalar` · `glm_diagonal` · `identity` |
| `labels` | `default` · `pnc_default` · `ies_immar` |

**Mode-specific extra arguments:**

| Mode | 6th arg | 7th arg |
|---|---|---|
| `train` | finetuning cfg (`default`) | — |
| `sweep` | sweeper cfg (`bayesian` / `gcn_embedding_dim`) | objective metric (`r2` / `val_mae`) |
| `finetune` | finetuning cfg (`from_age`) | checkpoint path (`checkpoints`) |

Any remaining arguments are passed directly as Hydra overrides, for example:

```bash
bash slurm/submit_job.sh train orbit gcn default default default \
    trainer.lr=0.0005 trainer.epochs=200
```

---

## Modes explained

### `train` — Standard 5-fold cross-validation

Runs `scripts/run_experiment.py` with the selected config. Checkpoints are saved
to `checkpoints/fold_*/` by default.

```bash
bash slurm/submit_job.sh train orbit gcn glm_scalar default
```

### `sweep` — Optuna hyperparameter search

Runs with `--multirun`. The sweeper config controls the search space and number
of trials (see `configs/sweeper/`).

```bash
bash slurm/submit_job.sh sweep orbit gcn default default bayesian r2
```

Available sweepers:
- `bayesian` — full Bayesian sweep (50 trials)
- `gcn_embedding_dim` — GCN architecture search (250 trials)

### `finetune` — Fine-tuning from a pretrained checkpoint

Loads weights from a previous training run and adapts them to a new target or
dataset.

```bash
bash slurm/submit_job.sh finetune orbit gcn identity default \
    from_age checkpoints
```

The `from_age` finetuning config (in `configs/finetuning/from_age.yaml`) freezes
the backbone and fine-tunes only the prediction head.

---

## Batch experiments — editing `run_experiments.sh`

Open `run_experiments.sh` and edit the arrays at the top:

```bash
MODE="train"

DATASETS=("orbit" "pnc")
MODELS=("gcn" "gat" "gin")
FEATURES=("default" "glm_scalar")
LABELS=("default")
```

Every combination is submitted as an independent SLURM job. With the example
above that is 2 × 3 × 2 × 1 = **12 jobs**.

Always do a dry run first:

```bash
DRY_RUN=1 bash slurm/run_experiments.sh
```

---

## Logs

SLURM stdout/stderr are written to `slurm/logs/`:

```
slurm/logs/train_<jobid>.log
slurm/logs/train_<jobid>.err
slurm/logs/sweep_<jobid>.log
slurm/logs/finetune_<jobid>.log
```

Check a running job:

```bash
tail -f slurm/logs/train_<jobid>.log
```

Check queue:

```bash
squeue -u $USER
```
