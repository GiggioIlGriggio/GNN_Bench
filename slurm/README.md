# SLURM Launcher — `gnn_bench`

This folder contains the Slurm job scripts and batch launcher that the
`/cluster-helper` skill drives. The cluster is `al5165@155.105.223.17`; the
deploy target is `/data/bdip_ssd/al5165/gnn_bench/`.

For the skill's own design and verbs, see
`~/.claude/skills/cluster-helper/README.md` (or the `CONTEXT.md` next to it).
For project-side configuration, see `../.cluster-helper.yaml`.

---

## Folder structure

```
slurm/
├── train.sh             ← standard 5-fold cross-validation job
├── sweep.sh             ← Optuna hyperparameter sweep (--multirun)
├── finetune.sh          ← fine-tuning from checkpoint
├── run_experiments.sh   ← batch launcher (loops over parameter grids)
└── logs/                ← stdout/stderr from Slurm (auto-populated)
```

Each `.sh` is a flat committed bash script. SBATCH headers are hardcoded;
the only runtime input is the `$RUN_ARGS` environment variable, which carries
Hydra overrides.

---

## How Hydra overrides reach the script: `$RUN_ARGS`

`cluster-submit` forwards extra positional args to **sbatch**, not to the
job script. So we cannot pass `dataset=orbit` as a positional arg — sbatch
would reject it. Instead, the slurm script reads `$RUN_ARGS`, and the caller
sets it via Slurm's `--export` mechanism:

```bash
cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default"
```

Inside `slurm/train.sh`, this expands to:

```bash
python scripts/run_experiment.py logging.project=orbitglm logging.entity=teampolpetta \
    dataset=orbit model=gcn features=default labels=default
```

The `ALL,` prefix tells Slurm to also forward the existing environment
(including `WANDB_API_KEY` from `~/.bashrc` on the cluster).

---

## Single submit

```bash
# Standard training
cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default"

# Optuna sweep
cluster-submit slurm/sweep.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default sweeper=bayesian objective_metric=r2"

# Fine-tuning from checkpoint
cluster-submit slurm/finetune.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=identity labels=default finetuning=from_age finetuning.checkpoint_path=checkpoints"
```

After each submit, `cluster-submit` prints `JOB_ID=...` on stdout. Tail the
log with:

```bash
cluster-tail <jobid>
```

---

## Batch experiments

Open `slurm/run_experiments.sh` and edit the arrays at the top:

```bash
MODE="train"                              # train | sweep | finetune
DATASETS=("orbit" "pnc")
MODELS=("gcn" "gat" "gin")
FEATURES=("default" "glm_scalar")
LABELS=("default")
```

Every combination is submitted as an independent Slurm job.

Always dry-run first:

```bash
DRY_RUN=1 bash slurm/run_experiments.sh
```

Then submit for real:

```bash
bash slurm/run_experiments.sh
```

---

## Logs

Slurm writes per-job logs to `slurm/logs/<jobid>.out` and `<jobid>.err` (the
two are gitignored — only `slurm/logs/.gitkeep` is tracked).

```bash
cluster-tail <jobid>           # follow stdout
cluster-tail <jobid> --err     # follow stderr
cluster-status                 # squeue for this user
```

Pull a file back from the cluster to `./cluster_outputs/`:

```bash
cluster-fetch outputs/run42/checkpoint.pt
```

---

## Container

The Singularity image is built locally from `Dockerfile` and rsynced to the
cluster as `${PROJECT_ROOT}/gnn_bench.sif`. Rebuild only when `Dockerfile` or
`requirements.txt` changes:

```bash
cluster-push-container
```

The `.sif` is gitignored — git carries the recipe (`Dockerfile`), not the
image.
