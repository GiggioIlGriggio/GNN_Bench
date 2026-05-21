#!/bin/bash
#SBATCH --job-name=gnn_bench_sweep
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=47:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

echo "[run] SHA=$(git rev-parse HEAD)"
echo "[run] host=$(hostname)"
echo "[run] container=$SIF"
echo "[run] RUN_ARGS=${RUN_ARGS:-<none>}"
nvidia-smi || true

PYTHONPATH="$(pwd)" singularity exec --nv \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    "$SIF" \
    python scripts/run_experiment.py \
        --multirun \
        logging.project=orbitglm \
        logging.entity=teampolpetta \
        ${RUN_ARGS:-}
