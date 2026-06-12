#!/bin/bash
#SBATCH --job-name=gnn_bench_sklearn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# NOTE: no --gres / no --partition / no --qos here on purpose.
# Submit via `cluster-submit --node <cpunode> slurm/train_sklearn.sh ...`,
# which injects --partition/--qos for the chosen CPU node and caps cpus.

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

case "${RUN_ARGS:-}" in
    *"dataset=orbit"*) DATASET_ROOT_OVERRIDE="dataset.root=/data/bdip_ssd/al5165/GNNBenchV2/data/ORBIT" ;;
    *"dataset=pnc"*)   DATASET_ROOT_OVERRIDE="dataset.root=/data/bdip_ssd/al5165/GNNBenchV2/data/PNC" ;;
    *)                 DATASET_ROOT_OVERRIDE="" ;;
esac

echo "[run] SHA=$(git rev-parse HEAD)"
echo "[run] host=$(hostname)"
echo "[run] container=$SIF"
echo "[run] RUN_ARGS=${RUN_ARGS:-<none>}"

# CPU-only: singularity WITHOUT --nv
PYTHONPATH="$(pwd)" singularity exec \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    --bind /data/bdip_ssd/al5165/GNNBenchV2/data:/data/bdip_ssd/al5165/GNNBenchV2/data \
    "$SIF" \
    python scripts/run_experiment.py \
        logging.entity=teampolpetta \
        logging.project=baselines \
        ${RUN_ARGS:-} \
        ${DATASET_ROOT_OVERRIDE:-}
