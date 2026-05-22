#!/bin/bash
#SBATCH --job-name=gnn_bench_train
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

# Cluster datasets live alongside the project root, not under it. Map
# dataset name -> cluster-side root so jobs don't try to read the laptop path
# baked into configs/dataset/*.yaml.
case "${RUN_ARGS:-}" in
    *"dataset=orbit"*) DATASET_ROOT_OVERRIDE="dataset.root=/data/bdip_ssd/al5165/GNNBenchV2/data/ORBIT" ;;
    *"dataset=pnc"*)   DATASET_ROOT_OVERRIDE="dataset.root=/data/bdip_ssd/al5165/GNNBenchV2/data/PNC" ;;
    *)                 DATASET_ROOT_OVERRIDE="" ;;
esac

echo "[run] SHA=$(git rev-parse HEAD)"
echo "[run] host=$(hostname)"
echo "[run] container=$SIF"
echo "[run] RUN_ARGS=${RUN_ARGS:-<none>}"
echo "[run] DATASET_ROOT_OVERRIDE=${DATASET_ROOT_OVERRIDE:-<none>}"
nvidia-smi || true

PYTHONPATH="$(pwd)" singularity exec --nv \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    --bind /data/bdip_ssd/al5165/GNNBenchV2/data:/data/bdip_ssd/al5165/GNNBenchV2/data \
    "$SIF" \
    python scripts/run_experiment.py \
        logging.project=orbitglm \
        logging.entity=teampolpetta \
        ${RUN_ARGS:-} \
        ${DATASET_ROOT_OVERRIDE:-}
