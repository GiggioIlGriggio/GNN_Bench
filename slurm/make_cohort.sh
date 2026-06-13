#!/bin/bash
#SBATCH --job-name=gnn_bench_cohort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# Generate the PNC VWM-cohort subject allowlist ON THE CLUSTER.
#
# Why a cluster job: the allowlist must list exactly the subjects loadable on the
# *cluster* filesystem (graph + non-NaN VWM + GLM map), so it has to be generated
# here, not copied from the laptop. The output (configs/subject_lists/*.txt) is
# git-ignored, so it is NOT shipped via git — but cluster-submit only does
# `git reset --hard` (never `git clean`), so once written it PERSISTS in the
# checkout for every later source/B-arm/A4 submit.
#
# CPU-only (just reads CSVs + graph files): no --gres / no --partition / no --qos.
# Submit via `cluster-submit --node <cpunode> slurm/make_cohort.sh`, which injects
# the CPU node's --partition/--qos/--account and caps cpus.
#
# Optional RUN_ARGS forwards extra generator args, e.g. a different cohort:
#   RUN_ARGS="configs/subject_lists/pnc_vwm_cohort_id.txt pnc_VWMdprime identity"
# Default (empty RUN_ARGS) = the 940-style GLM binding cohort
# (pnc_vwm_cohort.txt, labels=pnc_VWMdprime, features=glm_diagonal).

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

# PNC derivatives live alongside the project root on the cluster, not at the
# laptop path baked into configs/dataset/pnc.yaml — redirect dataset.root.
PNC_ROOT_OVERRIDE="dataset.root=/data/bdip_ssd/al5165/GNNBenchV2/data/PNC"

echo "[cohort] SHA=$(git rev-parse HEAD)"
echo "[cohort] host=$(hostname)"
echo "[cohort] container=$SIF"
echo "[cohort] RUN_ARGS=${RUN_ARGS:-<none>}"
echo "[cohort] root override=$PNC_ROOT_OVERRIDE"

# CPU-only: singularity WITHOUT --nv
PYTHONPATH="$(pwd)" singularity exec \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    --bind /data/bdip_ssd/al5165/GNNBenchV2/data:/data/bdip_ssd/al5165/GNNBenchV2/data \
    "$SIF" \
    python scripts/make_pnc_vwm_cohort.py \
        ${RUN_ARGS:-} \
        "$PNC_ROOT_OVERRIDE"

# Report the result so the job log shows the cohort size at a glance.
COHORT_FILE="configs/subject_lists/pnc_vwm_cohort.txt"
if [[ -f "$COHORT_FILE" ]]; then
    N=$(grep -cvE '^\s*(#|$)' "$COHORT_FILE")
    echo "[cohort] $COHORT_FILE has $N subject ids"
fi
