#!/bin/bash
# =============================================================================
# submit_job.sh — Submit a single experiment job to SLURM.
#
# Usage:
#   bash slurm/submit_job.sh <mode> <dataset> <model> <features> <labels> \
#                            [<sweeper|finetuning|checkpoint>] \
#                            [extra_overrides...]
#
# Modes:
#   train    — standard 5-fold cross-validation
#   sweep    — Optuna hyperparameter sweep (--multirun)
#   finetune — load pretrained checkpoint and fine-tune
#
# Examples:
#   # Standard training
#   bash slurm/submit_job.sh train orbit gcn default default
#
#   # Sweep
#   bash slurm/submit_job.sh sweep orbit gcn default default bayesian r2
#
#   # Fine-tuning from a pretrained checkpoint
#   bash slurm/submit_job.sh finetune orbit gcn identity default \
#       from_age checkpoints/fold_0
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Load cluster & environment config ---
source "${SCRIPT_DIR}/config.sh"

# ------------------------------------------------------------------
# Parse positional arguments
# ------------------------------------------------------------------
MODE="${1:?Usage: submit_job.sh <train|sweep|finetune> <dataset> <model> <features> <labels> [sweeper|finetuning|checkpoint] [extra...]}"
DATASET="${2:?Missing dataset argument}"
MODEL="${3:?Missing model argument}"
FEATURES="${4:?Missing features argument}"
LABELS="${5:?Missing labels argument}"
shift 5

# Mode-specific 6th argument
SWEEPER="null"
FINETUNING="default"
CHECKPOINT_PATH="checkpoints"
OBJECTIVE_METRIC="r2"

case "${MODE}" in
  train)
    FINETUNING="${1:-default}"
    shift 2>/dev/null || true
    ;;
  sweep)
    SWEEPER="${1:?sweep mode requires a sweeper name (e.g. bayesian)}"
    OBJECTIVE_METRIC="${2:-r2}"
    shift 2>/dev/null || true
    ;;
  finetune)
    FINETUNING="${1:-from_age}"
    CHECKPOINT_PATH="${2:-checkpoints}"
    shift 2>/dev/null || true
    ;;
  *)
    echo "ERROR: Unknown mode '${MODE}'. Choose from: train | sweep | finetune"
    exit 1
    ;;
esac

# Remaining args become extra Hydra overrides
EXTRA_OVERRIDES="${*:-}"

# ------------------------------------------------------------------
# Select template and wall-time based on mode
# ------------------------------------------------------------------
case "${MODE}" in
  train)
    TEMPLATE="${SCRIPT_DIR}/templates/train.slurm"
    WALL_TIME="${TIME_TRAIN}"
    ;;
  sweep)
    TEMPLATE="${SCRIPT_DIR}/templates/sweep.slurm"
    WALL_TIME="${TIME_SWEEP}"
    ;;
  finetune)
    TEMPLATE="${SCRIPT_DIR}/templates/finetune.slurm"
    WALL_TIME="${TIME_FINETUNE}"
    ;;
esac

# ------------------------------------------------------------------
# Create logs directory
# ------------------------------------------------------------------
mkdir -p "${SCRIPT_DIR}/logs"

# ------------------------------------------------------------------
# Generate temp SLURM script by substituting all placeholders
# ------------------------------------------------------------------
RAND_TAG="${MODE}_${MODEL}_${DATASET}_${RANDOM}"
TMP_SCRIPT="${SCRIPT_DIR}/tmp_${RAND_TAG}.slurm"

sed \
  -e "s|{{PROJECT_ROOT}}|${PROJECT_ROOT}|g" \
  -e "s|{{SINGULARITY_IMAGE}}|${SINGULARITY_IMAGE}|g" \
  -e "s|{{PARTITION}}|${PARTITION}|g" \
  -e "s|{{QOS}}|${QOS}|g" \
  -e "s|{{ACCOUNT}}|${ACCOUNT}|g" \
  -e "s|{{GPUS}}|${GPUS}|g" \
  -e "s|{{CPUS}}|${CPUS}|g" \
  -e "s|{{MEM}}|${MEM}|g" \
  -e "s|{{TIME}}|${WALL_TIME}|g" \
  -e "s|{{DATASET}}|${DATASET}|g" \
  -e "s|{{MODEL}}|${MODEL}|g" \
  -e "s|{{FEATURES}}|${FEATURES}|g" \
  -e "s|{{LABELS}}|${LABELS}|g" \
  -e "s|{{FINETUNING}}|${FINETUNING}|g" \
  -e "s|{{SWEEPER}}|${SWEEPER}|g" \
  -e "s|{{OBJECTIVE_METRIC}}|${OBJECTIVE_METRIC}|g" \
  -e "s|{{CHECKPOINT_PATH}}|${CHECKPOINT_PATH}|g" \
  -e "s|{{WANDB_PROJECT}}|${WANDB_PROJECT}|g" \
  -e "s|{{WANDB_ENTITY}}|${WANDB_ENTITY}|g" \
  -e "s|{{EXTRA_OVERRIDES}}|${EXTRA_OVERRIDES}|g" \
  "${TEMPLATE}" > "${TMP_SCRIPT}"

# ------------------------------------------------------------------
# Submit and clean up
# ------------------------------------------------------------------
echo "Submitting [${MODE}] dataset=${DATASET} model=${MODEL} features=${FEATURES} labels=${LABELS}"
sbatch "${TMP_SCRIPT}"
rm -f "${TMP_SCRIPT}"
