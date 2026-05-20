#!/bin/bash
# =============================================================================
# run_experiments.sh — Batch launcher for EF Neural Substrate experiments.
#
# Edit the parameter arrays below to define which combinations to run,
# then execute this script. Each combination is submitted as a separate job.
#
# Usage:
#   bash slurm/run_experiments.sh
#
# Tip: dry-run first to preview submissions without actually submitting:
#   DRY_RUN=1 bash slurm/run_experiments.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_job.sh"

DRY_RUN="${DRY_RUN:-0}"   # set DRY_RUN=1 to preview without submitting

# ===========================================================================
# ▶ CONFIGURE YOUR EXPERIMENT GRID HERE
# ===========================================================================

# --- Mode: train | sweep | finetune ---
MODE="train"

# --- Parameter arrays (all combinations will be submitted) ---
DATASETS=("orbit")                        # orbit | pnc
MODELS=("gcn" "gat")                     # gcn | gat | gin | mlp
FEATURES=("default")                     # default | glm_scalar | glm_diagonal | identity
LABELS=("default")                       # default | pnc_default | ies_immar

# --- Mode-specific settings ---
# For MODE=train: finetuning config to use (usually "default" = disabled)
FINETUNING_CFG="default"

# For MODE=sweep:
SWEEPER_CFG="bayesian"                   # bayesian | gcn_embedding_dim
OBJECTIVE_METRIC="r2"                    # r2 | val_mae

# For MODE=finetune:
FINETUNE_CFG="from_age"                  # finetuning yaml name
CHECKPOINT_PATH="checkpoints"            # path to pretrained checkpoints dir

# --- Optional extra Hydra overrides (space-separated, or empty) ---
# Example: EXTRA_OVERRIDES="trainer.lr=0.001 trainer.epochs=200"
EXTRA_OVERRIDES=""

# ===========================================================================

mkdir -p "${SCRIPT_DIR}/logs"
COUNT=0

echo "Mode: ${MODE}"
echo "Starting job submissions..."
echo "-----------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for features in "${FEATURES[@]}"; do
      for labels in "${LABELS[@]}"; do

        case "${MODE}" in
          train)
            SUBMIT_ARGS=("${MODE}" "${dataset}" "${model}" "${features}" "${labels}" "${FINETUNING_CFG}" ${EXTRA_OVERRIDES})
            ;;
          sweep)
            SUBMIT_ARGS=("${MODE}" "${dataset}" "${model}" "${features}" "${labels}" "${SWEEPER_CFG}" "${OBJECTIVE_METRIC}" ${EXTRA_OVERRIDES})
            ;;
          finetune)
            SUBMIT_ARGS=("${MODE}" "${dataset}" "${model}" "${features}" "${labels}" "${FINETUNE_CFG}" "${CHECKPOINT_PATH}" ${EXTRA_OVERRIDES})
            ;;
          *)
            echo "ERROR: Unknown mode '${MODE}'"
            exit 1
            ;;
        esac

        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "[DRY-RUN] bash ${SUBMIT} ${SUBMIT_ARGS[*]}"
        else
          bash "${SUBMIT}" "${SUBMIT_ARGS[@]}"
          sleep 1   # brief pause to avoid scheduler overload
        fi

        COUNT=$((COUNT + 1))
      done
    done
  done
done

echo "-----------------------------------------------------------"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete — ${COUNT} job(s) would be submitted."
else
  echo "Done — ${COUNT} job(s) submitted."
fi
