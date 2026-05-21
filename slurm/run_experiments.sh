#!/bin/bash
# =============================================================================
# run_experiments.sh — Batch launcher for cluster-helper-driven experiments.
#
# Edit the parameter arrays below, then run:
#   bash slurm/run_experiments.sh
# Or preview without submitting:
#   DRY_RUN=1 bash slurm/run_experiments.sh
#
# Internally, each combination becomes:
#   cluster-submit slurm/<MODE>.sh "--export=ALL,RUN_ARGS=dataset=... model=... ..."
# Hydra overrides arrive in the job script via the $RUN_ARGS env var.
# =============================================================================
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

# ===========================================================================
# ▶ CONFIGURE YOUR EXPERIMENT GRID HERE
# ===========================================================================

MODE="train"                              # train | sweep | finetune

DATASETS=("orbit")                        # orbit | pnc
MODELS=("gcn" "gat")                      # gcn | gat | gin | mlp
FEATURES=("default")                      # default | glm_scalar | glm_diagonal | identity
LABELS=("default")                        # default | pnc_default | ies_immar

# Mode-specific extras
FINETUNING_CFG="default"                  # for MODE=train (usually "default" = disabled) or MODE=finetune
SWEEPER_CFG="bayesian"                    # for MODE=sweep
OBJECTIVE_METRIC="r2"                     # for MODE=sweep
CHECKPOINT_PATH="checkpoints"             # for MODE=finetune

EXTRA_OVERRIDES=""                        # space-separated, e.g. "trainer.lr=0.001 trainer.epochs=50"

# ===========================================================================

SCRIPT="slurm/${MODE}.sh"
[[ -f "$SCRIPT" ]] || { echo "ERROR: $SCRIPT does not exist (MODE=$MODE)" >&2; exit 1; }

COUNT=0
echo "Mode: ${MODE}    DRY_RUN=${DRY_RUN}"
echo "-----------------------------------------------------------"

for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for f in "${FEATURES[@]}"; do
      for l in "${LABELS[@]}"; do

        # Compose Hydra overrides (space-separated; one --export arg).
        OVERRIDES="dataset=${d} model=${m} features=${f} labels=${l}"
        case "${MODE}" in
          train)
            OVERRIDES="${OVERRIDES} finetuning=${FINETUNING_CFG}"
            ;;
          sweep)
            OVERRIDES="${OVERRIDES} sweeper=${SWEEPER_CFG} objective_metric=${OBJECTIVE_METRIC}"
            ;;
          finetune)
            OVERRIDES="${OVERRIDES} finetuning=${FINETUNING_CFG} finetuning.checkpoint_path=${CHECKPOINT_PATH}"
            ;;
          *)
            echo "ERROR: Unknown MODE '${MODE}' (expected train|sweep|finetune)" >&2
            exit 1
            ;;
        esac
        if [[ -n "${EXTRA_OVERRIDES}" ]]; then
          OVERRIDES="${OVERRIDES} ${EXTRA_OVERRIDES}"
        fi

        EXPORT_ARG="--export=ALL,RUN_ARGS=${OVERRIDES}"

        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '[DRY-RUN] cluster-submit %s %q\n' "$SCRIPT" "$EXPORT_ARG"
        else
          echo "[submit] $SCRIPT  $EXPORT_ARG"
          cluster-submit "$SCRIPT" "$EXPORT_ARG"
          sleep 1
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
