#!/bin/bash
# =============================================================================
# submit_orbit_vwmhl_matrix.sh — 2026-06-09 ORBIT VWM-HL node-feature × backbone
# matrix. 4 backbones (gcn/gat/gin/transformer) × 7 GLM presets = 28 jobs.
# Reproduces the 2026-06-04 PNC batch with BrainGNN->GCN and PNC->ORBIT.
# Byte-identical 10x5 nested-CV protocol (ADR-0013). One Slurm job per cell.
#
#   bash slurm/submit_orbit_vwmhl_matrix.sh          # submit the 28 real jobs
#   DRY_RUN=1 bash slurm/submit_orbit_vwmhl_matrix.sh  # print, don't submit
#   SMOKE=1   bash slurm/submit_orbit_vwmhl_matrix.sh  # 28 tiny smoke jobs
#
# Env knobs: NODE (default gpunode02), TIME (2-00:00:00), PROJECT (orbitglm),
#            CLUSTER_SUBMIT (default "cluster-submit"; set to a full path when
#            cluster-* is not on PATH).
# =============================================================================
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
NODE="${NODE:-gpunode02}"
TIME="${TIME:-2-00:00:00}"
PROJECT="${PROJECT:-orbitglm}"
CLUSTER_SUBMIT="${CLUSTER_SUBMIT:-cluster-submit}"
SCRIPT="slurm/train.sh"

BACKBONES=(gcn gat gin transformer)
declare -A SWEEPER=(
  [gcn]=gcn_embedding_dim
  [gin]=gcn_embedding_dim
  [gat]=gat_embedding_dim
  [transformer]=transformer_embedding_dim
)

SUFFIXES=(glmdiag id-glmscalar id-glmdiag scprof-glmscalar scprof-glmdiag lappe-glmscalar lappe-glmdiag)
declare -A PRESET=(
  [glmdiag]=glm_diagonal
  [id-glmscalar]=identity_glm_scalar
  [id-glmdiag]=identity_glm_diagonal
  [scprof-glmscalar]=scprofile_glm_scalar
  [scprof-glmdiag]=scprofile_glm_diagonal
  [lappe-glmscalar]=laplacian_pe_glm_scalar
  [lappe-glmdiag]=laplacian_pe_glm_diagonal
)

if [[ "$SMOKE" == "1" ]]; then
  PROTO="trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=2 trainer.epochs=2"
  EXTRA="logging.enabled=false"
  NAME_SUFFIX="-smoke"
else
  PROTO="trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.epochs=300"
  EXTRA=""
  NAME_SUFFIX=""
fi

COUNT=0
for b in "${BACKBONES[@]}"; do
  for s in "${SUFFIXES[@]}"; do
    name="${b}-orbit-sc-vwmhl-${s}${NAME_SUFFIX}"
    overrides="experiment_name=${name} features=${PRESET[$s]} dataset=orbit model=${b} labels=orbit_mri_VWM_HL_p features.glm_normalize=true ${PROTO} trainer.search_space=configs/sweeper/${SWEEPER[$b]}.yaml trainer.hpo_metric=val_r2 logging.project=${PROJECT}${EXTRA:+ $EXTRA}"
    export_arg="--export=ALL,RUN_ARGS=${overrides}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'cluster-submit --node %s %s -J %s --time=%s %q\n' "$NODE" "$SCRIPT" "$name" "$TIME" "$export_arg"
    else
      echo "[submit] $name"
      "$CLUSTER_SUBMIT" --node "$NODE" "$SCRIPT" -J "$name" --time="$TIME" "$export_arg"
      sleep 1
    fi
    COUNT=$((COUNT + 1))
  done
done

echo "-----------------------------------------------------------"
echo "${DRY_RUN:+[DRY-RUN] }${SMOKE:+[SMOKE] }cells: ${COUNT}"
