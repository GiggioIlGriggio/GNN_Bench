#!/bin/bash
# =============================================================================
# config.sh — Cluster & environment settings
# Edit this file to match your cluster before submitting any jobs.
# =============================================================================

# --- Project paths ---
PROJECT_ROOT="/data/bdip_ssd/al5165/GNNBenchV2"   # absolute path on the cluster
# VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"        # virtualenv; replace with conda/module if needed

# To use conda instead, uncomment:
# CONDA_ENV="ef_neural"   # conda activate $CONDA_ENV

# Singularity container
SINGULARITY_IMAGE="${PROJECT_ROOT}/gnn_bench_v2_latest.sif"

# --- SLURM resource defaults (override per-job via CLI flags) ---
PARTITION="rad2"
QOS="16cpu"
ACCOUNT="rad"
GPUS=1
CPUS=4
MEM="32G"
TIME_TRAIN="23:59:00"    # wall-time for standard training jobs
TIME_SWEEP="47:59:00"    # wall-time for sweep/multirun jobs
TIME_FINETUNE="11:59:00" # wall-time for finetuning jobs

# --- WandB project (can be overridden per-run) ---
WANDB_PROJECT="orbitglm"
WANDB_ENTITY="teampolpetta"
