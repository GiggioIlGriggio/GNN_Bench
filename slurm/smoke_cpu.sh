#!/usr/bin/env bash
# Throwaway CPU smoke test for cluster-helper CPU-node support. Not an experiment.
# No container, no GPU. Asks for 32 cpus on purpose to exercise the cpus-per-task
# cap (CPU nodes have 16). Delete this branch after verifying placement.
#SBATCH --job-name=smoke-cpu
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=2G
#SBATCH --time=00:02:00
set -euo pipefail
echo "[smoke] host=$(hostname)"
echo "[smoke] nproc=$(nproc)"
echo "[smoke] partition=${SLURM_JOB_PARTITION:-?}"
echo "[smoke] account=${SLURM_JOB_ACCOUNT:-?}"
echo "[smoke] qos=${SLURM_JOB_QOS:-?}"
echo "[smoke] cpus_per_task=${SLURM_CPUS_PER_TASK:-?}"
echo "[smoke] gpus=${SLURM_JOB_GPUS:-none}"
echo "[smoke] done"
