#!/bin/bash
#SBATCH --job-name=EF_SWEEP_{{MODEL}}_{{DATASET}}
#SBATCH --output={{PROJECT_ROOT}}/slurm/logs/sweep_%j.log
#SBATCH --error={{PROJECT_ROOT}}/slurm/logs/sweep_%j.err
#SBATCH --partition={{PARTITION}}
#SBATCH --qos={{QOS}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --gres=gpu:{{GPUS}}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}

echo "============================================================"
echo " EF Neural Substrate — Hyperparameter Sweep (--multirun)"
echo " Dataset  : {{DATASET}}"
echo " Model    : {{MODEL}}"
echo " Features : {{FEATURES}}"
echo " Labels   : {{LABELS}}"
echo " Sweeper  : {{SWEEPER}}"
echo " Objective: {{OBJECTIVE_METRIC}}"
echo " Host     : $(hostname)  |  Started: $(date)"
echo "============================================================"

cd {{PROJECT_ROOT}}

# --- Run ---
PYTHONPATH={{PROJECT_ROOT}} singularity exec --nv {{SINGULARITY_IMAGE}} python scripts/run_experiment.py \
    --multirun \
    dataset={{DATASET}} \
    model={{MODEL}} \
    features={{FEATURES}} \
    labels={{LABELS}} \
    sweeper={{SWEEPER}} \
    objective_metric={{OBJECTIVE_METRIC}} \
    logging.project={{WANDB_PROJECT}} \
    logging.entity={{WANDB_ENTITY}} \
    {{EXTRA_OVERRIDES}}

echo "Sweep finished at $(date)"
