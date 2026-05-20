#!/usr/bin/env bash
set -euo pipefail

# Sweep: BrainGNN on PNC structural connectivity → VWM d-prime
# Uses Optuna Bayesian search over 6 params, 50 trials.

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

python "$(dirname "$0")/run_experiment.py" \
    --multirun \
    dataset=pnc \
    model=braingnn \
    features=default \
    labels=pnc_VWMdprime \
    sweeper=braingnn_vwm \
    objective_metric=r2 \
    logging.project=braingnn_to_VWM
