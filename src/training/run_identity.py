"""Run identity helpers — a globally-unique name for every execution.

Every nested-CV execution writes its artifacts (per-fold predictions,
checkpoints, ``nested_cv_result.json``) under
``<checkpoint_dir>/<run_name>/``. For those artifacts to survive across
runs, ``run_name`` must uniquely identify a single execution — the human
``experiment_name`` alone is a *reusable recipe label* (re-running the
same experiment would otherwise clobber the previous run's predictions).
See ADR-0012.

``run_name`` is therefore ``"<experiment_name>-<run_uid>"`` where the uid
is the Slurm job id on the cluster (which ties the artifact directory back
to the Slurm log, ``cluster-history``, and the ``Job ID`` recorded in
``EXPERIMENTS.md``), falling back to a UTC-timestamp-plus-pid string for
local runs where no Slurm job id exists.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional


def _utc_stamp() -> str:
    """``YYYYmmdd-HHMMSS`` in UTC. Factored out so tests can pin the clock."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def resolve_run_uid() -> str:
    """Return a uid that uniquely identifies this execution.

    Prefers ``$SLURM_JOB_ID`` (unique per submission, monotonic, and
    already the canonical run handle in this project's cluster workflow).
    Falls back to ``<utc-timestamp>-<pid>`` for local runs, where the pid
    disambiguates concurrent processes started in the same wall-clock
    second.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return job_id
    return f"{_utc_stamp()}-{os.getpid()}"


def build_run_name(experiment_name: str, uid: Optional[str] = None) -> str:
    """Compose the globally-unique run name ``"<experiment_name>-<uid>"``.

    Parameters
    ----------
    experiment_name : str
        The reusable recipe label (``cfg.experiment_name``).
    uid : str, optional
        Override the uid (mainly for tests). Defaults to
        :func:`resolve_run_uid`.
    """
    if uid is None:
        uid = resolve_run_uid()
    return f"{experiment_name}-{uid}"
