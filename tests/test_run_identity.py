"""Tests for run-identity helpers (ADR-0012, per-run checkpoint dir)."""

from __future__ import annotations

import re

from src.training.run_identity import build_run_name, resolve_run_uid


def test_resolve_run_uid_prefers_slurm_job_id(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "360751")
    assert resolve_run_uid() == "360751"


def test_resolve_run_uid_falls_back_to_timestamp_when_no_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    uid = resolve_run_uid()
    # UTC timestamp + pid: YYYYmmdd-HHMMSS-<pid>
    assert re.fullmatch(r"\d{8}-\d{6}-\d+", uid), uid


def test_build_run_name_combines_experiment_and_uid():
    assert build_run_name("gcn-pnc-sc-vwm-id-glmdiag", uid="360751") == (
        "gcn-pnc-sc-vwm-id-glmdiag-360751"
    )


def test_build_run_name_resolves_uid_from_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    assert build_run_name("exp") == "exp-999"


def test_two_local_runs_get_distinct_uids(monkeypatch):
    """Fallback uids must not collide for re-runs of the same experiment."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # Same wall-clock second, distinct pids -> distinct uids.
    monkeypatch.setattr(
        "src.training.run_identity._utc_stamp", lambda: "20260601-143022"
    )
    monkeypatch.setattr("os.getpid", lambda: 111)
    a = build_run_name("exp")
    monkeypatch.setattr("os.getpid", lambda: 222)
    b = build_run_name("exp")
    assert a != b
    assert a == "exp-20260601-143022-111"
    assert b == "exp-20260601-143022-222"
