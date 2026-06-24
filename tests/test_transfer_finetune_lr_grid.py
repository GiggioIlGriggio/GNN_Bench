"""Guard: the transfer fine-tune sweeper must not re-introduce lr=5e-3.

Per docs/superpowers/specs/2026-06-13-pnc-age-vwm-transfer-design.md follow-up
(age_vwm_transfer_dynamics): identity full-FT diverged in 7/50 folds at
lr=5e-3 on the dense-SC/identity backbone. The grid is pinned to 1e-4/5e-4/1e-3.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

SWEEPER = Path("configs/sweeper/transfer_finetune.yaml")


def test_lr_grid_excludes_5e3():
    cfg = OmegaConf.load(SWEEPER)
    lr_spec = str(cfg.params["trainer.lr"])
    assert "0.005" not in lr_spec, (
        f"lr=5e-3 diverges identity full-FT (7/50 folds); must stay dropped, got {lr_spec}"
    )


def test_lr_grid_keeps_the_three_stable_values():
    cfg = OmegaConf.load(SWEEPER)
    lr_spec = str(cfg.params["trainer.lr"])
    for v in ("0.0001", "0.0005", "0.001"):
        assert v in lr_spec, f"expected {v} in transfer lr grid, got {lr_spec}"
