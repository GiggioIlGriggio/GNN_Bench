"""Tests for the configurable `model.norm` knob and the build_norm factory.

All tests use synthetic data — no real datasets.
"""

from __future__ import annotations

import pytest

from src.configs.model_config import ModelConfig


# --- Config validation -----------------------------------------------------

def test_norm_defaults_to_batch() -> None:
    assert ModelConfig(name="test").norm == "batch"


def test_norm_accepts_known_kinds() -> None:
    for kind in ("batch", "layer", "none"):
        assert ModelConfig(name="test", norm=kind).norm == kind


def test_norm_rejects_unknown_kind() -> None:
    with pytest.raises(Exception):  # pydantic.ValidationError
        ModelConfig(name="test", norm="bogus")
