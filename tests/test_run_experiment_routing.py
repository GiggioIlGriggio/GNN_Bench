"""Unit tests for select_runner() pure routing function in run_experiment.

These tests are fast, pure-Python, and require no Hydra / dataset / GPU.
They cover the four precedence rules documented in select_runner()'s docstring.
"""

from __future__ import annotations

import pytest

# Verify the namespace-package import path works before any test collects.
from scripts.run_experiment import select_runner


class TestFlatCvDispatch:
    """Handoff test #1 — flat_cv → CrossValidator, NOT nested."""

    def test_flat_cv_returns_flat_cv(self):
        result = select_runner(is_sweep=False, finetuning_enabled=False, runner="flat_cv")
        assert result == "flat_cv"


class TestNestedRegressionGuard:
    """Handoff test #2 — default path must stay nested (regression guard)."""

    def test_nested_string_returns_nested(self):
        result = select_runner(is_sweep=False, finetuning_enabled=False, runner="nested")
        assert result == "nested"

    def test_none_runner_returns_nested(self):
        result = select_runner(is_sweep=False, finetuning_enabled=False, runner=None)
        assert result == "nested"


class TestSweepPrecedence:
    """Sweep (--multirun) overrides every other flag, including runner=flat_cv."""

    def test_sweep_overrides_flat_cv(self):
        result = select_runner(is_sweep=True, finetuning_enabled=False, runner="flat_cv")
        assert result == "sweep"

    def test_sweep_overrides_nested(self):
        result = select_runner(is_sweep=True, finetuning_enabled=False, runner="nested")
        assert result == "sweep"

    def test_sweep_overrides_none(self):
        result = select_runner(is_sweep=True, finetuning_enabled=False, runner=None)
        assert result == "sweep"

    def test_sweep_overrides_finetune(self):
        result = select_runner(is_sweep=True, finetuning_enabled=True, runner=None)
        assert result == "sweep"


class TestFinetuningPrecedence:
    """Finetuning enabled takes precedence over runner= when not a sweep."""

    def test_finetune_overrides_flat_cv(self):
        result = select_runner(is_sweep=False, finetuning_enabled=True, runner="flat_cv")
        assert result == "finetune"

    def test_finetune_overrides_nested(self):
        result = select_runner(is_sweep=False, finetuning_enabled=True, runner="nested")
        assert result == "finetune"

    def test_finetune_overrides_none(self):
        result = select_runner(is_sweep=False, finetuning_enabled=True, runner=None)
        assert result == "finetune"


class TestUnknownRunnerRejected:
    """Unknown runner= values must raise ValueError, never silently fall through."""

    def test_typo_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown runner"):
            select_runner(is_sweep=False, finetuning_enabled=False, runner="flat_cvv")

    def test_none_does_not_raise(self):
        # runner=None is the default/unset case — must never raise.
        result = select_runner(is_sweep=False, finetuning_enabled=False, runner=None)
        assert result == "nested"

    def test_nested_string_does_not_raise(self):
        # runner="nested" is explicit but valid — must never raise.
        result = select_runner(is_sweep=False, finetuning_enabled=False, runner="nested")
        assert result == "nested"


from scripts.run_experiment import require_nested_for_transfer


class TestRequireNestedForTransfer:
    """Transfer OR a non-empty frozen_layers requires the nested runner."""

    def test_enabled_non_nested_raises(self):
        with pytest.raises(ValueError, match="requires the nested runner"):
            require_nested_for_transfer(transfer_enabled=True, frozen_layers=[], runner="flat_cv")

    def test_frozen_layers_non_nested_raises(self):
        # frozen-random: enabled=False but frozen_layers set must still demand nested.
        with pytest.raises(ValueError, match="requires the nested runner"):
            require_nested_for_transfer(transfer_enabled=False, frozen_layers=["backbone"], runner="flat_cv")

    def test_nested_ok(self):
        require_nested_for_transfer(transfer_enabled=True, frozen_layers=["backbone"], runner="nested")

    def test_no_transfer_no_frozen_ok(self):
        require_nested_for_transfer(transfer_enabled=False, frozen_layers=[], runner="flat_cv")

    def test_frozen_only_nested_ok(self):
        # frozen-random: enabled=False, frozen_layers set, runner=nested — must not raise.
        require_nested_for_transfer(transfer_enabled=False, frozen_layers=["backbone"], runner="nested")
