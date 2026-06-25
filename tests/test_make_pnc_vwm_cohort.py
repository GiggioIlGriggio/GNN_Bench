"""Unit tests for the PNC cohort generator's CLI arg parsing + intersection.

Positionals are ``[OUT_PATH] [LABELS] [FEATURES] [ALSO_REQUIRE]``; ALSO_REQUIRE is
a comma-separated list of extra label configs whose targets must ALSO be non-NaN
(the TRITASK cohort = VWM ∩ PCPT). Any arg containing ``=`` is a Hydra override.
"""

from __future__ import annotations

from scripts.make_pnc_vwm_cohort import parse_args, intersect_cohorts


def test_defaults_when_no_args():
    out_path, labels, features, also_require, extra = parse_args([])
    assert out_path.name == "pnc_vwm_cohort.txt"
    assert labels == "pnc_VWMdprime"
    assert features == "glm_diagonal"
    assert also_require == []
    assert extra == []


def test_positional_out_labels_features():
    out_path, labels, features, also_require, extra = parse_args(
        ["my_cohort.txt", "pnc_default", "identity"]
    )
    assert out_path.name == "my_cohort.txt"
    assert labels == "pnc_default"
    assert features == "identity"
    assert also_require == []
    assert extra == []


def test_single_override_only():
    out_path, labels, features, also_require, extra = parse_args(["dataset.root=/cluster/PNC"])
    assert out_path.name == "pnc_vwm_cohort.txt"
    assert labels == "pnc_VWMdprime"
    assert features == "glm_diagonal"
    assert also_require == []
    assert extra == ["dataset.root=/cluster/PNC"]


def test_positionals_plus_overrides_any_order():
    out_path, labels, features, also_require, extra = parse_args(
        ["out.txt", "dataset.root=/data/PNC", "pnc_default", "glm_diagonal", "dataset.num_workers=1"]
    )
    assert out_path.name == "out.txt"
    assert labels == "pnc_default"
    assert features == "glm_diagonal"
    assert also_require == []
    assert extra == ["dataset.root=/data/PNC", "dataset.num_workers=1"]


def test_also_require_single():
    out_path, labels, features, also_require, extra = parse_args(
        ["tritask.txt", "pnc_VWMdprime", "glm_diagonal", "pnc_PCPT_accuracy", "dataset.root=/c"]
    )
    assert out_path.name == "tritask.txt"
    assert also_require == ["pnc_PCPT_accuracy"]
    assert extra == ["dataset.root=/c"]


def test_also_require_comma_separated():
    _o, _l, _f, also_require, _e = parse_args(
        ["t.txt", "pnc_VWMdprime", "glm_diagonal", "pnc_PCPT_accuracy,pnc_PCPT_RT"]
    )
    assert also_require == ["pnc_PCPT_accuracy", "pnc_PCPT_RT"]


def test_intersect_cohorts():
    base = ["sub-1", "sub-2", "sub-3"]
    extras = [["sub-2", "sub-3", "sub-9"], ["sub-3", "sub-2"]]
    assert intersect_cohorts(base, extras) == ["sub-2", "sub-3"]


def test_intersect_cohorts_no_extras_returns_sorted_base():
    assert intersect_cohorts(["sub-3", "sub-1", "sub-2"], []) == ["sub-1", "sub-2", "sub-3"]
