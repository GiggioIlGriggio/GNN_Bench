"""Unit tests for the PNC VWM-cohort generator's CLI arg parsing.

The generator's positional args are ``[OUT_PATH] [LABELS] [FEATURES]``; any arg
containing ``=`` is forwarded verbatim as an extra Hydra override (so the cluster
wrapper can redirect ``dataset.root`` to the cluster PNC path — the laptop path is
baked into ``configs/dataset/pnc.yaml`` and does not exist on the cluster).
"""

from __future__ import annotations

from scripts.make_pnc_vwm_cohort import parse_args


def test_defaults_when_no_args():
    out_path, labels, features, extra = parse_args([])
    assert out_path.name == "pnc_vwm_cohort.txt"
    assert labels == "pnc_VWMdprime"
    assert features == "glm_diagonal"
    assert extra == []


def test_positional_out_labels_features():
    out_path, labels, features, extra = parse_args(
        ["my_cohort.txt", "pnc_default", "identity"]
    )
    assert out_path.name == "my_cohort.txt"
    assert labels == "pnc_default"
    assert features == "identity"
    assert extra == []


def test_single_override_only():
    out_path, labels, features, extra = parse_args(["dataset.root=/cluster/PNC"])
    # positionals fall back to defaults; the '=' arg becomes an override.
    assert out_path.name == "pnc_vwm_cohort.txt"
    assert labels == "pnc_VWMdprime"
    assert features == "glm_diagonal"
    assert extra == ["dataset.root=/cluster/PNC"]


def test_positionals_plus_overrides_any_order():
    out_path, labels, features, extra = parse_args(
        [
            "out.txt",
            "dataset.root=/data/PNC",
            "pnc_default",
            "glm_diagonal",
            "dataset.num_workers=1",
        ]
    )
    assert out_path.name == "out.txt"
    assert labels == "pnc_default"
    assert features == "glm_diagonal"
    assert extra == ["dataset.root=/data/PNC", "dataset.num_workers=1"]
