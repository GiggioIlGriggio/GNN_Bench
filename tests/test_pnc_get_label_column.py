"""Tests for ``PNCDataset.get_label_column`` (PNC override).

The base-class ``get_label_column`` reuses ``LabelBuilder``'s single-target
path, which casts the ``sub-XXXXXXXXXX`` filename id to an int — that cannot
recover PNC's 12-digit ``SUBJID`` (``6000XXXXXXXXXX``).  PNC therefore overrides
``get_labels`` / ``get_label_components`` to read directly from the per-subject
metadata dicts, and must do the same for ``get_label_column`` so a source run
can stratify its CV folds on a column other than its own regression target
(e.g. an age-source run stratified on VWM).

These tests bypass ``__init__`` (which does disk I/O) via ``__new__`` and set
only the attributes the override reads: ``self._subject_ids`` and
``self._raw_data`` (a dict of objects each exposing a ``.metadata`` dict).
"""

import numpy as np
import pytest

from src.datasets.pnc_dataset import PNCDataset


class _StubRaw:
    """Tiny stand-in for ``RawGraphData`` exposing only ``.metadata``.

    The override touches nothing but ``.metadata``, so a full ``RawGraphData``
    (with edge tensors etc.) is unnecessary scaffolding.
    """

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


def _make_ds():
    """A PNCDataset with ``__init__`` bypassed and minimal state set."""
    ds = PNCDataset.__new__(PNCDataset)
    ds._subject_ids = ["sub-0001", "sub-0002", "sub-0003"]
    ds._raw_data = {
        "sub-0001": _StubRaw({"age": 10.0, "vwm": 0.5}),
        "sub-0002": _StubRaw({"age": 14.0, "vwm": 1.5}),
        "sub-0003": _StubRaw({"age": 18.0, "vwm": -0.2}),
    }
    return ds


def test_get_label_column_reads_metadata_in_subject_order():
    ds = _make_ds()

    vec = ds.get_label_column("vwm")

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.dtype(float)
    assert vec.tolist() == [0.5, 1.5, -0.2]


def test_get_label_column_supports_the_regression_target_too():
    """Selecting the dataset's own target column must also work."""
    ds = _make_ds()

    vec = ds.get_label_column("age")

    assert vec.tolist() == [10.0, 14.0, 18.0]


def test_get_label_column_missing_column_raises_keyerror():
    """A typo'd / absent column must fail loudly, not yield silent NaN."""
    ds = _make_ds()

    with pytest.raises(KeyError):
        ds.get_label_column("nonexistent_column")
