"""Tests for ``BrainGraphDataset.get_label_column``.

``get_label_column(name)`` builds a per-subject vector for an arbitrary
metadata column, reusing the same ``LabelBuilder`` single-target path that
``get_labels`` uses.  This lets an age-source run stratify its CV folds on a
column other than its own regression target (e.g. on VWM), so the folds match
the VWM runs byte-for-byte.
"""

import numpy as np
import pandas as pd
import pytest

from src.configs.label_config import LabelConfig
from src.datasets.base_dataset import BrainGraphDataset
from src.datasets.label_builder import LabelBuilder


class _StubDataset(BrainGraphDataset):
    """Minimal concrete subclass so the ABC can be instantiated for tests.

    ``BrainGraphDataset`` is abstract (``load_raw`` / ``build_graph``), so it
    cannot be ``__new__``-ed directly.  These stubs are never called — the test
    bypasses ``__init__`` and sets the attributes ``get_label_column`` reads.
    """

    def load_raw(self) -> None:  # pragma: no cover - never invoked
        raise NotImplementedError

    def build_graph(self, subject_id):  # pragma: no cover - never invoked
        raise NotImplementedError


def test_get_label_column_builds_vector_for_named_column():
    # Bypass __init__: __init__ does disk I/O we don't want in a unit test.
    ds = _StubDataset.__new__(_StubDataset)

    # Real attribute shapes (mirror what load_raw populates):
    #  * self._subject_ids : list of "sub-XXX" strings
    #  * self._metadata    : pandas DataFrame with an id_column ("ID" by default)
    #  * self.label_cfg    : LabelConfig carrying the dataset's id_column
    ds._subject_ids = ["sub-001", "sub-002", "sub-003"]
    ds._metadata = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "age": [10.0, 14.0, 18.0],
            "vwm": [0.5, 1.5, -0.2],
        }
    )
    # The dataset's own target is "age"; we stratify on "vwm" instead.
    ds.label_cfg = LabelConfig(target="age")
    ds._label_builder = LabelBuilder(ds.label_cfg)

    vec = ds.get_label_column("vwm")

    assert isinstance(vec, np.ndarray)
    assert vec.tolist() == [0.5, 1.5, -0.2]


def test_get_label_column_does_not_mutate_dataset_label_builder():
    """Building a column vector must not clobber the dataset's own label state."""
    ds = _StubDataset.__new__(_StubDataset)
    ds._subject_ids = ["sub-001", "sub-002", "sub-003"]
    ds._metadata = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "age": [10.0, 14.0, 18.0],
            "vwm": [0.5, 1.5, -0.2],
        }
    )
    ds.label_cfg = LabelConfig(target="age")
    ds._label_builder = LabelBuilder(ds.label_cfg)

    # Sanity: dataset's own target is age.
    age_vec = ds.get_labels()
    assert age_vec.tolist() == [10.0, 14.0, 18.0]

    # Building the vwm column must not change what get_labels returns.
    _ = ds.get_label_column("vwm")
    age_vec_again = ds.get_labels()
    assert age_vec_again.tolist() == [10.0, 14.0, 18.0]


def test_get_label_column_works_when_dataset_config_is_composite():
    """Regression guard: a composite-configured dataset must still extract the
    single named column raw, not run its composite op.

    ``get_label_column`` builds a fresh single-target ``LabelConfig`` via
    ``model_copy`` and explicitly nulls ``composite_columns`` /
    ``composite_method`` / ``composite_params``.  Pydantic's ``model_copy``
    skips validation, so without that nulling a config carrying both ``target``
    and ``composite_columns`` would silently take the composite path (because
    ``is_composite`` keys only off ``composite_columns``), ignore ``target``,
    and corrupt cross-column CV stratification.  This locks the nulling in.
    """
    ds = _StubDataset.__new__(_StubDataset)
    ds._subject_ids = ["sub-001", "sub-002", "sub-003"]
    ds._metadata = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            # Composite component columns (the dataset's own target).
            "vwm_rt": [0.5, 1.5, 2.5],
            "vwm_acc": [0.9, 0.8, 0.7],
            # A separate scalar column we want to stratify on.
            "age": [10.0, 14.0, 18.0],
        }
    )
    # The dataset's OWN config is composite (is_composite is True): target must
    # be None when composite_columns is set, and a "weighted" composite needs
    # one weight per column in composite_params.
    ds.label_cfg = LabelConfig(
        target=None,
        composite_columns=["vwm_rt", "vwm_acc"],
        composite_method="weighted",
        composite_params={"weights": [1.0, 1.0]},
    )
    ds._label_builder = LabelBuilder(ds.label_cfg)
    assert ds.label_cfg.is_composite  # sanity: parent config is composite

    vec = ds.get_label_column("age")

    # Raw "age" values in subject order — proves the single-target path ran,
    # NOT the weighted composite (which would have summed vwm_rt + vwm_acc).
    assert isinstance(vec, np.ndarray)
    assert vec.tolist() == [10.0, 14.0, 18.0]


def test_get_label_column_missing_column_raises():
    """Requesting a column absent from the metadata must fail loudly."""
    ds = _StubDataset.__new__(_StubDataset)
    ds._subject_ids = ["sub-001", "sub-002", "sub-003"]
    ds._metadata = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "age": [10.0, 14.0, 18.0],
            "vwm": [0.5, 1.5, -0.2],
        }
    )
    ds.label_cfg = LabelConfig(target="age")
    ds._label_builder = LabelBuilder(ds.label_cfg)

    with pytest.raises(KeyError):
        ds.get_label_column("nope_not_a_column")
