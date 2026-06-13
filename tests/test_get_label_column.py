"""Tests for ``BrainGraphDataset.get_label_column``.

``get_label_column(name)`` builds a per-subject vector for an arbitrary
metadata column, reusing the same ``LabelBuilder`` single-target path that
``get_labels`` uses.  This lets an age-source run stratify its CV folds on a
column other than its own regression target (e.g. on VWM), so the folds match
the VWM runs byte-for-byte.
"""

import numpy as np
import pandas as pd

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
