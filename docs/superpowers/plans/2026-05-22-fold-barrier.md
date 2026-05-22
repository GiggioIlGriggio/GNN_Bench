# FoldBarrier (PR1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `FoldBarrier` as the single coordinator for the three per-fold leakage protections (LabelNormalizer + GLMFeatureNormalizer + composite LabelBuilder), migrate both CV paths and the Trainer signature to use it, and replace `normalizer.pkl` with `barrier.pt` in the fold checkpoint layout. Scope locked by [ADR-0009](../../adr/0009-fold-barrier.md).

**Architecture:** A new module `src/training/fold_barrier.py` owns one `LabelNormalizer`, one `GLMFeatureNormalizer` (optional), and one `LabelBuilder` (optional) as a single fitted bundle per outer fold. The barrier exposes `fit`, `transform_graphs` (returns NEW graphs), `transform_labels`, `inverse_transform_labels`, plus `state_dict` / `save` / `load`. Each underlying transformer gains `state_dict()` / `load_state_dict(state)`. Both `NestedCrossValidator._make_split_loaders` and the legacy `CrossValidator.run()` are rewritten as thin wrappers that construct a barrier and call it. `Trainer.fit` / `Trainer.predict` / `Trainer.evaluate` take a `Callable[[np.ndarray], np.ndarray]` for inverse-transforming predictions (not a `LabelNormalizer`). Persisted as `torch.save(state_dict, "barrier.pt")` — no pickle. `normalizer.pkl` is removed from the checkpoint layout in this PR (B-fresh per ADR-0009 — no migration shim for old checkpoints).

**Tech Stack:** Python 3.10, PyTorch + PyG, pandas, numpy. Existing CV stack in `src/training/`.

**Out of scope:** `CheckpointManager → FoldCheckpoint` rename (PR2). Explainer's `glm_norm` re-fit branch (PR2). Backwards-compat for old `normalizer.pkl` checkpoints (B-fresh).

---

## File Structure

**Created**
- `src/training/fold_barrier.py` — `FoldBarrier` class.
- `tests/test_fold_barrier.py` — unit tests for round-trip, graph non-mutation, save/load.

**Modified**
- `src/training/label_normalizer.py` — add `state_dict()` / `load_state_dict()` (keep existing `save`/`load` JSON API working).
- `src/training/glm_normalizer.py` — add `state_dict()` / `load_state_dict()`.
- `src/datasets/label_builder.py` — add `state_dict()` / `load_state_dict()`.
- `src/training/checkpoint_manager.py` — `save_fold_checkpoint` takes a `barrier: FoldBarrier` instead of `label_normalizer`; writes `barrier.pt`; `FoldCheckpoint.normalizer` field → `FoldCheckpoint.barrier`; `load_fold_checkpoint` and `load_model_for_fold` load `barrier.pt`; `normalizer.pkl` no longer read or written.
- `src/training/trainer.py` — `fit` / `predict` / `evaluate` take `inverse_transform: Callable[[np.ndarray], np.ndarray]` instead of `label_normalizer: LabelNormalizer`. Drop the `LabelNormalizer` import.
- `src/training/nested_cross_validation.py` — `_make_split_loaders` returns `(loaders, barrier)` instead of `(loaders, LabelNormalizer)`; constructs a `FoldBarrier`, removes the five-step manual dance. `_refit_on_trainval` and the outer-test call pass `barrier.inverse_transform_labels` to the Trainer; `save_fold_checkpoint` receives the outer barrier.
- `src/training/cross_validation.py` — `CrossValidator.run`'s per-fold block builds a `FoldBarrier` and uses it for all three protections. `trainer.fit` / `trainer.predict` calls pass `barrier.inverse_transform_labels`.
- `src/finetuning/finetuner.py` — drop the dead `from src.training.label_normalizer import LabelNormalizer` import.
- `CONTEXT.md` — checkpoint layout updated: `normalizer.pkl` → `barrier.pt`. Glossary's `Checkpoint` entry already mentions barrier — verify it's correct.

**Untouched (out of scope)**
- `src/gnn_explainer/explainer.py` — keeps the GLM re-fit branch. PR2 will migrate it.

---

## Task 1: `LabelNormalizer.state_dict` / `load_state_dict`

**Files:**
- Modify: `src/training/label_normalizer.py`
- Test: `tests/test_fold_barrier.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_fold_barrier.py`:

```python
"""Tests for FoldBarrier and its state-dict surface."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


class TestLabelNormalizerStateDict:
    """state_dict round-trip preserves fitted statistics."""

    def test_standard_roundtrip(self) -> None:
        from src.training.label_normalizer import LabelNormalizer

        rng = np.random.default_rng(0)
        y = rng.normal(loc=3.0, scale=2.0, size=200).astype(np.float64)

        n1 = LabelNormalizer(strategy="standard")
        n1.fit(y)
        state = n1.state_dict()
        assert state["strategy"] == "standard"
        assert state["mean"] == pytest.approx(float(np.mean(y)))
        assert state["std"] == pytest.approx(float(np.std(y)))

        n2 = LabelNormalizer(strategy="standard")
        n2.load_state_dict(state)
        np.testing.assert_allclose(n2.transform(y), n1.transform(y))
        np.testing.assert_allclose(
            n2.inverse_transform(n2.transform(y)), y, atol=1e-6
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestLabelNormalizerStateDict::test_standard_roundtrip -v`
Expected: FAIL with `AttributeError: 'LabelNormalizer' object has no attribute 'state_dict'`.

- [ ] **Step 3: Add `state_dict` and `load_state_dict` to `LabelNormalizer`**

Modify `src/training/label_normalizer.py`. Insert directly above the existing `save` method:

```python
    # ------------------------------------------------------------------
    # State-dict interface (composes into FoldBarrier)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return fitted state as a plain dict, suitable for ``torch.save``.

        Returns
        -------
        dict
        """
        return {
            "strategy": self.strategy,
            "mean": self._mean,
            "std": self._std,
            "median": self._median,
            "iqr": self._iqr,
            "min": self._min,
            "max": self._max,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore fitted state from ``state_dict``.

        Parameters
        ----------
        state : dict
        """
        self.strategy = state["strategy"]
        self._mean = state.get("mean")
        self._std = state.get("std")
        self._median = state.get("median")
        self._iqr = state.get("iqr")
        self._min = state.get("min")
        self._max = state.get("max")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestLabelNormalizerStateDict -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/label_normalizer.py tests/test_fold_barrier.py
git commit -m "feat(label_normalizer): add state_dict/load_state_dict surface

Composes the existing fitted statistics into a plain dict so a
FoldBarrier can persist the three per-fold transformers as one
torch-saveable bundle (ADR-0009)."
```

---

## Task 2: `GLMFeatureNormalizer.state_dict` / `load_state_dict`

**Files:**
- Modify: `src/training/glm_normalizer.py`
- Test: `tests/test_fold_barrier.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fold_barrier.py`:

```python
class TestGLMNormalizerStateDict:
    """state_dict carries (mean, std, columns) and survives a roundtrip."""

    def _make_graphs(self, n: int = 8, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        for _ in range(n):
            x = torch.tensor(
                rng.normal(size=(n_rois, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
        return graphs

    def test_roundtrip(self) -> None:
        from src.training.glm_normalizer import GLMFeatureNormalizer

        graphs = self._make_graphs(n=12, n_rois=5)
        n1 = GLMFeatureNormalizer(col_start=1, col_end=3)
        n1.fit(graphs)

        state = n1.state_dict()
        assert state["col_start"] == 1
        assert state["col_end"] == 3
        assert state["fitted"] is True

        n2 = GLMFeatureNormalizer(col_start=0, col_end=0)  # placeholders
        n2.load_state_dict(state)
        assert n2.col_start == 1 and n2.col_end == 3
        assert torch.allclose(n2.mean_, n1.mean_)
        assert torch.allclose(n2.std_, n1.std_)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestGLMNormalizerStateDict::test_roundtrip -v`
Expected: FAIL — `state_dict` missing.

- [ ] **Step 3: Add `state_dict` and `load_state_dict` to `GLMFeatureNormalizer`**

Modify `src/training/glm_normalizer.py`. Append after `fit_transform`:

```python
    # ------------------------------------------------------------------
    # State-dict interface (composes into FoldBarrier)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return fitted state as a plain dict for ``torch.save``.

        Returns
        -------
        dict
        """
        return {
            "col_start": self.col_start,
            "col_end": self.col_end,
            "fitted": self._fitted,
            "mean": self.mean_.detach().clone() if self.mean_ is not None else None,
            "std": self.std_.detach().clone() if self.std_ is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore fitted state from ``state_dict``.

        Parameters
        ----------
        state : dict
        """
        self.col_start = int(state["col_start"])
        self.col_end = int(state["col_end"])
        self._fitted = bool(state["fitted"])
        self.mean_ = state["mean"]
        self.std_ = state["std"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestGLMNormalizerStateDict -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/glm_normalizer.py tests/test_fold_barrier.py
git commit -m "feat(glm_normalizer): add state_dict/load_state_dict surface

Persisting per-node z-score statistics so they survive in barrier.pt
(ADR-0009). Closes the latent gap where GLM stats were never written
to the fold checkpoint."
```

---

## Task 3: `LabelBuilder.state_dict` / `load_state_dict`

**Files:**
- Modify: `src/datasets/label_builder.py`
- Test: `tests/test_fold_barrier.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fold_barrier.py`:

```python
class TestLabelBuilderStateDict:
    """state_dict carries component normalisation stats + composite op state."""

    def _cfg_single(self):
        from src.configs.label_config import LabelConfig
        return LabelConfig(target="score", id_column="ID")

    def test_single_column_roundtrip(self) -> None:
        import pandas as pd
        from src.datasets.label_builder import LabelBuilder

        cfg = self._cfg_single()
        lb1 = LabelBuilder(cfg)
        components = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
        lb1.fit(components)
        state = lb1.state_dict()

        lb2 = LabelBuilder(cfg)
        lb2.load_state_dict(state)
        np.testing.assert_allclose(
            lb2.transform(components), lb1.transform(components),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestLabelBuilderStateDict -v`
Expected: FAIL — `state_dict` missing.

- [ ] **Step 3: Add `state_dict` and `load_state_dict` to `LabelBuilder`**

Modify `src/datasets/label_builder.py`. Append after `fit_transform`:

```python
    # ------------------------------------------------------------------
    # State-dict interface (composes into FoldBarrier)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return fitted state as a plain dict for ``torch.save``.

        Returns
        -------
        dict
        """
        return {
            "label_names": list(self._label_names),
            "component_means": (
                self._component_means.tolist()
                if self._component_means is not None
                else None
            ),
            "component_stds": (
                self._component_stds.tolist()
                if self._component_stds is not None
                else None
            ),
            "composite_state": (
                self._composite_op.state_dict()
                if self._composite_op is not None
                and hasattr(self._composite_op, "state_dict")
                else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore fitted state from ``state_dict``.

        Parameters
        ----------
        state : dict
        """
        self._label_names = list(state.get("label_names", []))
        cm = state.get("component_means")
        cs = state.get("component_stds")
        self._component_means = (
            np.asarray(cm, dtype=float) if cm is not None else None
        )
        self._component_stds = (
            np.asarray(cs, dtype=float) if cs is not None else None
        )
        cop_state = state.get("composite_state")
        if (
            cop_state is not None
            and self._composite_op is not None
            and hasattr(self._composite_op, "load_state_dict")
        ):
            self._composite_op.load_state_dict(cop_state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestLabelBuilderStateDict -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/datasets/label_builder.py tests/test_fold_barrier.py
git commit -m "feat(label_builder): add state_dict/load_state_dict surface

Carries component-normalisation stats and any stateful composite op
state in a torch-saveable dict, so FoldBarrier can persist composite
labels as part of barrier.pt (ADR-0009)."
```

---

## Task 4: `FoldBarrier` class — `fit` + `transform_labels` + `inverse_transform_labels`

**Files:**
- Create: `src/training/fold_barrier.py`
- Test: `tests/test_fold_barrier.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fold_barrier.py`:

```python
class TestFoldBarrierLabels:
    """Round-trip on label normalisation (non-composite mode)."""

    def _make_graphs(self, n: int = 30, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(n):
            x = torch.tensor(
                rng.normal(size=(n_rois, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
            labels.append(float(rng.normal(loc=2.0, scale=1.5)))
        return graphs, np.array(labels, dtype=np.float64)

    def test_label_roundtrip_non_composite(self) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(seed=1)
        barrier = FoldBarrier(label_norm_strategy="standard")
        barrier.fit(graphs, labels)

        y_norm = barrier.transform_labels(labels)
        assert abs(float(np.mean(y_norm))) < 1e-6
        assert abs(float(np.std(y_norm)) - 1.0) < 1e-6

        y_recovered = barrier.inverse_transform_labels(y_norm)
        np.testing.assert_allclose(y_recovered, labels, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestFoldBarrierLabels -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create `src/training/fold_barrier.py` (label-only surface)**

```python
"""Per-fold leakage-protection coordinator (ADR-0009).

``FoldBarrier`` owns the three per-fold-fit transformers — the
composite ``LabelBuilder`` (when configured), the ``LabelNormalizer``,
and the ``GLMFeatureNormalizer`` (when GLM columns are present) —
as a single fitted bundle for one outer fold's train pool. It does
not know about splits, batch sizes, or DataLoader construction.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric.data

from src.datasets.label_builder import LabelBuilder
from src.training.glm_normalizer import GLMFeatureNormalizer
from src.training.label_normalizer import LabelNormalizer


class FoldBarrier:
    """The per-fold leakage barrier. Fit once on outer-train; transform anything consistent.

    Parameters
    ----------
    label_norm_strategy : Literal["standard", "robust", "minmax", "none"]
        Strategy passed to the inner :class:`LabelNormalizer`.
    glm_col_range : Optional[Tuple[int, int]]
        ``(col_start, col_end)`` of the GLM feature columns in
        ``data.x``. When ``None`` (or when ``glm_normalize`` is
        ``False`` at fit-time), the GLM step is a no-op.
    glm_normalize : bool
        Whether to apply GLM normalisation when ``glm_col_range`` is set.
    label_builder : Optional[LabelBuilder]
        Composite-label builder. When provided, ``fit`` and
        ``transform_labels`` expect a ``pd.DataFrame`` of components
        rather than a 1-D label vector.
    """

    def __init__(
        self,
        *,
        label_norm_strategy: Literal["standard", "robust", "minmax", "none"] = "standard",
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        label_builder: Optional[LabelBuilder] = None,
    ) -> None:
        self._label_norm = LabelNormalizer(strategy=label_norm_strategy)
        self._label_builder = label_builder
        self._glm: Optional[GLMFeatureNormalizer] = None
        if glm_col_range is not None and glm_normalize:
            col_start, col_end = glm_col_range
            self._glm = GLMFeatureNormalizer(col_start, col_end)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_graphs: List[torch_geometric.data.Data],
        train_labels_or_components: Union[np.ndarray, pd.DataFrame],
    ) -> "FoldBarrier":
        """Fit all configured transformers on the outer-train pool.

        Parameters
        ----------
        train_graphs : List[Data]
            Training-split graphs (used to fit the GLM normaliser; not
            mutated).
        train_labels_or_components : np.ndarray | pd.DataFrame
            A 1-D label vector in non-composite mode, a DataFrame slice
            of label components in composite mode.

        Returns
        -------
        FoldBarrier
        """
        if self._label_builder is not None:
            if not isinstance(train_labels_or_components, pd.DataFrame):
                raise TypeError(
                    "Composite mode expects a pd.DataFrame of label components."
                )
            y_train = self._label_builder.fit_transform(train_labels_or_components)
        else:
            y_train = np.asarray(train_labels_or_components, dtype=float)

        self._label_norm.fit(y_train)

        if self._glm is not None:
            self._glm.fit(train_graphs)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform_labels(
        self,
        labels_or_components: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Apply composite construction (if configured) followed by z-scoring.

        Parameters
        ----------
        labels_or_components : np.ndarray | pd.DataFrame

        Returns
        -------
        np.ndarray
        """
        self._require_fitted()
        if self._label_builder is not None:
            if not isinstance(labels_or_components, pd.DataFrame):
                raise TypeError(
                    "Composite mode expects a pd.DataFrame of label components."
                )
            y = self._label_builder.transform(labels_or_components)
        else:
            y = np.asarray(labels_or_components, dtype=float)
        return self._label_norm.transform(y)

    def inverse_transform_labels(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalise predictions back to the original target scale.

        Parameters
        ----------
        y_norm : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self._require_fitted()
        return self._label_norm.inverse_transform(y_norm)

    # Placeholder for graphs — implemented in Task 5.
    def transform_graphs(
        self, graphs: List[torch_geometric.data.Data],
    ) -> List[torch_geometric.data.Data]:
        raise NotImplementedError("Implemented in Task 5.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "FoldBarrier.transform_*/inverse_transform_* called before fit()."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestFoldBarrierLabels -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/fold_barrier.py tests/test_fold_barrier.py
git commit -m "feat(fold_barrier): introduce FoldBarrier with label surface

ADR-0009 step 1: stateful 'transform-anything-consistent-with-my-fit'
object owning LabelBuilder + LabelNormalizer. GLM and graph-level
transform land in the next commit."
```

---

## Task 5: `FoldBarrier.transform_graphs` — clones the GLM slice, never mutates inputs

**Files:**
- Modify: `src/training/fold_barrier.py`
- Test: `tests/test_fold_barrier.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fold_barrier.py`:

```python
class TestFoldBarrierGraphs:
    """transform_graphs returns NEW graphs; original dataset stays untouched."""

    def _make_graphs(self, n: int = 12, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(n):
            x = torch.tensor(
                rng.normal(size=(n_rois, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
            labels.append(float(rng.normal()))
        return graphs, np.array(labels, dtype=np.float64)

    def test_inputs_not_mutated(self) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(seed=2)
        originals = [g.x.clone() for g in graphs]

        barrier = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        barrier.fit(graphs, labels)
        _ = barrier.transform_graphs(graphs)

        for orig, g in zip(originals, graphs):
            assert torch.allclose(g.x, orig), (
                "FoldBarrier.transform_graphs must not mutate input graphs."
            )

    def test_two_fold_sequence_does_not_corrupt_dataset(self) -> None:
        """Two folds in sequence — the second fold must see the original data."""
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(seed=3)
        snapshot = [g.x.clone() for g in graphs]

        for fold in range(2):
            barrier = FoldBarrier(
                label_norm_strategy="standard", glm_col_range=(1, 3),
            )
            barrier.fit(graphs, labels)
            _ = barrier.transform_graphs(graphs)

        for orig, g in zip(snapshot, graphs):
            assert torch.allclose(g.x, orig), (
                "After two folds, original graphs must remain unchanged."
            )

    def test_glm_columns_are_zscored_on_train(self) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make_graphs(n=80, seed=4)
        barrier = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        barrier.fit(graphs, labels)
        transformed = barrier.transform_graphs(graphs)

        stacked = torch.stack([g.x[:, 1:3] for g in transformed], dim=0)
        # mean ≈ 0 per-(node, col) since fit was on the same set
        assert stacked.mean(dim=0).abs().max().item() < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestFoldBarrierGraphs -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `transform_graphs` (replace the placeholder)**

Replace the `transform_graphs` stub in `src/training/fold_barrier.py` with:

```python
    def transform_graphs(
        self,
        graphs: List[torch_geometric.data.Data],
    ) -> List[torch_geometric.data.Data]:
        """Return new graphs with the GLM slice z-scored on the fit pool.

        The returned graphs are shallow copies whose ``x`` tensor is
        replaced by a fresh tensor (cloned slice on the GLM range,
        original tensor elsewhere). The input graphs and the dataset
        they reference are never mutated.

        Parameters
        ----------
        graphs : List[Data]

        Returns
        -------
        List[Data]
        """
        self._require_fitted()
        if self._glm is None:
            # No GLM step configured — still return shallow copies so the
            # caller can attach `.y` without touching the source dataset.
            return [copy.copy(g) for g in graphs]

        col_start, col_end = self._glm.col_start, self._glm.col_end
        out: List[torch_geometric.data.Data] = []
        for g in graphs:
            new_g = copy.copy(g)  # shallow copy → shares tensors with g
            new_x = g.x.clone()   # break the shared-storage tie on x
            new_x[:, col_start:col_end] = (
                (new_x[:, col_start:col_end] - self._glm.mean_) / self._glm.std_
            )
            new_g.x = new_x
            out.append(new_g)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestFoldBarrierGraphs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/fold_barrier.py tests/test_fold_barrier.py
git commit -m "feat(fold_barrier): transform_graphs returns new graphs

Closes the shallow-copy fragility that lived at the
_make_split_loaders call site: x is cloned before the GLM slice is
overwritten, so the source dataset can be transformed twice (inner
+ outer barriers per fold) without corruption. Regression-tested."
```

---

## Task 6: `FoldBarrier.state_dict` / `save` / `load`

**Files:**
- Modify: `src/training/fold_barrier.py`
- Test: `tests/test_fold_barrier.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fold_barrier.py`:

```python
class TestFoldBarrierPersistence:
    """save() / load() round-trip preserves transform behaviour."""

    def _make(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(40):
            x = torch.tensor(
                rng.normal(size=(6, 4)), dtype=torch.float32,
            )
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=6,
            )
            graphs.append(g)
            labels.append(float(rng.normal(loc=1.0, scale=2.0)))
        return graphs, np.array(labels, dtype=np.float64)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = self._make(seed=5)
        b1 = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        b1.fit(graphs, labels)

        b1.save(tmp_path / "barrier.pt")
        assert (tmp_path / "barrier.pt").exists()

        b2 = FoldBarrier.load(
            tmp_path / "barrier.pt",
            label_norm_strategy="standard",
            glm_col_range=(1, 3),
        )

        y_norm_a = b1.transform_labels(labels)
        y_norm_b = b2.transform_labels(labels)
        np.testing.assert_allclose(y_norm_a, y_norm_b)

        ga = b1.transform_graphs(graphs)
        gb = b2.transform_graphs(graphs)
        for x_a, x_b in zip(ga, gb):
            assert torch.allclose(x_a.x, x_b.x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestFoldBarrierPersistence -v`
Expected: FAIL — `save` / `load` missing.

- [ ] **Step 3: Add `state_dict`, `save`, `load` to `FoldBarrier`**

Append to `src/training/fold_barrier.py` (inside the class, before `_require_fitted`):

```python
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Aggregate state of the three transformers.

        Returns
        -------
        dict
        """
        return {
            "fitted": self._fitted,
            "label_norm": self._label_norm.state_dict(),
            "label_builder": (
                self._label_builder.state_dict()
                if self._label_builder is not None else None
            ),
            "glm": self._glm.state_dict() if self._glm is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore aggregated state.

        Parameters
        ----------
        state : dict
        """
        self._fitted = bool(state.get("fitted", False))
        self._label_norm.load_state_dict(state["label_norm"])
        lb_state = state.get("label_builder")
        if lb_state is not None and self._label_builder is not None:
            self._label_builder.load_state_dict(lb_state)
        glm_state = state.get("glm")
        if glm_state is not None:
            if self._glm is None:
                self._glm = GLMFeatureNormalizer(
                    int(glm_state["col_start"]), int(glm_state["col_end"]),
                )
            self._glm.load_state_dict(glm_state)

    def save(self, path: Path) -> None:
        """Persist the barrier's state to ``path`` via ``torch.save``.

        Parameters
        ----------
        path : Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        label_norm_strategy: Literal["standard", "robust", "minmax", "none"] = "standard",
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        label_builder: Optional[LabelBuilder] = None,
    ) -> "FoldBarrier":
        """Reconstruct a barrier from a saved state-dict.

        The transformer instances are reconstructed from the caller-
        supplied configuration; the fitted statistics come from disk.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            label_norm_strategy=label_norm_strategy,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
            label_builder=label_builder,
        )
        obj.load_state_dict(state)
        return obj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py::TestFoldBarrierPersistence -v`
Expected: PASS.

- [ ] **Step 5: Run the whole new test file**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py -v`
Expected: every test PASS.

- [ ] **Step 6: Commit**

```bash
git add src/training/fold_barrier.py tests/test_fold_barrier.py
git commit -m "feat(fold_barrier): persist via torch.save (barrier.pt, no pickle)

ADR-0009 serialisation contract: one typed state-dict per outer
fold, survives class renames, partial-write detection trivial."
```

---

## Task 7: `Trainer` accepts a callable `inverse_transform`

**Files:**
- Modify: `src/training/trainer.py`

- [ ] **Step 1: Replace the `LabelNormalizer` parameter in `fit`, `predict`, and `evaluate`**

Edit `src/training/trainer.py`.

**At the top of the file**, drop the `LabelNormalizer` import:

Old:
```python
from src.training.label_normalizer import LabelNormalizer
```

Replace with:
```python
# (LabelNormalizer no longer imported — Trainer takes a plain callable;
# the barrier is built and owned by the CV path. ADR-0009.)
```

**`fit` signature** (around line 159):

Old:
```python
    def fit(
        self,
        model: BrainGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_normalizer: LabelNormalizer,
        fold_idx: int,
        on_epoch_end_callback: Optional[Callable[[int, dict, bool], None]] = None,
    ) -> TrainResult:
```

Replace with:
```python
    def fit(
        self,
        model: BrainGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        inverse_transform: Callable[[np.ndarray], np.ndarray],
        fold_idx: int,
        on_epoch_end_callback: Optional[Callable[[int, dict, bool], None]] = None,
    ) -> TrainResult:
```

Replace the docstring lines describing `label_normalizer` with:
```
        inverse_transform : Callable[[np.ndarray], np.ndarray]
            Function that maps normalised predictions/targets back to
            the original label scale. Pass
            ``barrier.inverse_transform_labels`` from the CV path.
```

Inside the body, replace the two `label_normalizer.inverse_transform(...)` calls (around lines 236–237):

Old:
```python
                train_preds = label_normalizer.inverse_transform(train_preds_norm)
                train_targets = label_normalizer.inverse_transform(train_targets_norm)
```

Replace with:
```python
                train_preds = inverse_transform(train_preds_norm)
                train_targets = inverse_transform(train_targets_norm)
```

And the validation evaluate call (around line 240):

Old:
```python
                val_metrics = self.evaluate(model, val_loader, label_normalizer, "val")
```

Replace with:
```python
                val_metrics = self.evaluate(model, val_loader, inverse_transform, "val")
```

**`predict` signature** (around line 332):

Old:
```python
    def predict(
        self,
        model: BrainGNN,
        loader: DataLoader,
        label_normalizer: LabelNormalizer,
    ) -> Tuple[np.ndarray, np.ndarray]:
```

Replace with:
```python
    def predict(
        self,
        model: BrainGNN,
        loader: DataLoader,
        inverse_transform: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
```

Replace the docstring line and the two `label_normalizer.inverse_transform(...)` calls inside the body (around lines 366–367):

Old:
```python
        y_pred = label_normalizer.inverse_transform(np.concatenate(all_preds))
        y_true = label_normalizer.inverse_transform(np.concatenate(all_targets))
```

Replace with:
```python
        y_pred = inverse_transform(np.concatenate(all_preds))
        y_true = inverse_transform(np.concatenate(all_targets))
```

**`evaluate` signature** (around line 370):

Old:
```python
    def evaluate(
        self,
        model: BrainGNN,
        loader: DataLoader,
        label_normalizer: LabelNormalizer,
        split: Literal["val", "test"],
    ) -> MetricDict:
```

Replace with:
```python
    def evaluate(
        self,
        model: BrainGNN,
        loader: DataLoader,
        inverse_transform: Callable[[np.ndarray], np.ndarray],
        split: Literal["val", "test"],
    ) -> MetricDict:
```

Body change (around line 395):

Old:
```python
        y_true, y_pred = self.predict(model, loader, label_normalizer)
```

Replace with:
```python
        y_true, y_pred = self.predict(model, loader, inverse_transform)
```

- [ ] **Step 2: Verify by importing the module**

Run: `.venv/bin/python -c "from src.training.trainer import Trainer; print(Trainer.fit.__doc__[:60])"`
Expected: prints a docstring snippet without any `LabelNormalizer` reference; no ImportError.

- [ ] **Step 3: Commit**

```bash
git add src/training/trainer.py
git commit -m "refactor(trainer): take inverse_transform callable, not LabelNormalizer

ADR-0009 T2: the training loop no longer imports the leakage
abstraction. Callers (both CV paths) pass
barrier.inverse_transform_labels."
```

---

## Task 8: `CheckpointManager` reads / writes `barrier.pt` instead of `normalizer.pkl`

**Files:**
- Modify: `src/training/checkpoint_manager.py`

- [ ] **Step 1: Drop the `pickle` + `LabelNormalizer` imports; add `FoldBarrier`**

Edit `src/training/checkpoint_manager.py`. Replace the imports near the top:

Old:
```python
import pickle
...
from src.training.label_normalizer import LabelNormalizer
```

Replace with:
```python
# pickle no longer used — barrier is torch.save'd as a typed state-dict
...
from src.training.fold_barrier import FoldBarrier
```

(Keep the other imports unchanged. If `pickle` is not used elsewhere in the file, remove the line entirely.)

- [ ] **Step 2: Update `FoldCheckpoint` dataclass — `normalizer` → `barrier`**

Replace the `normalizer: Optional[LabelNormalizer] = None` line in the `FoldCheckpoint` dataclass (around line 54):

Old:
```python
    normalizer : LabelNormalizer
        ...
    normalizer: Optional[LabelNormalizer] = None
```

Replace with:
```python
    barrier : FoldBarrier
        Outer-train-fit leakage barrier — replaces normalizer.pkl (ADR-0009).
    barrier: Optional[FoldBarrier] = None
```

- [ ] **Step 3: Update `save_fold_checkpoint` signature and body**

Old (line 77 onwards):
```python
    def save_fold_checkpoint(
        self,
        fold_idx: int,
        best_model_state_dict: dict,
        last_model_state_dict: dict,
        label_normalizer: LabelNormalizer,
        best_metrics: MetricDict,
        ...
    ) -> Path:
        ...
        torch.save(best_model_state_dict, fold_dir / "model_best.pt")
        torch.save(last_model_state_dict, fold_dir / "model_last.pt")

        with open(fold_dir / "normalizer.pkl", "wb") as f:
            pickle.dump(label_normalizer, f)
```

Replace with:
```python
    def save_fold_checkpoint(
        self,
        fold_idx: int,
        best_model_state_dict: dict,
        last_model_state_dict: dict,
        barrier: FoldBarrier,
        best_metrics: MetricDict,
        ...
    ) -> Path:
        ...
        torch.save(best_model_state_dict, fold_dir / "model_best.pt")
        torch.save(last_model_state_dict, fold_dir / "model_last.pt")

        barrier.save(fold_dir / "barrier.pt")
```

(Keep the rest of the method — `metrics.json`, `model_config.json`, `feature_config.json` — unchanged. Update the docstring parameter line from `label_normalizer : LabelNormalizer` to `barrier : FoldBarrier`.)

- [ ] **Step 4: Update `load_fold_checkpoint` body**

Old:
```python
        with open(fold_dir / "normalizer.pkl", "rb") as f:
            normalizer = pickle.load(f)
        ...
        return FoldCheckpoint(
            fold_idx=fold_idx,
            best_model_state_dict=best_state_dict,
            last_model_state_dict=last_state_dict,
            normalizer=normalizer,
            best_metrics=metrics_data["best"]["metrics"],
            ...
        )
```

Replace with:
```python
        barrier_path = fold_dir / "barrier.pt"
        barrier: Optional[FoldBarrier]
        if barrier_path.exists():
            # The barrier needs configuration (norm strategy, GLM range,
            # composite mode) to know which transformer instances to
            # reconstruct. Read those from the model/feature configs
            # at the call site instead; for the bare reload path,
            # default to the standard non-composite shape and load
            # whatever statistics survived. The Explainer's reload
            # migration (PR2) will pass explicit configuration.
            barrier = FoldBarrier(label_norm_strategy="standard")
            barrier.load_state_dict(
                torch.load(barrier_path, map_location="cpu", weights_only=False)
            )
        else:
            barrier = None
        ...
        return FoldCheckpoint(
            fold_idx=fold_idx,
            best_model_state_dict=best_state_dict,
            last_model_state_dict=last_state_dict,
            barrier=barrier,
            best_metrics=metrics_data["best"]["metrics"],
            ...
        )
```

- [ ] **Step 5: Update `load_model_for_fold` body**

Old:
```python
        with open(fold_dir / "normalizer.pkl", "rb") as f:
            normalizer = pickle.load(f)

        return model, normalizer
```

Replace with:
```python
        barrier_path = fold_dir / "barrier.pt"
        if barrier_path.exists():
            barrier = FoldBarrier(label_norm_strategy="standard")
            barrier.load_state_dict(
                torch.load(barrier_path, map_location="cpu", weights_only=False)
            )
        else:
            barrier = None

        return model, barrier
```

(Also update the return type annotation from `Tuple[BrainGNN, LabelNormalizer]` to `Tuple[BrainGNN, Optional[FoldBarrier]]`.)

- [ ] **Step 6: Smoke import**

Run: `.venv/bin/python -c "from src.training.checkpoint_manager import CheckpointManager, FoldCheckpoint; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add src/training/checkpoint_manager.py
git commit -m "feat(checkpoint_manager): persist FoldBarrier as barrier.pt

Replaces normalizer.pkl in the fold checkpoint layout (ADR-0009).
B-fresh — old checkpoints with only normalizer.pkl no longer
reload as full FoldCheckpoints; model weights remain loadable."
```

---

## Task 9: Migrate `NestedCrossValidator._make_split_loaders` to use `FoldBarrier`

**Files:**
- Modify: `src/training/nested_cross_validation.py`

- [ ] **Step 1: Rewrite `_make_split_loaders`**

Replace lines 702–764 of `src/training/nested_cross_validation.py`:

Old:
```python
    def _make_split_loaders(
        self,
        *,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
        label_builder: Optional[LabelBuilder],
        label_components: Optional[pd.DataFrame],
        glm_col_range: Optional[Tuple[int, int]],
        glm_normalize: bool,
    ) -> Tuple[Dict[str, DataLoader], LabelNormalizer]:
        """Build train/val/test DataLoaders with normaliser fit on train only.

        Set ``val_idx=[]`` for the outer refit phase — the returned dict will
        omit the ``"val"`` key.
        """
        if label_builder is not None and label_components is not None:
            y_train = label_builder.fit_transform(label_components.iloc[train_idx])
            y_test = label_builder.transform(label_components.iloc[test_idx])
            y_val = (
                label_builder.transform(label_components.iloc[val_idx])
                if val_idx else np.zeros(0)
            )
        else:
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            y_val = labels[val_idx] if val_idx else np.zeros(0)

        normalizer = LabelNormalizer(strategy=self.cfg.label_norm_strategy)
        y_train_norm = normalizer.fit_transform(y_train)
        y_test_norm = normalizer.transform(y_test)
        y_val_norm = normalizer.transform(y_val) if val_idx else np.zeros(0)

        bs = self.cfg.batch_size
        loaders: Dict[str, DataLoader] = {
            "train": _make_loader(
                dataset, train_idx, y_train_norm, shuffle=True,
                batch_size=bs, drop_last=len(train_idx) > bs,
            ),
            "test": _make_loader(
                dataset, test_idx, y_test_norm, shuffle=False,
                batch_size=bs, drop_last=False,
            ),
        }
        if val_idx:
            loaders["val"] = _make_loader(
                dataset, val_idx, y_val_norm, shuffle=False,
                batch_size=bs, drop_last=False,
            )

        if glm_col_range is not None and glm_normalize:
            col_start, col_end = glm_col_range
            glm_norm = GLMFeatureNormalizer(col_start, col_end)
            train_graphs = list(loaders["train"].dataset)
            glm_norm.fit(train_graphs)
            glm_norm.transform(train_graphs)
            glm_norm.transform(list(loaders["test"].dataset))
            if "val" in loaders:
                glm_norm.transform(list(loaders["val"].dataset))

        return loaders, normalizer
```

Replace with:
```python
    def _make_split_loaders(
        self,
        *,
        dataset: List[torch_geometric.data.Data],
        labels: np.ndarray,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
        label_builder: Optional[LabelBuilder],
        label_components: Optional[pd.DataFrame],
        glm_col_range: Optional[Tuple[int, int]],
        glm_normalize: bool,
    ) -> Tuple[Dict[str, DataLoader], FoldBarrier]:
        """Build train/val/test DataLoaders behind a FoldBarrier (ADR-0009).

        Set ``val_idx=[]`` for the outer refit phase — the returned dict
        omits the ``"val"`` key.
        """
        barrier = FoldBarrier(
            label_norm_strategy=self.cfg.label_norm_strategy,
            glm_col_range=glm_col_range,
            glm_normalize=glm_normalize,
            label_builder=label_builder,
        )

        train_graphs_raw = [dataset[i] for i in train_idx]
        if label_builder is not None and label_components is not None:
            barrier.fit(train_graphs_raw, label_components.iloc[train_idx])
            y_train_norm = barrier.transform_labels(label_components.iloc[train_idx])
            y_test_norm = barrier.transform_labels(label_components.iloc[test_idx])
            y_val_norm = (
                barrier.transform_labels(label_components.iloc[val_idx])
                if val_idx else np.zeros(0)
            )
        else:
            barrier.fit(train_graphs_raw, labels[train_idx])
            y_train_norm = barrier.transform_labels(labels[train_idx])
            y_test_norm = barrier.transform_labels(labels[test_idx])
            y_val_norm = (
                barrier.transform_labels(labels[val_idx])
                if val_idx else np.zeros(0)
            )

        train_graphs = barrier.transform_graphs(train_graphs_raw)
        test_graphs = barrier.transform_graphs([dataset[i] for i in test_idx])
        val_graphs = (
            barrier.transform_graphs([dataset[i] for i in val_idx])
            if val_idx else []
        )

        bs = self.cfg.batch_size
        loaders: Dict[str, DataLoader] = {
            "train": _make_loader_from_graphs(
                train_graphs, y_train_norm, shuffle=True,
                batch_size=bs, drop_last=len(train_idx) > bs,
            ),
            "test": _make_loader_from_graphs(
                test_graphs, y_test_norm, shuffle=False,
                batch_size=bs, drop_last=False,
            ),
        }
        if val_idx:
            loaders["val"] = _make_loader_from_graphs(
                val_graphs, y_val_norm, shuffle=False,
                batch_size=bs, drop_last=False,
            )

        return loaders, barrier
```

- [ ] **Step 2: Add a new graph-list helper alongside the existing `_make_loader`**

The current `_make_loader` (around line 791) takes `(dataset, indices, y_values, ...)` and slices the dataset internally. The barrier now returns ready-to-use graphs, so add a sibling helper that takes graphs directly. Insert immediately above the existing `_make_loader`:

```python
def _make_loader_from_graphs(
    graphs: List[torch_geometric.data.Data],
    y_values: np.ndarray,
    *,
    shuffle: bool,
    batch_size: int,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap pre-transformed graphs in a PyG DataLoader, attaching labels.

    The graphs must have been produced by ``FoldBarrier.transform_graphs``
    (or otherwise be safe to mutate in ``.y``).
    """
    for i, g in enumerate(graphs):
        g.y = torch.tensor([y_values[i]], dtype=torch.float32)
    return DataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
    )
```

(The existing `_make_loader` may remain in place for now; it has other callers inside the file. We add the new helper rather than rewriting the old one.)

- [ ] **Step 3: Rename `outer_normalizer` → `outer_barrier` and update the outer-fold trainer call**

Grep first: `grep -n "outer_normalizer" src/training/nested_cross_validation.py` — at the time of writing there are two hits (the tuple unpack at line ~431, the predict call at line ~476, and the `save_fold_checkpoint` kwarg at line ~502 covered in step 4).

Replace every `outer_normalizer` token in `src/training/nested_cross_validation.py` with `outer_barrier`. The unpack site becomes:

```python
        outer_loaders, outer_barrier = self._make_split_loaders(...)
```

The predict call (around line 475–477) becomes:

```python
        y_true, y_pred = eval_trainer.predict(
            model, outer_loaders["test"], outer_barrier.inverse_transform_labels,
        )
```

- [ ] **Step 4: Update `save_fold_checkpoint` call (line ~498)**

Old:
```python
        ckpt_mgr.save_fold_checkpoint(
            fold_idx=fold,
            best_model_state_dict=model.state_dict(),
            last_model_state_dict=model.state_dict(),
            label_normalizer=outer_normalizer,
            ...
        )
```

Replace with:
```python
        ckpt_mgr.save_fold_checkpoint(
            fold_idx=fold,
            best_model_state_dict=model.state_dict(),
            last_model_state_dict=model.state_dict(),
            barrier=outer_barrier,
            ...
        )
```

- [ ] **Step 5: Rename `inner_normalizer` → `inner_barrier` in the two inner-HPO helpers**

There are two methods that take `inner_normalizer` as a parameter:

- `_run_inner_hpo(..., inner_normalizer: LabelNormalizer, ...)` (around line 549)
- `_train_inner_trial(..., inner_normalizer: LabelNormalizer, ...)` (around line 632)

For both methods:
1. Change the parameter name to `inner_barrier` and type to `FoldBarrier`.
2. Update every caller in this file (around lines 382, 406, 422, 588) — both the unpack site (`inner_loaders, inner_barrier = self._make_split_loaders(...)`) and the keyword-argument call sites (`inner_barrier=inner_barrier,`).

Inside `_train_inner_trial`, the `trainer.fit` call (around line 645) currently passes `inner_normalizer` positionally. Replace:

Old:
```python
        result = trainer.fit(
            model,
            inner_loaders["train"],
            inner_loaders["val"],
            inner_normalizer,
            fold_idx=fold,
            on_epoch_end_callback=None,
        )
```

Replace with:
```python
        result = trainer.fit(
            model,
            inner_loaders["train"],
            inner_loaders["val"],
            inner_barrier.inverse_transform_labels,
            fold_idx=fold,
            on_epoch_end_callback=None,
        )
```

After this step, the only remaining occurrence of the string `normalizer` in this file should be inside docstrings (if any) — `_refit_on_trainval` never took a normalizer in the first place, so it needs no change.

Verify with: `grep -n "normalizer" src/training/nested_cross_validation.py`
Expected: empty (or only matches inside docstrings — clean those up if present).

- [ ] **Step 6: Drop the now-unused imports**

At the top of the file, remove:
```python
from src.training.label_normalizer import LabelNormalizer
from src.training.glm_normalizer import GLMFeatureNormalizer
```

Add:
```python
from src.training.fold_barrier import FoldBarrier
```

(Keep `from src.datasets.label_builder import LabelBuilder` — still used in `_make_split_loaders`'s parameter type.)

- [ ] **Step 7: Verify import**

Run: `.venv/bin/python -c "from src.training.nested_cross_validation import NestedCrossValidator; print('ok')"`
Expected: `ok`.

- [ ] **Step 8: Commit**

```bash
git add src/training/nested_cross_validation.py
git commit -m "refactor(nested_cv): build a FoldBarrier instead of the 5-step dance

_make_split_loaders constructs one FoldBarrier per (rep, fold)
phase, calls it for labels + graphs, returns (loaders, barrier).
Outer-fold path persists outer_barrier via barrier.pt (ADR-0009)."
```

---

## Task 10: Migrate `CrossValidator.run` (legacy CV) to use `FoldBarrier`

**Files:**
- Modify: `src/training/cross_validation.py`

- [ ] **Step 1: Replace the per-fold dance (lines 236–303)**

Edit `src/training/cross_validation.py`. Replace the block from the comment `# Build per-fold labels` through the GLM normalisation block:

Old:
```python
            # ------------------------------------------------------------------
            # Build per-fold labels
            # ------------------------------------------------------------------
            if label_builder is not None:
                y_train = label_builder.fit_transform(
                    label_components.iloc[train_idx]
                )
                y_val = label_builder.transform(label_components.iloc[val_idx])
                y_test = label_builder.transform(label_components.iloc[test_idx])
            else:
                y_train = labels[train_idx]
                y_val = labels[val_idx]
                y_test = labels[test_idx]

            # ------------------------------------------------------------------
            # Normalise labels (fit on train only)
            # ------------------------------------------------------------------
            normalizer = LabelNormalizer(strategy=self.cfg.label_norm_strategy)
            y_train_norm = normalizer.fit_transform(y_train)
            y_val_norm = normalizer.transform(y_val)
            y_test_norm = normalizer.transform(y_test)

            # ------------------------------------------------------------------
            # Build DataLoaders with labels attached to each graph
            # ------------------------------------------------------------------
            bs = self.cfg.batch_size

            def _make_loader(
                indices: List[int],
                y_values: np.ndarray,
                shuffle: bool,
                drop_last: bool = False,
            ) -> DataLoader:
                graphs = []
                for i, idx in enumerate(indices):
                    g = copy.copy(dataset[idx])
                    g.y = torch.tensor([y_values[i]], dtype=torch.float32)
                    graphs.append(g)
                return DataLoader(
                    graphs,
                    batch_size=bs,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )

            train_loader = _make_loader(
                train_idx, y_train_norm, shuffle=True,
                drop_last=len(train_idx) > bs,
            )
            val_loader = _make_loader(val_idx, y_val_norm, shuffle=False)
            test_loader = _make_loader(test_idx, y_test_norm, shuffle=False)

            # ------------------------------------------------------------------
            # GLM feature normalisation (fit on train only)
            # ------------------------------------------------------------------
            if glm_col_range is not None and glm_normalize:
                col_start, col_end = glm_col_range
                glm_norm = GLMFeatureNormalizer(col_start, col_end)
                # Collect the train graphs from the loader to fit stats
                train_graphs = [g for g in train_loader.dataset]
                glm_norm.fit(train_graphs)
                glm_norm.transform(train_graphs)
                # Apply same stats to val and test graphs
                val_graphs = [g for g in val_loader.dataset]
                glm_norm.transform(val_graphs)
                test_graphs = [g for g in test_loader.dataset]
                glm_norm.transform(test_graphs)
```

Replace with:
```python
            # ------------------------------------------------------------------
            # Per-fold leakage barrier (ADR-0009): one fitted bundle of
            # composite-label + label-norm + GLM transformers, fit on train.
            # ------------------------------------------------------------------
            barrier = FoldBarrier(
                label_norm_strategy=self.cfg.label_norm_strategy,
                glm_col_range=glm_col_range,
                glm_normalize=glm_normalize,
                label_builder=label_builder,
            )

            train_graphs_raw = [dataset[i] for i in train_idx]
            if label_builder is not None and label_components is not None:
                barrier.fit(train_graphs_raw, label_components.iloc[train_idx])
                y_train_norm = barrier.transform_labels(label_components.iloc[train_idx])
                y_val_norm = barrier.transform_labels(label_components.iloc[val_idx])
                y_test_norm = barrier.transform_labels(label_components.iloc[test_idx])
            else:
                barrier.fit(train_graphs_raw, labels[train_idx])
                y_train_norm = barrier.transform_labels(labels[train_idx])
                y_val_norm = barrier.transform_labels(labels[val_idx])
                y_test_norm = barrier.transform_labels(labels[test_idx])

            train_graphs = barrier.transform_graphs(train_graphs_raw)
            val_graphs = barrier.transform_graphs([dataset[i] for i in val_idx])
            test_graphs = barrier.transform_graphs([dataset[i] for i in test_idx])

            bs = self.cfg.batch_size

            def _make_loader(
                graphs_list: List[torch_geometric.data.Data],
                y_values: np.ndarray,
                shuffle: bool,
                drop_last: bool = False,
            ) -> DataLoader:
                for i, g in enumerate(graphs_list):
                    g.y = torch.tensor([y_values[i]], dtype=torch.float32)
                return DataLoader(
                    graphs_list,
                    batch_size=bs,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )

            train_loader = _make_loader(
                train_graphs, y_train_norm, shuffle=True,
                drop_last=len(train_idx) > bs,
            )
            val_loader = _make_loader(val_graphs, y_val_norm, shuffle=False)
            test_loader = _make_loader(test_graphs, y_test_norm, shuffle=False)
```

- [ ] **Step 2: Update the Trainer calls and `save_fold_checkpoint` call**

The lines around 330–341 currently pass `normalizer` to the Trainer; the line around 391 passes it to `save_fold_checkpoint`. Make the following replacements:

Old:
```python
            result = trainer.fit(
                model, train_loader, val_loader, normalizer, fold_idx,
                on_epoch_end_callback=epoch_end_callback,
            )
            ...
            y_true_fold, y_pred_fold = trainer.predict(model, test_loader, normalizer)
            ...
            test_metrics = trainer.evaluate(model, test_loader, normalizer, "test")
            ...
                checkpoint_manager.save_fold_checkpoint(
                    fold_idx,
                    result.best_model_state_dict,
                    result.last_model_state_dict,
                    normalizer,
                    result.best_val_metrics,
                    ...
                )
```

Replace with:
```python
            inv = barrier.inverse_transform_labels
            result = trainer.fit(
                model, train_loader, val_loader, inv, fold_idx,
                on_epoch_end_callback=epoch_end_callback,
            )
            ...
            y_true_fold, y_pred_fold = trainer.predict(model, test_loader, inv)
            ...
            test_metrics = trainer.evaluate(model, test_loader, inv, "test")
            ...
                checkpoint_manager.save_fold_checkpoint(
                    fold_idx,
                    result.best_model_state_dict,
                    result.last_model_state_dict,
                    barrier,
                    result.best_val_metrics,
                    ...
                )
```

- [ ] **Step 3: Drop the unused imports**

At the top of `src/training/cross_validation.py`, remove:
```python
import copy
...
from src.training.glm_normalizer import GLMFeatureNormalizer
from src.training.label_normalizer import LabelNormalizer
```

Add:
```python
from src.training.fold_barrier import FoldBarrier
```

(Verify `copy` is not used elsewhere in the file; if it is, leave the `import copy` line.)

- [ ] **Step 4: Verify import**

Run: `.venv/bin/python -c "from src.training.cross_validation import CrossValidator; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/training/cross_validation.py
git commit -m "refactor(cross_validation): legacy CV consumes FoldBarrier too

Drops the duplicated five-step leakage dance — same barrier
abstraction as NestedCrossValidator (ADR-0009)."
```

---

## Task 11: Drop dead `LabelNormalizer` import from `Finetuner`

**Files:**
- Modify: `src/finetuning/finetuner.py`

- [ ] **Step 1: Remove the unused import (line 23)**

Old:
```python
from src.training.label_normalizer import LabelNormalizer
```

Delete the line outright (no remaining references — confirmed by grep).

- [ ] **Step 2: Smoke import**

Run: `.venv/bin/python -c "from src.finetuning.finetuner import Finetuner; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/finetuning/finetuner.py
git commit -m "chore(finetuner): drop unused LabelNormalizer import

Finetuner only delegates per-fold leakage to CrossValidator, which
now constructs a FoldBarrier internally (ADR-0009). The local
import was already dead."
```

---

## Task 12: Update `CONTEXT.md` checkpoint layout

**Files:**
- Modify: `CONTEXT.md`

- [ ] **Step 1: Replace `normalizer.pkl` in the layout diagram**

Edit `CONTEXT.md`. In the `## Checkpoint layout` section (around line 232):

Old:
```
        ├── normalizer.pkl          # outer-TrainVal-fit LabelNormalizer (pickle)
```

Replace with:
```
        ├── barrier.pt              # outer-TrainVal-fit FoldBarrier state-dict (torch.save, ADR-0009)
```

- [ ] **Step 2: Update the `Checkpoint` glossary entry**

Around line 77, the entry mentions `LabelNormalizer` / `GLMFeatureNormalizer` / `LabelBuilder` by name. Replace that fragment:

Old:
```
the fitted per-fold leakage state (`LabelNormalizer` and, where applicable, `GLMFeatureNormalizer` and composite-label `LabelBuilder` state)
```

Replace with:
```
the fitted per-fold leakage state inside `barrier.pt` (see [Fold barrier](#glossary))
```

- [ ] **Step 3: Commit**

```bash
git add CONTEXT.md
git commit -m "docs(context): fold checkpoint stores barrier.pt, not normalizer.pkl

Aligns the layout diagram and the Checkpoint glossary entry with
ADR-0009. The Fold barrier glossary entry was already in place."
```

---

## Task 13: Full test sweep + verify no `normalizer.pkl` writers remain

**Files:** none (verification)

- [ ] **Step 1: Run the new FoldBarrier tests**

Run: `.venv/bin/python -m pytest tests/test_fold_barrier.py -v`
Expected: every test PASS.

- [ ] **Step 2: Verify no code path still writes `normalizer.pkl`**

Run: `grep -rn "normalizer.pkl" src/ tests/`
Expected: zero matches.

- [ ] **Step 3: Verify no code path still imports `LabelNormalizer` outside `label_normalizer.py` and `fold_barrier.py`**

Run: `grep -rn "from src.training.label_normalizer import\|import label_normalizer" src/`
Expected: only `src/training/fold_barrier.py` matches.

- [ ] **Step 4: Verify the Trainer no longer references `LabelNormalizer`**

Run: `grep -n "LabelNormalizer\|label_normalizer" src/training/trainer.py`
Expected: zero matches.

- [ ] **Step 5: Smoke-import every modified module**

Run:
```bash
.venv/bin/python -c "
from src.training.label_normalizer import LabelNormalizer
from src.training.glm_normalizer import GLMFeatureNormalizer
from src.datasets.label_builder import LabelBuilder
from src.training.fold_barrier import FoldBarrier
from src.training.checkpoint_manager import CheckpointManager, FoldCheckpoint
from src.training.cross_validation import CrossValidator
print('all imports ok')
"
```

(`Trainer` and `NestedCrossValidator` import `src.models.braingnn_model`, which requires `torch_sparse` and is unavailable in this local venv. Confirm with `git diff main -- src/training/trainer.py src/training/nested_cross_validation.py` that the only diffs are the planned ones.)

Expected: `all imports ok`.

- [ ] **Step 6: If the existing `tests/test_training.py::TestNestedCrossValidator` suite can be collected (i.e. `torch_sparse` is installed), run it**

Run: `.venv/bin/python -m pytest tests/test_training.py::TestNestedCrossValidator -v 2>&1 | tail -20`
- If collection fails on `torch_sparse`, the suite is pre-existing-broken locally — note in the PR description that this gate must be re-run on the cluster.
- If it runs, every test should PASS (numerical equivalence is preserved by construction: same normalisation strategies, same fit pool, same z-scores; only the orchestration changed).

- [ ] **Step 7: No commit (verification only).**

---

## Task 14: Push branch and open PR

**Files:** none

- [ ] **Step 1: Push**

```bash
git push -u origin feature/fold-barrier
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --base main --head feature/fold-barrier \
  --title "feat(training): introduce FoldBarrier (ADR-0009 PR1)" \
  --body "$(cat <<'EOF'
## Summary

- New module `src/training/fold_barrier.py` coordinates the three per-fold leakage protections (`LabelBuilder` + `LabelNormalizer` + `GLMFeatureNormalizer`) as one fitted bundle per outer fold.
- Both CV paths (`NestedCrossValidator._make_split_loaders` and legacy `CrossValidator.run`) drop the duplicated five-step leakage dance and consume `FoldBarrier`.
- `Trainer.fit` / `predict` / `evaluate` now take a plain `Callable[[np.ndarray], np.ndarray]` for inverse-transforming predictions — the training loop no longer imports the leakage abstraction (ADR-0009 T2).
- The fold checkpoint layout swaps `normalizer.pkl` for `barrier.pt` (`torch.save` of a typed state-dict, no pickle). **B-fresh** — old `normalizer.pkl` checkpoints are not migrated; their model weights remain loadable, but a full reload requires re-running.
- `transform_graphs` returns new graphs (clones the GLM slice). Regression-tested: two folds in sequence leave the source dataset untouched.

Out of scope (PR2): `CheckpointManager → FoldCheckpoint` rename; Explainer's full-reload migration.

## Test plan

- [ ] `tests/test_fold_barrier.py` — round-trip + non-mutation + save/load all green locally.
- [ ] `tests/test_training.py::TestNestedCrossValidator` smoke suite green on the cluster (the `torch_sparse` import gate is not satisfied in the local venv).
- [ ] No remaining writers of `normalizer.pkl`: `grep -r "normalizer.pkl" src/ tests/` is empty.
- [ ] No remaining `LabelNormalizer` imports outside the module itself and `FoldBarrier`.
- [ ] Manual smoke run of one nested-CV fold on a small dataset to confirm numeric equivalence with `main`.

ADR: [`docs/adr/0009-fold-barrier.md`](docs/adr/0009-fold-barrier.md)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Report PR URL to user.**
