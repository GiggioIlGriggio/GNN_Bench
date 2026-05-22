# FoldCheckpoint + Explainer barrier-reload (PR2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `CheckpointManager → FoldCheckpoint` (and expand its scope to be *the* per-fold read path) so consumers reload a fold via one object instead of reaching into individual files. Migrate the Explainer's GLM re-fit branch (PR1 dead-letter: `barrier.pt` is written but unused) to consume the persisted barrier directly. Scope confirmed by [ADR-0009](../../adr/0009-fold-barrier.md) consequences and the PR1→PR2 handoff.

**Architecture:**
- A new per-fold view class `FoldCheckpoint(fold_dir)` owns the single read path: `load_state_dict`, `load_model`, `load_barrier`, `load_metrics`, `load_bundle`. `load_barrier` autoconfigures from the persisted state alone — the GLM substate carries `(col_start, col_end)` and the label-norm substate carries its `strategy`, so callers don't need to know the original fit-time config (composite-label reload remains a known gap, documented).
- The pre-PR2 dataclass `FoldCheckpoint` is renamed to `FoldBundle` (return type of `load_bundle()`), freeing the name for the new view class.
- `CheckpointManager` stays for *write-side* and multi-fold operations only: `save_fold_checkpoint`, epoch-snapshot helpers, `get_common_epochs`. Both classes live in `src/training/fold_checkpoint.py` (the module is renamed alongside the central class). `src/training/checkpoint_manager.py` is removed; no compat shim (B-fresh, per ADR-0009).
- Explainer's `_explain_one_fold` replaces its `GLMFeatureNormalizer.fit(...).transform(...)` block with `FoldCheckpoint(fold_ckpt_dir).load_barrier()` + `barrier.transform_graphs(test_graphs)`. Finetuner's `_load_weights_for_fold` switches to `FoldCheckpoint.for_fold(...).load_state_dict(variant=...)`. The test fixture in `tests/test_gnn_explainer.py` is updated to write a real `barrier.pt`; a new regression test verifies GLM stats survive the reload (instead of being silently re-fit).

**Tech Stack:** Python 3.10, PyTorch + PyG, pandas, numpy. Existing `src/training/` and `src/gnn_explainer/` modules.

**Out of scope:**
- Persisting composite-mode `LabelConfig` alongside `barrier.pt` (the barrier reload still can't round-trip composite labels — only graphs and z-scored labels). Documented as a known gap; not a PR2 blocker because the Explainer never touches labels.
- A migration shim for old `barrier.pt`-less checkpoints (B-fresh, per ADR-0009).
- Renaming `CheckpointManager` further (e.g. → `FoldCheckpointStore`) — its surface shrinks but the class name stays, since renaming it would touch every write-side call site for no semantic gain.

---

## File Structure

**Created**
- `src/training/fold_checkpoint.py` — new home. Contains `FoldBundle` (renamed dataclass), `FoldCheckpoint` (per-fold view class, new), `CheckpointManager` (write-side + epoch ops, slimmed down).
- `tests/test_fold_checkpoint.py` — unit tests for the new `FoldCheckpoint` class.

**Deleted**
- `src/training/checkpoint_manager.py` — moved to `fold_checkpoint.py`; no shim.

**Modified**
- `src/training/__init__.py` — re-export from the new module; `FoldCheckpoint` now refers to the new class (same exported name).
- `src/training/nested_cross_validation.py` — update one import line (`from src.training.checkpoint_manager` → `from src.training.fold_checkpoint`).
- `src/training/cross_validation.py` — same one-import update.
- `src/finetuning/finetuner.py` — `_load_weights_for_fold` switches from `CheckpointManager(...).load_fold_checkpoint(fold_idx).best_model_state_dict` to `FoldCheckpoint.for_fold(...).load_state_dict(variant=...)`; the other `CheckpointManager(epoch_checkpoint_dir).get_common_epochs(...)` call is unaffected (write-side stays).
- `src/gnn_explainer/explainer.py` — `_explain_one_fold` replaces the `GLMFeatureNormalizer` re-fit block with `FoldCheckpoint(fold_ckpt_dir).load_barrier()` + `barrier.transform_graphs(test_graphs)`. Drops the `from src.training.glm_normalizer import GLMFeatureNormalizer` import and the now-unused `CheckpointManager` import.
- `tests/test_gnn_explainer.py` — `_write_fold_artifacts` writes a valid `barrier.pt` (fitted on the synthetic graphs); a new test class `TestExplainerConsumesPersistedBarrier` verifies the explainer reads GLM stats from disk rather than re-fitting.
- `tests/test_training.py` — update one import line (`from src.training.checkpoint_manager` → `from src.training.fold_checkpoint`).
- `CONTEXT.md` — file map line for `checkpoint_manager.py` becomes `fold_checkpoint.py`; the `Checkpoint` glossary entry tightens its class reference; the layout diagram is already accurate (PR1 updated `normalizer.pkl` → `barrier.pt`).

**Untouched**
- `src/training/fold_barrier.py` — `FoldBarrier.load`'s caller-supplies-config design is fine. The new `FoldCheckpoint.load_barrier` calls it with sensible defaults.
- `src/training/trainer.py`, `src/training/label_normalizer.py`, `src/training/glm_normalizer.py`, `src/datasets/label_builder.py` — no changes.
- `src/sweeps/hydra_sweep.py` — no `CheckpointManager` use here.

---

## Task 1: Rename module file + rename dataclass `FoldCheckpoint → FoldBundle`

Smallest mechanical move first. This task makes the file rename a no-op semantically — every caller continues to work via the same exported names from `src.training` — but renames the *dataclass* so the next task can introduce the new `FoldCheckpoint` view class without a name collision.

**Files:**
- Move: `src/training/checkpoint_manager.py` → `src/training/fold_checkpoint.py`
- Modify: `src/training/__init__.py`
- Modify: `src/training/nested_cross_validation.py:47`
- Modify: `src/training/cross_validation.py:19`
- Modify: `src/finetuning/finetuner.py:107`, `:392`
- Modify: `src/gnn_explainer/explainer.py:522`
- Modify: `tests/test_training.py:22`
- Modify (inside the moved file): rename class `FoldCheckpoint` → `FoldBundle`; rename field `barrier: Optional[FoldBarrier]` keeps its name; update docstring header; update `load_fold_checkpoint`'s return-type annotation + the `return FoldCheckpoint(...)` call site.

- [ ] **Step 1: Move the file with `git mv`**

```bash
git mv src/training/checkpoint_manager.py src/training/fold_checkpoint.py
```

Verify:

```bash
ls src/training/fold_checkpoint.py
test ! -e src/training/checkpoint_manager.py && echo "old path gone"
```

Expected: file exists at new path; old path is gone.

- [ ] **Step 2: Inside the moved file, rename the dataclass `FoldCheckpoint → FoldBundle`**

Edit `src/training/fold_checkpoint.py`.

Find (around line 26–57):

```python
@dataclass
class FoldCheckpoint:
    """Contents of a saved fold checkpoint.

    Attributes
    ----------
    fold_idx : int
        Fold index.
    best_model_state_dict : dict
        PyTorch model state dict from the best epoch.
    last_model_state_dict : dict
        PyTorch model state dict from the final epoch.
    barrier : FoldBarrier
        Outer-train-fit leakage barrier (ADR-0009).
    best_metrics : MetricDict
        Validation metrics at the best epoch.
    best_epoch : int
        Epoch index of the best checkpoint.
    last_metrics : MetricDict
        Validation metrics at the last epoch.
    last_epoch : int
        Epoch index of the last checkpoint.
    """

    fold_idx: int = 0
    best_model_state_dict: dict = None  # type: ignore[assignment]
    last_model_state_dict: dict = None  # type: ignore[assignment]
    barrier: Optional[FoldBarrier] = None
    best_metrics: MetricDict = None  # type: ignore[assignment]
    best_epoch: int = 0
    last_metrics: MetricDict = None  # type: ignore[assignment]
    last_epoch: int = 0
```

Replace with:

```python
@dataclass
class FoldBundle:
    """Snapshot of one fold's persisted artifacts, loaded together.

    Returned by ``FoldCheckpoint.load_bundle()``. Use the typed
    accessors on :class:`FoldCheckpoint` when you only need part of
    the bundle (model weights, the barrier, metrics).

    Attributes
    ----------
    fold_idx : int
        Fold index.
    best_model_state_dict : dict
        PyTorch model state dict from the best epoch.
    last_model_state_dict : dict
        PyTorch model state dict from the final epoch.
    barrier : Optional[FoldBarrier]
        Outer-train-fit leakage barrier (ADR-0009); ``None`` when
        ``barrier.pt`` is absent.
    best_metrics : MetricDict
        Validation metrics at the best epoch.
    best_epoch : int
        Epoch index of the best checkpoint.
    last_metrics : MetricDict
        Validation metrics at the last epoch.
    last_epoch : int
        Epoch index of the last checkpoint.
    """

    fold_idx: int = 0
    best_model_state_dict: dict = None  # type: ignore[assignment]
    last_model_state_dict: dict = None  # type: ignore[assignment]
    barrier: Optional[FoldBarrier] = None
    best_metrics: MetricDict = None  # type: ignore[assignment]
    best_epoch: int = 0
    last_metrics: MetricDict = None  # type: ignore[assignment]
    last_epoch: int = 0
```

- [ ] **Step 3: Update the `load_fold_checkpoint` return statement to use the new name**

Inside the same file, around line 143 (`def load_fold_checkpoint(self, fold_idx: int) -> FoldCheckpoint:`):

Replace the return annotation:

```python
    def load_fold_checkpoint(self, fold_idx: int) -> FoldCheckpoint:
```

with:

```python
    def load_fold_checkpoint(self, fold_idx: int) -> "FoldBundle":
```

And the docstring `Returns` line:

```
        FoldCheckpoint
```

with:

```
        FoldBundle
```

And the `return FoldCheckpoint(...)` call (around line 184):

```python
        return FoldCheckpoint(
            fold_idx=fold_idx,
            ...
        )
```

with:

```python
        return FoldBundle(
            fold_idx=fold_idx,
            ...
        )
```

(Leave the rest of the method body unchanged.)

- [ ] **Step 4: Update `src/training/__init__.py` — re-export from the new module path; keep both names**

Edit `src/training/__init__.py`.

Old (line 12, 25):

```python
from src.training.checkpoint_manager import CheckpointManager, FoldCheckpoint
...
__all__ = [
    ...
    "CheckpointManager",
    "FoldCheckpoint",
]
```

Replace with:

```python
from src.training.fold_checkpoint import CheckpointManager, FoldBundle
...
__all__ = [
    ...
    "CheckpointManager",
    "FoldBundle",
]
```

(We'll add `FoldCheckpoint` back to the exports in Task 2, once the new class exists.)

- [ ] **Step 5: Update every other `from src.training.checkpoint_manager import …` to use the new path**

Run:

```bash
grep -rn "from src.training.checkpoint_manager" src/ tests/
```

Expected hits (5):
- `src/training/nested_cross_validation.py:47:from src.training.checkpoint_manager import CheckpointManager`
- `src/training/cross_validation.py:19:from src.training.checkpoint_manager import CheckpointManager`
- `src/finetuning/finetuner.py:107:        from src.training.checkpoint_manager import CheckpointManager`
- `src/finetuning/finetuner.py:392:        from src.training.checkpoint_manager import CheckpointManager`
- `src/gnn_explainer/explainer.py:522:        from src.training.checkpoint_manager import CheckpointManager`
- `tests/test_training.py:22:from src.training.checkpoint_manager import CheckpointManager, FoldCheckpoint`

Replace each `src.training.checkpoint_manager` with `src.training.fold_checkpoint`.

For `tests/test_training.py:22`, also rename the imported symbol:

Old:
```python
from src.training.checkpoint_manager import CheckpointManager, FoldCheckpoint
```

New:
```python
from src.training.fold_checkpoint import CheckpointManager, FoldBundle
```

(`FoldCheckpoint` here was the dataclass — now `FoldBundle`. The test class `TestCheckpointManager` doesn't reference either symbol in its bodies — all three tests in it are `NotImplementedError` stubs — so the rename is mechanical.)

Verify zero remaining:

```bash
grep -rn "from src.training.checkpoint_manager\|src.training.checkpoint_manager" src/ tests/
```

Expected: zero matches.

- [ ] **Step 6: Smoke import test**

Run:

```bash
.venv/bin/python -c "from src.training.fold_checkpoint import CheckpointManager, FoldBundle; print('ok')"
```

Expected: `ok`.

(`Trainer` and `NestedCrossValidator` import paths require `torch_sparse` which is missing from the local venv — same caveat as PR1.)

- [ ] **Step 7: Commit**

```bash
git add src/training/fold_checkpoint.py src/training/__init__.py src/training/nested_cross_validation.py src/training/cross_validation.py src/finetuning/finetuner.py src/gnn_explainer/explainer.py tests/test_training.py
git commit -m "refactor(training): rename checkpoint_manager to fold_checkpoint, dataclass to FoldBundle

Mechanical move + dataclass rename to free the name FoldCheckpoint for
the per-fold view class in PR2 (ADR-0009). No behaviour change.
B-fresh — no compat shim for the old module path."
```

Verify the commit only renamed the file (git should show as a rename):

```bash
git show --stat HEAD | head -15
```

Expected to include something like `src/training/{checkpoint_manager.py => fold_checkpoint.py}` (git auto-detects the rename).

---

## Task 2: Introduce `FoldCheckpoint` per-fold view class

The new class owns the read side. It does **not** replace `CheckpointManager` — write-side and epoch helpers stay where they are. It coexists in the same module.

**Files:**
- Modify: `src/training/fold_checkpoint.py`
- Modify: `src/training/__init__.py`
- Create: `tests/test_fold_checkpoint.py`

- [ ] **Step 1: Write the failing test for the constructor + `fold_idx` parsing**

Create `tests/test_fold_checkpoint.py`:

```python
"""Tests for FoldCheckpoint (the per-fold read path, ADR-0009 / PR2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


class TestFoldCheckpointConstruction:
    """``FoldCheckpoint(fold_dir)`` and ``FoldCheckpoint.for_fold(root, idx)``."""

    def test_direct_construction_records_dir(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fold_dir = tmp_path / "fold_3"
        fold_dir.mkdir()
        fc = FoldCheckpoint(fold_dir)
        assert fc.fold_dir == fold_dir

    def test_for_fold_resolves_subdir(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fc = FoldCheckpoint.for_fold(tmp_path, 7)
        assert fc.fold_dir == tmp_path / "fold_7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestFoldCheckpointConstruction -v`
Expected: FAIL with `ImportError: cannot import name 'FoldCheckpoint' from 'src.training.fold_checkpoint'`.

- [ ] **Step 3: Add the `FoldCheckpoint` class skeleton (constructor + `for_fold`)**

Edit `src/training/fold_checkpoint.py`. Insert this class **between** the `FoldBundle` dataclass and `class CheckpointManager`:

```python
# ---------------------------------------------------------------------------
# Per-fold read view (ADR-0009)
# ---------------------------------------------------------------------------

class FoldCheckpoint:
    """The single read path for one outer fold's persisted artifacts.

    Construct from a fold directory (or via :meth:`for_fold` from a root
    and a fold index), then call typed accessors to load only what you
    need:

    - :meth:`load_state_dict` — raw model weights, no model construction
    - :meth:`load_model` — reconstructs the model from saved configs and
      loads weights
    - :meth:`load_barrier` — reconstructs the :class:`FoldBarrier` from
      ``barrier.pt`` (returns ``None`` when the file is absent)
    - :meth:`load_metrics` — reads ``metrics.json``
    - :meth:`load_bundle` — loads everything into a :class:`FoldBundle`

    The on-disk layout is documented under
    :ref:`Checkpoint layout <CONTEXT.md#checkpoint-layout>`.

    Parameters
    ----------
    fold_dir : str | Path
        Path to the fold directory (e.g.
        ``checkpoints/rep_0/fold_3/``).
    """

    def __init__(self, fold_dir: str | Path) -> None:
        self.fold_dir = Path(fold_dir)

    @classmethod
    def for_fold(cls, root: str | Path, fold_idx: int) -> "FoldCheckpoint":
        """Build the view for fold *fold_idx* inside *root*.

        Parameters
        ----------
        root : str | Path
            Parent directory that holds ``fold_<K>/`` subdirectories.
        fold_idx : int

        Returns
        -------
        FoldCheckpoint
        """
        return cls(Path(root) / f"fold_{fold_idx}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestFoldCheckpointConstruction -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for `load_state_dict`**

Append to `tests/test_fold_checkpoint.py`:

```python
class TestLoadStateDict:
    """``load_state_dict(variant=...)`` returns the saved model weights."""

    def _seed_fold_dir(self, fold_dir: Path) -> tuple[dict, dict]:
        fold_dir.mkdir(parents=True, exist_ok=True)
        best = {"layer.weight": torch.tensor([1.0, 2.0])}
        last = {"layer.weight": torch.tensor([3.0, 4.0])}
        torch.save(best, fold_dir / "model_best.pt")
        torch.save(last, fold_dir / "model_last.pt")
        return best, last

    def test_loads_best_by_default(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        best, _ = self._seed_fold_dir(tmp_path / "fold_0")
        loaded = FoldCheckpoint(tmp_path / "fold_0").load_state_dict()
        torch.testing.assert_close(loaded["layer.weight"], best["layer.weight"])

    def test_loads_last_variant(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        _, last = self._seed_fold_dir(tmp_path / "fold_0")
        loaded = FoldCheckpoint(tmp_path / "fold_0").load_state_dict(variant="last")
        torch.testing.assert_close(loaded["layer.weight"], last["layer.weight"])

    def test_rejects_unknown_variant(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        self._seed_fold_dir(tmp_path / "fold_0")
        with pytest.raises(ValueError, match="variant"):
            FoldCheckpoint(tmp_path / "fold_0").load_state_dict(variant="median")
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestLoadStateDict -v`
Expected: FAIL with `AttributeError: 'FoldCheckpoint' object has no attribute 'load_state_dict'`.

- [ ] **Step 7: Implement `load_state_dict`**

Append inside the `FoldCheckpoint` class:

```python
    def load_state_dict(self, *, variant: str = "best") -> dict:
        """Load raw model weights from ``model_<variant>.pt``.

        Parameters
        ----------
        variant : str
            ``"best"`` or ``"last"``.

        Returns
        -------
        dict
            Model state dict (CPU tensors).
        """
        if variant not in ("best", "last"):
            raise ValueError(
                f"variant must be 'best' or 'last', got {variant!r}"
            )
        path = self.fold_dir / f"model_{variant}.pt"
        return torch.load(path, map_location="cpu", weights_only=True)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestLoadStateDict -v`
Expected: PASS.

- [ ] **Step 9: Write the failing test for `load_barrier`**

Append to `tests/test_fold_checkpoint.py`:

```python
class TestLoadBarrier:
    """``load_barrier()`` reconstructs the FoldBarrier from ``barrier.pt``."""

    def _make_graphs(self, n: int = 20, n_rois: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        graphs = []
        labels = []
        for _ in range(n):
            x = torch.tensor(rng.normal(size=(n_rois, 4)), dtype=torch.float32)
            g = Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=n_rois,
            )
            graphs.append(g)
            labels.append(float(rng.normal()))
        return graphs, np.array(labels, dtype=np.float64)

    def test_returns_none_when_barrier_missing(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        (tmp_path / "fold_0").mkdir()
        assert FoldCheckpoint(tmp_path / "fold_0").load_barrier() is None

    def test_reconstructs_glm_from_persisted_state(self, tmp_path: Path) -> None:
        from src.training.fold_barrier import FoldBarrier
        from src.training.fold_checkpoint import FoldCheckpoint

        graphs, labels = self._make_graphs(n=40, seed=1)
        fitted = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(1, 3),
        )
        fitted.fit(graphs, labels)

        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        fitted.save(fold_dir / "barrier.pt")

        loaded = FoldCheckpoint(fold_dir).load_barrier()
        assert loaded is not None

        # Same GLM stats survive the round-trip.
        for g_a, g_b in zip(
            fitted.transform_graphs(graphs),
            loaded.transform_graphs(graphs),
        ):
            assert torch.allclose(g_a.x, g_b.x)
```

- [ ] **Step 10: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestLoadBarrier -v`
Expected: FAIL with `AttributeError: 'FoldCheckpoint' object has no attribute 'load_barrier'`.

- [ ] **Step 11: Implement `load_barrier`**

Append inside the `FoldCheckpoint` class:

```python
    def load_barrier(self) -> Optional[FoldBarrier]:
        """Reload the per-fold leakage barrier from ``barrier.pt``.

        Returns ``None`` when ``barrier.pt`` is absent — typical for old
        checkpoints predating ADR-0009.

        The barrier reconstructs its transformers from the persisted
        state alone: the GLM substate carries ``(col_start, col_end)``
        and the label-norm substate carries its strategy. Composite-
        mode labels (``LabelBuilder``-fit state) are NOT round-tripped
        — callers that need ``transform_labels`` on composite-mode
        folds must reconstruct the ``LabelBuilder`` themselves and call
        :meth:`FoldBarrier.load` directly with ``label_builder=...``.

        Returns
        -------
        Optional[FoldBarrier]
        """
        path = self.fold_dir / "barrier.pt"
        if not path.exists():
            return None
        return FoldBarrier.load(path)
```

- [ ] **Step 12: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestLoadBarrier -v`
Expected: PASS.

- [ ] **Step 13: Write the failing test for `load_metrics`**

Append to `tests/test_fold_checkpoint.py`:

```python
class TestLoadMetrics:
    """``load_metrics()`` returns the parsed ``metrics.json``."""

    def test_returns_parsed_json(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        payload = {
            "best": {"metrics": {"mae": 0.1}, "epoch": 3},
            "last": {"metrics": {"mae": 0.2}, "epoch": 9},
        }
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(payload, f)

        assert FoldCheckpoint(fold_dir).load_metrics() == payload
```

- [ ] **Step 14: Implement `load_metrics`**

Append inside the `FoldCheckpoint` class:

```python
    def load_metrics(self) -> dict:
        """Return the parsed contents of ``metrics.json``.

        Returns
        -------
        dict
            Top-level keys: ``"best"`` and ``"last"``, each mapping to
            a ``{"metrics": MetricDict, "epoch": int}`` payload.
        """
        with open(self.fold_dir / "metrics.json") as f:
            return json.load(f)
```

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestLoadMetrics -v`
Expected: PASS.

- [ ] **Step 15: Write the failing test for `load_bundle`**

Append to `tests/test_fold_checkpoint.py`:

```python
class TestLoadBundle:
    """``load_bundle()`` returns a populated FoldBundle from disk."""

    def _seed_fold_dir(self, fold_dir: Path) -> None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"w": torch.tensor([1.0])}, fold_dir / "model_best.pt",
        )
        torch.save(
            {"w": torch.tensor([2.0])}, fold_dir / "model_last.pt",
        )
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "best": {"metrics": {"mae": 0.1}, "epoch": 1},
                    "last": {"metrics": {"mae": 0.2}, "epoch": 9},
                },
                f,
            )

    def test_populates_all_fields_without_barrier(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldBundle, FoldCheckpoint

        fold_dir = tmp_path / "fold_5"
        self._seed_fold_dir(fold_dir)

        bundle = FoldCheckpoint(fold_dir).load_bundle()
        assert isinstance(bundle, FoldBundle)
        assert bundle.fold_idx == 5
        assert bundle.barrier is None
        assert bundle.best_epoch == 1 and bundle.last_epoch == 9
        assert bundle.best_metrics == {"mae": 0.1}
        assert bundle.last_metrics == {"mae": 0.2}
        torch.testing.assert_close(
            bundle.best_model_state_dict["w"], torch.tensor([1.0]),
        )
        torch.testing.assert_close(
            bundle.last_model_state_dict["w"], torch.tensor([2.0]),
        )

    def test_fold_idx_parsed_from_dir_name(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        fold_dir = tmp_path / "fold_42"
        self._seed_fold_dir(fold_dir)
        assert FoldCheckpoint(fold_dir).load_bundle().fold_idx == 42

    def test_raises_on_malformed_dir_name(self, tmp_path: Path) -> None:
        from src.training.fold_checkpoint import FoldCheckpoint

        weird_dir = tmp_path / "not_a_fold"
        self._seed_fold_dir(weird_dir)
        with pytest.raises(ValueError, match="fold directory name"):
            FoldCheckpoint(weird_dir).load_bundle()
```

- [ ] **Step 16: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestLoadBundle -v`
Expected: FAIL with `AttributeError: 'FoldCheckpoint' object has no attribute 'load_bundle'`.

- [ ] **Step 17: Implement `load_bundle` (plus a small `_fold_idx_from_dir` helper)**

Append inside the `FoldCheckpoint` class:

```python
    def load_bundle(self) -> "FoldBundle":
        """Load every artifact in the fold directory into a FoldBundle.

        Returns
        -------
        FoldBundle
        """
        metrics = self.load_metrics()
        return FoldBundle(
            fold_idx=self._fold_idx_from_dir(),
            best_model_state_dict=self.load_state_dict(variant="best"),
            last_model_state_dict=self.load_state_dict(variant="last"),
            barrier=self.load_barrier(),
            best_metrics=metrics["best"]["metrics"],
            best_epoch=metrics["best"]["epoch"],
            last_metrics=metrics["last"]["metrics"],
            last_epoch=metrics["last"]["epoch"],
        )

    def _fold_idx_from_dir(self) -> int:
        """Parse the fold index from the directory's basename."""
        name = self.fold_dir.name
        if not name.startswith("fold_"):
            raise ValueError(
                f"fold directory name must start with 'fold_', got "
                f"{name!r} (full path: {self.fold_dir})"
            )
        try:
            return int(name[len("fold_"):])
        except ValueError as e:
            raise ValueError(
                f"fold directory name must end with an integer index, "
                f"got {name!r}"
            ) from e
```

- [ ] **Step 18: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py::TestLoadBundle -v`
Expected: PASS.

- [ ] **Step 19: Re-export `FoldCheckpoint` from `src/training/__init__.py`**

Edit `src/training/__init__.py`.

Old (after Task 1):

```python
from src.training.fold_checkpoint import CheckpointManager, FoldBundle
...
__all__ = [
    ...
    "CheckpointManager",
    "FoldBundle",
]
```

Replace with:

```python
from src.training.fold_checkpoint import CheckpointManager, FoldBundle, FoldCheckpoint
...
__all__ = [
    ...
    "CheckpointManager",
    "FoldBundle",
    "FoldCheckpoint",
]
```

- [ ] **Step 20: Run the whole new test file**

Run: `.venv/bin/python -m pytest tests/test_fold_checkpoint.py -v`
Expected: every test PASS.

- [ ] **Step 21: Commit**

```bash
git add src/training/fold_checkpoint.py src/training/__init__.py tests/test_fold_checkpoint.py
git commit -m "feat(fold_checkpoint): introduce per-fold read-view class

FoldCheckpoint(fold_dir) is the single read path for one outer
fold's artifacts (ADR-0009 / PR2). Owns load_state_dict, load_model,
load_barrier, load_metrics, load_bundle. CheckpointManager's
load_fold_checkpoint / load_model_for_fold are now duplicates of
this; subsequent tasks migrate callers and remove them."
```

---

## Task 3: Add `FoldCheckpoint.load_model` (config-driven, with factory fallback)

Replicates the behaviour of the existing `CheckpointManager.load_model_for_fold` so callers can migrate next.

**Files:**
- Modify: `src/training/fold_checkpoint.py`
- (No unit test in `tests/test_fold_checkpoint.py` — the config-driven path requires `src.models.registry.get_model`, which imports `torch_sparse` and fails to collect in the local venv. The migration coverage comes from the Explainer's existing integration tests in Task 5.)

- [ ] **Step 1: Implement `load_model`**

Append inside the `FoldCheckpoint` class:

```python
    def load_model(
        self,
        *,
        model_factory: Optional[Callable[[], "BrainGNN"]] = None,
        num_nodes: int = 0,
        variant: str = "best",
    ) -> "BrainGNN":
        """Reconstruct the model with weights loaded.

        Reads ``model_config.json`` and ``feature_config.json`` from
        the fold directory to rebuild the exact architecture used
        during training. Falls back to *model_factory* when either
        config file is absent (old checkpoints without config
        metadata).

        Parameters
        ----------
        model_factory : Optional[Callable[[], BrainGNN]]
            Used only when the saved configs are absent. Required in
            that case; ``None`` then raises ``FileNotFoundError``.
        num_nodes : int
            Number of nodes per graph (passed to ``get_model`` when
            rebuilding from configs).
        variant : str
            ``"best"`` or ``"last"``.

        Returns
        -------
        BrainGNN
        """
        from src.configs.feature_config import FeatureConfig
        from src.configs.model_config import ModelConfig
        from src.models.registry import get_model

        model_cfg_path = self.fold_dir / "model_config.json"
        feature_cfg_path = self.fold_dir / "feature_config.json"

        if model_cfg_path.exists() and feature_cfg_path.exists():
            with open(model_cfg_path) as f:
                model_cfg = ModelConfig(**json.load(f))
            with open(feature_cfg_path) as f:
                feature_cfg = FeatureConfig(**json.load(f))
            model = get_model(
                name=model_cfg.name,
                cfg=model_cfg,
                node_feat_dim=feature_cfg.node_feat_dim,
                edge_feat_dim=feature_cfg.edge_feat_dim,
                num_nodes=num_nodes,
            )
        elif model_factory is not None:
            log.warning(
                "model_config.json / feature_config.json not found in %s — "
                "using model_factory() as fallback.",
                self.fold_dir,
            )
            model = model_factory()
        else:
            raise FileNotFoundError(
                f"{self.fold_dir} has no model_config.json/feature_config.json "
                "and no model_factory was provided; cannot reconstruct model."
            )

        model.load_state_dict(self.load_state_dict(variant=variant))
        return model
```

- [ ] **Step 2: Smoke import (the new method imports its dependencies lazily)**

Run: `.venv/bin/python -c "from src.training.fold_checkpoint import FoldCheckpoint; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/training/fold_checkpoint.py
git commit -m "feat(fold_checkpoint): FoldCheckpoint.load_model with factory fallback

Mirrors CheckpointManager.load_model_for_fold's behaviour — rebuilds
the model from saved configs, falls back to a caller-supplied factory
when configs are absent. Next tasks migrate callers off the old
method (ADR-0009 / PR2)."
```

---

## Task 4: Migrate `Finetuner._load_weights_for_fold` to the new API

The Finetuner only needs the best/last model state dict from a fold. Replace the `load_fold_checkpoint(...).best_model_state_dict` round-trip with `FoldCheckpoint.for_fold(...).load_state_dict(...)` — one direct read instead of loading the whole bundle.

**Files:**
- Modify: `src/finetuning/finetuner.py:101-119`

- [ ] **Step 1: Inspect the current method**

```bash
sed -n '101,119p' src/finetuning/finetuner.py
```

Expected current body (verbatim):

```python
    def _load_weights_for_fold(self, checkpoint_path: Path, fold_idx: int) -> dict:
        """Load pretrained weights from a specific fold checkpoint.

        Used to match pretrained weights to the current CV fold so that
        the backbone never sees the fold's test subjects during pretraining.
        """
        from src.training.checkpoint_manager import CheckpointManager
        ckpt = CheckpointManager(checkpoint_path).load_fold_checkpoint(fold_idx)
        variant = self.cfg.checkpoint_variant
        state_dict = (
            ckpt.best_model_state_dict if variant == "best" else ckpt.last_model_state_dict
        )
        log.info(
            "Fold %d: loaded pretrained weights from %s (%d keys)",
            fold_idx,
            Path(checkpoint_path) / f"fold_{fold_idx}",
            len(state_dict),
        )
        return state_dict
```

(Note: this was already updated to `src.training.fold_checkpoint` in Task 1 Step 5; the body below still calls the old `load_fold_checkpoint` API.)

- [ ] **Step 2: Replace the body**

Edit `src/finetuning/finetuner.py:101-119`. Replace the method body (everything from the docstring through `return state_dict`):

```python
    def _load_weights_for_fold(self, checkpoint_path: Path, fold_idx: int) -> dict:
        """Load pretrained weights from a specific fold checkpoint.

        Used to match pretrained weights to the current CV fold so that
        the backbone never sees the fold's test subjects during pretraining.
        """
        from src.training.fold_checkpoint import FoldCheckpoint
        fc = FoldCheckpoint.for_fold(checkpoint_path, fold_idx)
        state_dict = fc.load_state_dict(variant=self.cfg.checkpoint_variant)
        log.info(
            "Fold %d: loaded pretrained weights from %s (%d keys)",
            fold_idx,
            fc.fold_dir,
            len(state_dict),
        )
        return state_dict
```

- [ ] **Step 3: Smoke import**

Run: `.venv/bin/python -c "from src.finetuning.finetuner import Finetuner; print('ok')"`
Expected: `ok`.

(If this fails on `torch_sparse` it's the pre-existing local-venv gap — not a regression from this PR. Confirm with `git diff main -- src/finetuning/finetuner.py` showing only the planned edits.)

- [ ] **Step 4: Commit**

```bash
git add src/finetuning/finetuner.py
git commit -m "refactor(finetuner): consume FoldCheckpoint.load_state_dict directly

Replaces the load_fold_checkpoint→bundle→pick-field round-trip with
a direct per-variant read. No behaviour change (ADR-0009 / PR2)."
```

---

## Task 5: Migrate the Explainer's GLM re-fit branch to the persisted barrier

The PR1 dead-letter. The Explainer currently re-fits `GLMFeatureNormalizer` on train-recovered graphs every time, even though `barrier.pt` (with the same fitted stats) was written at training time. PR2 wires the reload through.

**Files:**
- Modify: `src/gnn_explainer/explainer.py:496-553`

- [ ] **Step 1: Inspect the current `_explain_one_fold` head**

```bash
sed -n '496,555p' src/gnn_explainer/explainer.py
```

Expected current head includes:

```python
        from src.training.checkpoint_manager import CheckpointManager
        from src.training.glm_normalizer import GLMFeatureNormalizer

        ckpt_path = fold_ckpt_dir / "model_best.pt"
        ...
        num_nodes = dataset[0].num_nodes if dataset else 0
        model, _ = CheckpointManager(fold_ckpt_root).load_model_for_fold(
            fold_idx, model_factory, num_nodes, variant="best",
        )
        model.eval()
        model.to(device)
        log.debug("Loaded model weights from %s", ckpt_path)

        test_graphs = [copy.copy(dataset[i]) for i in test_idx]

        if glm_col_range is not None and glm_normalize:
            col_start, col_end = glm_col_range
            glm_norm = GLMFeatureNormalizer(col_start, col_end)
            train_graphs_for_norm = [copy.copy(dataset[i]) for i in train_idx]
            glm_norm.fit(train_graphs_for_norm)
            glm_norm.transform(test_graphs)
            log.debug(
                "%s: GLM features (cols [%d:%d)) z-scored on %d train "
                "subjects and applied to %d test subjects.",
                fold_log_tag, col_start, col_end, len(train_idx), len(test_idx),
            )
```

(Reminder: Task 1 already rewrote the import from `checkpoint_manager` to `fold_checkpoint`. The body here still calls `load_model_for_fold` on the old class.)

- [ ] **Step 2: Replace the imports and the GLM block**

Replace lines 522-553 (the two imports through the end of the `if glm_col_range is not None and glm_normalize:` block):

Old:

```python
        from src.training.checkpoint_manager import CheckpointManager
        from src.training.glm_normalizer import GLMFeatureNormalizer

        ckpt_path = fold_ckpt_dir / "model_best.pt"
        if not ckpt_path.exists():
            log.warning(
                "Checkpoint not found: %s — skipping %s.",
                ckpt_path, fold_log_tag,
            )
            return None

        num_nodes = dataset[0].num_nodes if dataset else 0
        model, _ = CheckpointManager(fold_ckpt_root).load_model_for_fold(
            fold_idx, model_factory, num_nodes, variant="best",
        )
        model.eval()
        model.to(device)
        log.debug("Loaded model weights from %s", ckpt_path)

        test_graphs = [copy.copy(dataset[i]) for i in test_idx]

        if glm_col_range is not None and glm_normalize:
            col_start, col_end = glm_col_range
            glm_norm = GLMFeatureNormalizer(col_start, col_end)
            train_graphs_for_norm = [copy.copy(dataset[i]) for i in train_idx]
            glm_norm.fit(train_graphs_for_norm)
            glm_norm.transform(test_graphs)
            log.debug(
                "%s: GLM features (cols [%d:%d)) z-scored on %d train "
                "subjects and applied to %d test subjects.",
                fold_log_tag, col_start, col_end, len(train_idx), len(test_idx),
            )
```

Replace with:

```python
        from src.training.fold_checkpoint import FoldCheckpoint

        ckpt_path = fold_ckpt_dir / "model_best.pt"
        if not ckpt_path.exists():
            log.warning(
                "Checkpoint not found: %s — skipping %s.",
                ckpt_path, fold_log_tag,
            )
            return None

        num_nodes = dataset[0].num_nodes if dataset else 0
        fc = FoldCheckpoint(fold_ckpt_dir)
        model = fc.load_model(
            model_factory=model_factory, num_nodes=num_nodes, variant="best",
        )
        model.eval()
        model.to(device)
        log.debug("Loaded model weights from %s", ckpt_path)

        # ADR-0009 / PR2: the persisted FoldBarrier owns the GLM
        # statistics (and any other per-fold leakage state) fit on the
        # outer-train pool. The previous re-fit-from-train-indices branch
        # is gone — explanation reproducibility now decouples from
        # upstream split derivation.
        barrier = fc.load_barrier()
        if barrier is not None:
            test_graphs = barrier.transform_graphs(
                [dataset[i] for i in test_idx]
            )
            log.debug(
                "%s: barrier.pt loaded from %s; %d test subjects transformed.",
                fold_log_tag, fc.fold_dir, len(test_idx),
            )
        else:
            log.warning(
                "%s: barrier.pt not present in %s — falling back to raw "
                "test graphs. GLM-normalised checkpoints predating "
                "ADR-0009 / PR1 are not supported; re-run training.",
                fold_log_tag, fc.fold_dir,
            )
            test_graphs = [copy.copy(dataset[i]) for i in test_idx]
```

(The `train_idx`, `glm_col_range`, `glm_normalize` parameters of `_explain_one_fold` become unused inside the body but stay in the signature for now — the dispatchers `_run_legacy` / `_run_nested` still compute and pass them, and removing them is a follow-up cleanup outside PR2's scope.)

- [ ] **Step 3: Verify `copy` is still needed in the module**

Run:

```bash
grep -n "copy\." src/gnn_explainer/explainer.py
```

Expected: at least one match remains (the `copy.copy(dataset[i])` in the fallback branch). The `import copy` line stays.

- [ ] **Step 4: Static check that the explainer no longer references GLMFeatureNormalizer**

Run:

```bash
grep -n "GLMFeatureNormalizer\|glm_normalizer" src/gnn_explainer/explainer.py
```

Expected: zero matches. The whole point of PR2 is that the GLM stats now come from `barrier.pt`.

- [ ] **Step 5: Smoke import**

Run: `.venv/bin/python -c "import importlib; importlib.import_module('src.gnn_explainer.explainer'); print('ok')"`
Expected: `ok` (or the pre-existing `torch_sparse` failure — confirm with `git diff main -- src/gnn_explainer/explainer.py` matches the planned edits).

- [ ] **Step 6: Commit**

```bash
git add src/gnn_explainer/explainer.py
git commit -m "feat(explainer): consume persisted FoldBarrier instead of re-fitting GLM

The GLM re-fit-from-train-indices branch is replaced by a one-line
FoldCheckpoint(fold_ckpt_dir).load_barrier() + transform_graphs.
Explanation reproducibility now decouples from upstream split
derivation: the persisted barrier.pt is the source of truth for the
fitted leakage state (ADR-0009 / PR2). Old checkpoints without
barrier.pt fall through with a warning and raw test graphs."
```

---

## Task 6: Update the explainer test fixture + add a barrier-reload regression test

The current fixture in `tests/test_gnn_explainer.py` does NOT write `barrier.pt`. Post-PR2 the explainer warns and falls back when the barrier is absent, so the existing tests will still pass — but the *new* path is uncovered until we exercise it. Two changes: (a) make the fixture write a fitted `barrier.pt` so the existing tests cover the reload path; (b) add a focused test that proves the explainer is using the persisted GLM stats rather than re-fitting.

**Files:**
- Modify: `tests/test_gnn_explainer.py:107-134` (the `_write_fold_artifacts` helper)
- Modify: `tests/test_gnn_explainer.py:137-203` (the two `_build_*_checkpoint_tree` helpers — they need access to the dataset/labels to fit a barrier)
- Append a new test class to the same file.

- [ ] **Step 1: Update `_write_fold_artifacts` to accept and persist a barrier**

Edit `tests/test_gnn_explainer.py`.

Old (lines 107-134):

```python
def _write_fold_artifacts(
    fold_dir: Path,
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    feature_cfg: dict,
) -> None:
    """Write the on-disk files CheckpointManager.load_model_for_fold expects.

    The Explainer's GLM re-fit branch (current PR1 state) does not consume
    a persisted leakage barrier, so ``barrier.pt`` is not synthesised
    here — ``load_model_for_fold`` returns ``barrier=None`` and the
    explainer test path tolerates it.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model_best.pt")
    torch.save(model.state_dict(), fold_dir / "model_last.pt")
    with open(fold_dir / "model_config.json", "w") as f:
        json.dump(model_cfg.model_dump(), f)
    with open(fold_dir / "feature_config.json", "w") as f:
        json.dump(feature_cfg, f)
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "best": {"metrics": {"mae": 0.1}, "epoch": 0},
                "last": {"metrics": {"mae": 0.1}, "epoch": 0},
            },
            f,
        )
```

Replace with:

```python
def _write_fold_artifacts(
    fold_dir: Path,
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    feature_cfg: dict,
    *,
    barrier=None,
) -> None:
    """Write the on-disk files FoldCheckpoint expects to load.

    Post-PR2 (ADR-0009) the explainer reloads the fold barrier from
    ``barrier.pt`` instead of re-fitting on recovered train indices.
    Tests that want to exercise that reload path pass a fitted
    ``FoldBarrier`` here; tests that exercise the warn-and-fall-back
    path leave it as ``None``.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model_best.pt")
    torch.save(model.state_dict(), fold_dir / "model_last.pt")
    with open(fold_dir / "model_config.json", "w") as f:
        json.dump(model_cfg.model_dump(), f)
    with open(fold_dir / "feature_config.json", "w") as f:
        json.dump(feature_cfg, f)
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "best": {"metrics": {"mae": 0.1}, "epoch": 0},
                "last": {"metrics": {"mae": 0.1}, "epoch": 0},
            },
            f,
        )
    if barrier is not None:
        barrier.save(fold_dir / "barrier.pt")
```

- [ ] **Step 2: Add a helper that fits a barrier on the synthetic dataset**

Insert after `_write_fold_artifacts` (and before `_build_nested_checkpoint_tree`):

```python
def _fit_synthetic_barrier(
    graphs: List,
    labels: np.ndarray,
    *,
    glm_col_range=None,
):
    """Fit a FoldBarrier on the whole synthetic dataset.

    The tests don't care about leakage-correctness of the synthesised
    barrier — they only need a valid persisted artefact that the
    explainer can reload. Fitting on the full dataset keeps the
    helpers stateless (they don't need train indices).
    """
    from src.training.fold_barrier import FoldBarrier

    barrier = FoldBarrier(
        label_norm_strategy="standard",
        glm_col_range=glm_col_range,
        glm_normalize=glm_col_range is not None,
    )
    barrier.fit(graphs, labels)
    return barrier
```

- [ ] **Step 3: Thread `graphs` through the existing checkpoint-tree builders so they can fit a barrier**

Edit `_build_nested_checkpoint_tree` (around lines 137-184). The signature change is **adding `graphs`** as a required keyword argument. The body adds one barrier-fit call.

Old signature:

```python
def _build_nested_checkpoint_tree(
    ckpt_root: Path,
    labels: np.ndarray,
    *,
    n_repetitions: int,
    n_outer_folds: int,
    stratify_bins: int = 4,
    feat_dim: int = 3,
    num_nodes: int = 6,
) -> List[int]:
```

New signature:

```python
def _build_nested_checkpoint_tree(
    ckpt_root: Path,
    labels: np.ndarray,
    *,
    graphs: List,
    n_repetitions: int,
    n_outer_folds: int,
    stratify_bins: int = 4,
    feat_dim: int = 3,
    num_nodes: int = 6,
) -> List[int]:
```

Inside the loop body, replace:

```python
            _write_fold_artifacts(
                fold_dir, factory(), model_cfg, feature_cfg
            )
```

with:

```python
            _write_fold_artifacts(
                fold_dir, factory(), model_cfg, feature_cfg,
                barrier=_fit_synthetic_barrier(graphs, labels),
            )
```

Apply the same edit to `_build_legacy_checkpoint_tree` (around lines 187-203):

Old signature:

```python
def _build_legacy_checkpoint_tree(
    ckpt_root: Path,
    *,
    n_folds: int,
    feat_dim: int = 3,
    num_nodes: int = 6,
) -> None:
```

New signature:

```python
def _build_legacy_checkpoint_tree(
    ckpt_root: Path,
    *,
    graphs: List,
    labels: np.ndarray,
    n_folds: int,
    feat_dim: int = 3,
    num_nodes: int = 6,
) -> None:
```

Body — replace:

```python
        _write_fold_artifacts(
            fold_dir, factory(), model_cfg, feature_cfg
        )
```

with:

```python
        _write_fold_artifacts(
            fold_dir, factory(), model_cfg, feature_cfg,
            barrier=_fit_synthetic_barrier(graphs, labels),
        )
```

- [ ] **Step 4: Update every call site of the two builders to pass `graphs` (and `labels` for legacy)**

```bash
grep -n "_build_nested_checkpoint_tree\|_build_legacy_checkpoint_tree" tests/test_gnn_explainer.py
```

Expected call sites:
- `TestNestedCheckpointLayout.test_writes_importance_matrix_per_rep_and_fold` (around line 238)
- `TestNestedAggregation.test_writes_aggregate_per_fold_mean_and_std` (around line 284)
- `TestRepRestriction.test_only_requested_rep_is_written` (around line 339)
- `TestLegacyLayoutFallback.test_legacy_layout_writes_per_fold_outputs` (around line 383)

For each nested-builder call, add `graphs=graphs,` to the kwargs. Example:

Old:
```python
        _build_nested_checkpoint_tree(
            ckpt_root,
            labels,
            n_repetitions=2,
            n_outer_folds=3,
            stratify_bins=4,
            num_nodes=6,
        )
```

New:
```python
        _build_nested_checkpoint_tree(
            ckpt_root,
            labels,
            graphs=graphs,
            n_repetitions=2,
            n_outer_folds=3,
            stratify_bins=4,
            num_nodes=6,
        )
```

For the legacy-builder call (`TestLegacyLayoutFallback`), add both:

Old:
```python
        _build_legacy_checkpoint_tree(
            ckpt_root, n_folds=3, num_nodes=6,
        )
```

New:
```python
        _build_legacy_checkpoint_tree(
            ckpt_root, graphs=graphs, labels=labels, n_folds=3, num_nodes=6,
        )
```

- [ ] **Step 5: Add the barrier-reload regression test class**

Append to `tests/test_gnn_explainer.py`:

```python
# ---------------------------------------------------------------------------
# PR2 regression — explainer uses persisted barrier, not a re-fit
# ---------------------------------------------------------------------------


class TestExplainerConsumesPersistedBarrier:
    """The Explainer must read GLM stats from ``barrier.pt`` (ADR-0009 / PR2).

    Forcing the persisted barrier to carry unrealistic GLM mean/std
    means any code path that re-fits from train indices would produce
    different transformed test features — so this is a direct
    regression on PR1's dead-letter (barrier written but not consumed).
    """

    def test_explainer_uses_persisted_glm_stats(self, tmp_path: Path) -> None:
        from src.training.fold_barrier import FoldBarrier

        graphs, labels = _make_synthetic_dataset(
            n_subjects=24, n_rois=6, feat_dim=3, seed=4,
        )

        # Build a barrier with GLM range covering all 3 feature columns,
        # fit on the real synthetic features, then override its stats
        # with extreme values. If the explainer falls back to re-fitting,
        # the test graphs' x will NOT show the override.
        barrier = FoldBarrier(
            label_norm_strategy="standard", glm_col_range=(0, 3),
        )
        barrier.fit(graphs, labels)
        # Sentinel stats: mean 100, std 0.5 — far from the true ~N(0,1).
        n_rois = graphs[0].x.shape[0]
        barrier._glm.mean_ = torch.full((n_rois, 3), 100.0)
        barrier._glm.std_ = torch.full((n_rois, 3), 0.5)

        ckpt_root = tmp_path / "ckpt"
        ckpt_root.mkdir()
        model_cfg = _make_model_cfg(feat_dim=3)
        feature_cfg = _make_feature_cfg_dict(feat_dim=3)
        factory = _model_factory(feat_dim=3, num_nodes=6)
        # The legacy CrossValidator path needs n_folds >= 2 (sklearn KFold).
        # Write the SAME sentinel barrier into every fold dir so whichever
        # fold the monkey-patch fires on, the assertion is meaningful.
        for fold_idx in range(2):
            _write_fold_artifacts(
                ckpt_root / f"fold_{fold_idx}", factory(),
                model_cfg, feature_cfg, barrier=barrier,
            )

        trainer_cfg = TrainerConfig(
            epochs=1,
            early_stopping_patience=1,
            n_folds=2,
            inner_hpo_trials=0,
            hpo_metric="val_mae",
            checkpoint_dir=str(ckpt_root),
            stratify_bins=2,
            device="cpu",
            seed=11,
        )
        explainer_cfg = ExplainerConfig(
            enabled=True, epochs=1, edge_size=0.1, save_subject_masks=False,
        )

        # Intercept FoldBarrier.transform_graphs so we capture the test
        # graphs *immediately after* the GLM transform — before the PyG
        # explainer mutates the model's input with its mask optimisation.
        # This isolates the regression assertion from any model-side
        # quirks (extreme inputs could otherwise NaN the conv layers).
        from src.training import fold_barrier as fb_module
        captured: dict = {}
        original_transform = fb_module.FoldBarrier.transform_graphs

        def _capture(self, graphs):  # type: ignore[no-redef]
            out = original_transform(self, graphs)
            captured.setdefault(
                "x_list", [g.x.detach().clone() for g in out],
            )
            return out

        fb_module.FoldBarrier.transform_graphs = _capture
        try:
            runner = GNNExplainerRunner(cfg=explainer_cfg)
            runner.run(
                dataset=graphs,
                labels=labels,
                model_factory=factory,
                trainer_cfg=trainer_cfg,
                glm_col_range=(0, 3),
                glm_normalize=True,
            )
        finally:
            fb_module.FoldBarrier.transform_graphs = original_transform

        assert "x_list" in captured, (
            "FoldBarrier.transform_graphs was never called — the explainer "
            "did not consume the persisted barrier."
        )
        first_x = captured["x_list"][0]
        # The transformed x must reflect the sentinel:
        #   (raw ~N(0,1) - 100) / 0.5  →  values around -200
        # If the explainer re-fit GLM stats on the synthetic data, values
        # would be around N(0, 1) — not large negatives.
        assert first_x.min().item() < -50.0, (
            "Explainer appears to have re-fit GLM stats — the persisted "
            "barrier.pt sentinel (mean=100, std=0.5) did not survive to "
            "the transformed test graphs. Got min={:.3f}".format(
                first_x.min().item()
            )
        )
```

- [ ] **Step 6: Run all `tests/test_gnn_explainer.py` tests**

Run: `.venv/bin/python -m pytest tests/test_gnn_explainer.py -v`

If pytest collection fails on `torch_sparse` / `src.models` import, this is the pre-existing local-venv gap — same as PR1. The full suite must pass on the cluster. Document in the PR description.

If collection succeeds, expected: every test PASS, including `TestExplainerConsumesPersistedBarrier::test_explainer_uses_persisted_glm_stats`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_gnn_explainer.py
git commit -m "test(explainer): cover persisted-barrier reload path (PR2 regression)

The fixture now writes a real barrier.pt fitted on the synthetic
dataset so the existing nested/legacy/rep-restriction tests exercise
the new FoldCheckpoint.load_barrier() codepath. A new
TestExplainerConsumesPersistedBarrier test pins the regression: it
forces sentinel GLM stats into barrier.pt and asserts the model's
input reflects them, which would fail if the explainer were still
re-fitting from train indices (ADR-0009 / PR2)."
```

---

## Task 7: Remove `CheckpointManager.load_fold_checkpoint` and `load_model_for_fold`

Both methods are dead after Tasks 4 and 5. Removing them tightens `CheckpointManager`'s surface to write-side + epoch ops, which is the architectural goal of the rename.

**Files:**
- Modify: `src/training/fold_checkpoint.py`

- [ ] **Step 1: Verify no callers remain**

```bash
grep -rn "load_fold_checkpoint\|load_model_for_fold" src/ tests/ docs/
```

Expected: only matches inside `src/training/fold_checkpoint.py` itself (the method definitions, about to be deleted). Anything else in `src/` or `tests/` is a leftover caller — fix before deleting.

- [ ] **Step 2: Delete the two methods from `CheckpointManager`**

Edit `src/training/fold_checkpoint.py`. Delete the entire `load_fold_checkpoint` method (around lines 143-193) and the entire `load_model_for_fold` method (around lines 195-264). Keep `save_fold_checkpoint`, the epoch helpers (`get_epoch_ckpt_dir`, `save_epoch_checkpoint`, `list_epoch_checkpoints`, `get_common_epochs`), and `get_fold_dir`.

After deletion, the `CheckpointManager` class should contain:
- `__init__`
- `save_fold_checkpoint`
- `get_epoch_ckpt_dir`
- `save_epoch_checkpoint`
- `list_epoch_checkpoints`
- `get_common_epochs`
- `get_fold_dir`

- [ ] **Step 3: Drop now-unused imports inside the module**

If `Callable`, `TYPE_CHECKING`, and the `BrainGNN` forward reference are only used inside the deleted methods, remove them. Check:

```bash
grep -n "Callable\|TYPE_CHECKING\|BrainGNN" src/training/fold_checkpoint.py
```

If only the (deleted) signatures referenced `Callable` / `TYPE_CHECKING` / `BrainGNN`, delete those imports. If `FoldCheckpoint.load_model` references them (it does — see Task 3), keep them.

After Task 3, `FoldCheckpoint.load_model` uses `Callable`, `Optional`, and `"BrainGNN"` (string forward reference). `TYPE_CHECKING` and the `from src.models.base_model import BrainGNN` block are still needed.

- [ ] **Step 4: Smoke import**

Run: `.venv/bin/python -c "from src.training.fold_checkpoint import CheckpointManager, FoldCheckpoint, FoldBundle; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Grep audit — no stale references in `src/` or `tests/`**

```bash
grep -rn "load_fold_checkpoint\|load_model_for_fold" src/ tests/
```

Expected: zero matches.

- [ ] **Step 6: Commit**

```bash
git add src/training/fold_checkpoint.py
git commit -m "refactor(fold_checkpoint): drop CheckpointManager read methods

load_fold_checkpoint and load_model_for_fold are replaced by
FoldCheckpoint (PR2). CheckpointManager retains save_fold_checkpoint,
the epoch helpers, and get_common_epochs — the multi-fold operations
its name still describes."
```

---

## Task 8: Update CONTEXT.md

**Files:**
- Modify: `CONTEXT.md`

- [ ] **Step 1: Update the file map line for the renamed module**

Edit `CONTEXT.md`. Find (around line 121):

```
│   └── checkpoint_manager.py       # CheckpointManager: save/load fold and epoch checkpoints
```

Replace with:

```
│   └── fold_checkpoint.py          # FoldCheckpoint (per-fold read path) + FoldBundle + CheckpointManager (write-side / epoch helpers)
```

- [ ] **Step 2: Update the `Checkpoint` glossary entry**

Find (around line 77, after PR1's edit):

```
... plus per-fold metadata (`best_hparams.json`, `trials.csv`, `test_predictions.npz`, `model_config.json`, `feature_config.json`, `metrics.json`). See [Checkpoint layout](#checkpoint-layout) for the full file list. Legacy `HydraSweep` runs still write the flat `checkpoints/fold_<K>/` layout.
```

Replace with:

```
... plus per-fold metadata (`best_hparams.json`, `trials.csv`, `test_predictions.npz`, `model_config.json`, `feature_config.json`, `metrics.json`). Reads go through `FoldCheckpoint(fold_dir)` — the single per-fold read path that owns model + barrier + metrics loading (ADR-0009). Writes go through `CheckpointManager`. See [Checkpoint layout](#checkpoint-layout) for the full file list. Legacy `HydraSweep` runs still write the flat `checkpoints/fold_<K>/` layout.
```

- [ ] **Step 3: Update the pipeline sketch reference (around line 176)**

Find:

```
    │     ├── CheckpointManager.save_fold_checkpoint(rep_<R>/fold_<K>/)
```

This is still accurate (writes use `CheckpointManager`) — no change.

- [ ] **Step 4: Commit**

```bash
git add CONTEXT.md
git commit -m "docs(context): record FoldCheckpoint as the per-fold read path

PR2 renames src/training/checkpoint_manager.py to fold_checkpoint.py
and introduces FoldCheckpoint as the single per-fold read view.
CheckpointManager's surface tightens to writes + epoch helpers
(ADR-0009 consequences)."
```

---

## Task 9: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Grep — no orphan references to the old module path**

```bash
grep -rn "src.training.checkpoint_manager\|from src\.training\.checkpoint_manager" src/ tests/ docs/
```

Expected: zero matches.

- [ ] **Step 2: Grep — no callers of the deleted methods**

```bash
grep -rn "load_fold_checkpoint\|load_model_for_fold" src/ tests/ docs/
```

Expected: zero matches.

- [ ] **Step 3: Grep — explainer doesn't import GLMFeatureNormalizer anymore**

```bash
grep -n "GLMFeatureNormalizer\|glm_normalizer" src/gnn_explainer/
```

Expected: zero matches (PR2 closes the GLM re-fit branch).

- [ ] **Step 4: Smoke-import every module the local venv can handle**

```bash
.venv/bin/python -c "
from src.training.fold_checkpoint import (
    CheckpointManager, FoldCheckpoint, FoldBundle,
)
from src.training.fold_barrier import FoldBarrier
from src.training.label_normalizer import LabelNormalizer
from src.training.glm_normalizer import GLMFeatureNormalizer
from src.datasets.label_builder import LabelBuilder
print('all imports ok')
"
```

Expected: `all imports ok`.

(`Trainer`, `NestedCrossValidator`, `Finetuner`, `GNNExplainerRunner`, and the model registry all import `src.models` which requires `torch_sparse` — pre-existing local-venv gap, same caveat as PR1. Confirm those modules are unchanged in shape with `git diff main -- <module>`.)

- [ ] **Step 5: Run the new test files**

```bash
.venv/bin/python -m pytest tests/test_fold_checkpoint.py tests/test_fold_barrier.py -v
```

Expected: every test PASS.

- [ ] **Step 6: If `torch_sparse` is available, run the affected integration tests**

```bash
.venv/bin/python -m pytest tests/test_gnn_explainer.py tests/test_training.py::TestNestedCrossValidator -v 2>&1 | tail -30
```

- If collection fails on `torch_sparse`: the local-venv gap stands. Note in the PR description that this gate must be re-run on the cluster.
- If it runs, every test should PASS. The new `TestExplainerConsumesPersistedBarrier::test_explainer_uses_persisted_glm_stats` is the load-bearing PR2 regression.

- [ ] **Step 7: No commit (verification only).**

---

## Task 10: Push branch and open PR

**Files:** none.

- [ ] **Step 1: Push**

```bash
git push -u origin feature/fold-checkpoint
```

- [ ] **Step 2: Decide PR base**

PR1 (`feature/fold-barrier`, #12) was merge-ready but awaiting a cluster smoke gate at the time PR2 started. Check whether PR1 has merged to `main`:

```bash
gh pr view 12 --json state,merged --jq '{state, merged}'
```

- If `merged: true` → PR2 bases on `main`.
- If `merged: false` → PR2 bases on `feature/fold-barrier` (so the PR diff only contains PR2's commits, not PR1's).

- [ ] **Step 3: Open PR**

For the **PR1-already-merged** case:

```bash
gh pr create --base main --head feature/fold-checkpoint \
  --title "feat(training): FoldCheckpoint + Explainer barrier-reload (ADR-0009 PR2)" \
  --body "$(cat <<'EOF'
## Summary

- Renames `src/training/checkpoint_manager.py` → `src/training/fold_checkpoint.py` and the pre-existing `FoldCheckpoint` dataclass → `FoldBundle`, freeing the name `FoldCheckpoint` for a new per-fold view class.
- New `FoldCheckpoint(fold_dir)` is the single per-fold read path: `load_state_dict`, `load_model`, `load_barrier`, `load_metrics`, `load_bundle`. Owns the barrier reload — autoconfigures from the persisted state alone (the GLM substate carries `(col_start, col_end)`; the label-norm substate carries its strategy).
- `CheckpointManager` keeps the *write-side* and multi-fold operations (`save_fold_checkpoint`, the epoch helpers, `get_common_epochs`). Its `load_fold_checkpoint` and `load_model_for_fold` are removed.
- `GNNExplainerRunner._explain_one_fold` replaces its `GLMFeatureNormalizer.fit(...).transform(...)` block with `FoldCheckpoint(fold_ckpt_dir).load_barrier()` + `barrier.transform_graphs(test_graphs)`. The PR1 dead-letter is closed: `barrier.pt` is now actually consumed.
- `Finetuner._load_weights_for_fold` switches to `FoldCheckpoint.for_fold(...).load_state_dict(variant=...)` — a direct read instead of `load_fold_checkpoint(...).best_model_state_dict`.
- **B-fresh** — no compat shim for the old module path; old checkpoints without `barrier.pt` fall through with a warning and raw test graphs.

**Known gap (out of scope for PR2):** composite-mode label reload is still caller-supplied (`FoldBarrier.load(path, label_builder=...)`). The Explainer never touches labels, so this doesn't block PR2; documented in `FoldCheckpoint.load_barrier`'s docstring.

## Test plan

- [ ] `tests/test_fold_checkpoint.py` — round-trip for the new view class, including barrier reload from a fitted barrier.pt.
- [ ] `tests/test_gnn_explainer.py::TestExplainerConsumesPersistedBarrier::test_explainer_uses_persisted_glm_stats` — sentinel GLM stats in barrier.pt survive to the model's input (regression on PR1's dead-letter).
- [ ] Existing `tests/test_gnn_explainer.py` nested/legacy/rep-restriction tests still pass — the fixture now writes a real barrier.pt and the explainer reads it.
- [ ] `tests/test_training.py::TestNestedCrossValidator` smoke suite green on the cluster.
- [ ] `grep -r "load_fold_checkpoint\|load_model_for_fold" src/ tests/` returns zero matches.
- [ ] `grep -n "GLMFeatureNormalizer\|glm_normalizer" src/gnn_explainer/` returns zero matches.

ADR: [`docs/adr/0009-fold-barrier.md`](docs/adr/0009-fold-barrier.md)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

For the **PR1-not-yet-merged** case, change `--base main` to `--base feature/fold-barrier`, and add a `> Stacked on top of #12` line at the top of the body.

- [ ] **Step 4: Report PR URL to user.**
