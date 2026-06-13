# PNC Age→VWM Transfer Infrastructure — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe **nested-CV transfer** (age-pretrained backbone → VWM) to the existing GNN pipeline, plus the A4/A5 baselines — unlocking thesis arms B1–B4.

**Architecture:** Reuse the existing `NestedCrossValidator` and its per-(rep,fold) checkpointing. Two age "source" runs (ID-carrier, GLM-carrier), stratified on VWM so their outer folds match every VWM run byte-for-byte, emit per-fold backbones. A `SourceBackboneProvider` injects those backbones into the VWM run by wrapping the per-fold `model_factory`, after asserting fold-index alignment from a persisted manifest. Transfer HPO is restricted to head/optimizer knobs (backbone shape is frozen by the loaded weights). Frozen-vs-finetune is an arm, set by `frozen_layers`.

**Tech Stack:** PyTorch, PyTorch-Geometric, Optuna, Hydra + Pydantic configs, pytest. Companion spec: `docs/superpowers/specs/2026-06-13-pnc-age-vwm-transfer-design.md`.

---

## File structure

| File | Responsibility | New/Modify |
|------|----------------|-----------|
| `src/training/transfer_ops.py` | `freeze_layers`, `reinit_head` — one impl for flat + nested transfer | **New** |
| `src/finetuning/finetuner.py` | refactor `_freeze_layers`/`_reinit_head` to call `transfer_ops` | Modify |
| `src/training/trainer.py` | `_build_optimizer` filters `requires_grad=False` params | Modify (`:463`) |
| `src/datasets/base_dataset.py` | `get_label_column(name)` — vector for any metadata column | Modify |
| `src/training/nested_cross_validation.py` | `stratify_labels` arg; write `fold_indices.json`; optional `source_provider`/`frozen_layers`; `_wrap_factory_for_fold` | Modify |
| `src/finetuning/transfer_nested.py` | `SourceBackboneProvider` — load source manifest + per-fold weights, assert alignment | **New** |
| `src/configs/transfer_config.py` | `TransferConfig` pydantic schema | **New** |
| `configs/transfer/{none,from_age_ft,from_age_frozen}.yaml` | transfer config presets | **New** |
| `configs/sweeper/transfer_finetune.yaml` | restricted HPO search space | **New** |
| `configs/experiment.yaml` | add `transfer: none` default; add `stratify_target: null` | Modify |
| `scripts/run_experiment.py` | wire `stratify_target` + transfer provider into the nested route | Modify |
| `tests/test_transfer_ops.py`, `tests/test_transfer_nested.py`, `tests/test_splits_stratify_override.py`, `tests/test_nested_transfer_integration.py` | tests | **New** |

**Source-checkpoint on-disk layout** (already produced by nested CV at `nested_cross_validation.py:404,504`): `<root>/rep_<R>/fold_<F>/model_best.pt` + `model_config.json` + `feature_config.json`. This plan adds `<root>/fold_indices.json`.

---

## Task 1: Shared `transfer_ops` module (lift from finetuner)

**Files:**
- Create: `src/training/transfer_ops.py`
- Modify: `src/finetuning/finetuner.py:562-581`
- Test: `tests/test_transfer_ops.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_ops.py
import torch
import torch.nn as nn
from src.training.transfer_ops import freeze_layers, reinit_head


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def test_freeze_layers_freezes_only_matching_prefixes():
    m = _Tiny()
    n = freeze_layers(m, ["backbone"])
    assert n == 2  # weight + bias
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())


def test_freeze_layers_empty_list_is_noop():
    m = _Tiny()
    assert freeze_layers(m, []) == 0
    assert all(p.requires_grad for p in m.parameters())


def test_reinit_head_changes_head_weights_only():
    m = _Tiny()
    torch.nn.init.constant_(m.head[0].weight, 0.123)
    backbone_before = m.backbone.weight.detach().clone()
    reinit_head(m)
    assert not torch.allclose(m.head[0].weight, torch.full_like(m.head[0].weight, 0.123))
    assert torch.allclose(m.backbone.weight, backbone_before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: src.training.transfer_ops`.

- [ ] **Step 3: Write the module**

```python
# src/training/transfer_ops.py
"""Backbone-transfer primitives shared by the flat and nested fine-tuning paths."""

from __future__ import annotations

import logging
from typing import List

import torch

log = logging.getLogger(__name__)


def reinit_head(model: torch.nn.Module) -> None:
    """Reinitialise every ``nn.Linear`` inside ``model.head`` (Kaiming + zero bias)."""
    if not hasattr(model, "head"):
        return
    for m in model.head.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    log.info("Reinitialised prediction head")


def freeze_layers(model: torch.nn.Module, frozen_prefixes: List[str]) -> int:
    """Set ``requires_grad=False`` on params whose name starts with any prefix.

    Returns the number of parameters frozen.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_prefixes):
            param.requires_grad = False
            frozen += 1
    if frozen:
        log.info("Froze %d parameters matching prefixes: %s", frozen, frozen_prefixes)
    return frozen
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_ops.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Refactor finetuner to delegate (no behaviour change)**

In `src/finetuning/finetuner.py`, replace the module-level `_reinit_head` and `_freeze_layers` bodies (lines 562-581) with delegations, keeping the private names as aliases so existing call sites are untouched:

```python
# src/finetuning/finetuner.py  (replace the two functions at the bottom)
from src.training.transfer_ops import freeze_layers as _freeze_layers_impl
from src.training.transfer_ops import reinit_head as _reinit_head_impl


def _reinit_head(model: BrainGNN) -> None:
    _reinit_head_impl(model)


def _freeze_layers(model: BrainGNN, frozen_prefixes: List[str]) -> None:
    _freeze_layers_impl(model, frozen_prefixes)
```

(Add the two imports near the top with the other imports; delete the old function bodies.)

- [ ] **Step 6: Run finetuner tests to verify no regression**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/ -k "finetun or transfer_ops" -v`
Expected: PASS (no failures introduced).

- [ ] **Step 7: Commit**

```bash
git add src/training/transfer_ops.py src/finetuning/finetuner.py tests/test_transfer_ops.py
git commit -m "refactor: extract freeze_layers/reinit_head into src/training/transfer_ops"
```

---

## Task 2: Optimizer skips frozen params

**Files:**
- Modify: `src/training/trainer.py:451-463`
- Test: `tests/test_transfer_ops.py` (extend)

The nested path builds `Trainer` without param-groups, so frozen arms (B2/B4) need `_build_optimizer` to exclude `requires_grad=False` params.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_ops.py  (append)
from src.configs.trainer_config import TrainerConfig
from src.training.trainer import Trainer


def test_build_optimizer_excludes_frozen_params():
    m = _Tiny()
    freeze_layers(m, ["backbone"])
    trainer = Trainer(cfg=TrainerConfig(), logger=_NullLogger())
    opt = trainer._build_optimizer(m)
    opt_param_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    assert all(id(p) not in opt_param_ids for p in m.backbone.parameters())
    assert all(id(p) in opt_param_ids for p in m.head.parameters())


class _NullLogger:
    class _Cfg:
        enabled = False
    def __init__(self):
        self.cfg = self._Cfg()
    def __getattr__(self, name):
        return lambda *a, **k: None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_ops.py::test_build_optimizer_excludes_frozen_params -v`
Expected: FAIL — backbone params currently included.

- [ ] **Step 3: Patch `_build_optimizer`**

In `src/training/trainer.py:463`, change the `params` line:

```python
# before:
#   params = self._param_groups if self._param_groups is not None else model.parameters()
# after:
        params = (
            self._param_groups
            if self._param_groups is not None
            else [p for p in model.parameters() if p.requires_grad]
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_ops.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — a normal (no-freeze) model still trains all params**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/ -k "trainer" -v`
Expected: PASS (full-param models unaffected — every param has `requires_grad=True`).

- [ ] **Step 6: Commit**

```bash
git add src/training/trainer.py tests/test_transfer_ops.py
git commit -m "fix(trainer): _build_optimizer excludes frozen (requires_grad=False) params"
```

---

## Task 3: `get_label_column` on the dataset

**Files:**
- Modify: `src/datasets/base_dataset.py` (near `get_labels`, ~line 270-283)
- Test: `tests/test_get_label_column.py`

Needed so a source age run can stratify on the VWM column without re-loading graphs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_get_label_column.py
import numpy as np
from src.datasets.base_dataset import BaseDataset


def test_get_label_column_builds_vector_for_named_column(monkeypatch):
    ds = BaseDataset.__new__(BaseDataset)  # bypass __init__
    ds._subject_ids = ["s1", "s2", "s3"]
    ds._metadata = {
        "s1": {"age": 10.0, "vwm": 0.5},
        "s2": {"age": 14.0, "vwm": 1.5},
        "s3": {"age": 18.0, "vwm": -0.2},
    }
    vec = ds.get_label_column("vwm")
    assert isinstance(vec, np.ndarray)
    assert vec.tolist() == [0.5, 1.5, -0.2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_get_label_column.py -v`
Expected: FAIL — `AttributeError: get_label_column`.

- [ ] **Step 3: Implement `get_label_column`**

Add to `src/datasets/base_dataset.py` right after `get_labels` (uses the same `LabelBuilder` machinery `get_labels` relies on):

```python
    def get_label_column(self, column: str) -> np.ndarray:
        """Build a per-subject vector for an arbitrary metadata column.

        Reuses the LabelBuilder single-target path so binning/ordering match
        ``get_labels``. Used to stratify a run on a column other than its
        regression target (e.g. age-source runs stratified on VWM).
        """
        from src.configs.label_config import LabelConfig
        from src.datasets.label_builder import LabelBuilder

        builder = LabelBuilder(LabelConfig(target=column))
        return builder.build(self._subject_ids, self._metadata)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_get_label_column.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/datasets/base_dataset.py tests/test_get_label_column.py
git commit -m "feat(dataset): get_label_column(name) for cross-column stratification"
```

---

## Task 4: Stratification override + fold-index manifest in nested CV

**Files:**
- Modify: `src/training/nested_cross_validation.py` (`run`, ~line 206-349)
- Test: `tests/test_splits_stratify_override.py`

Two changes: (a) `run(stratify_labels=None)` bins folds on that vector instead of `labels`; (b) every nested run writes `fold_indices.json` under `ckpt_root`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits_stratify_override.py
import json
import numpy as np
from src.training.splits import stratify_bins, outer_folds


def test_same_bins_same_folds_diff_bins_diff_folds():
    n, seeds, n_outer = 60, [0, 1], 5
    rng = np.random.default_rng(0)
    age = rng.normal(size=n)
    vwm = rng.normal(size=n)
    bins_age = stratify_bins(age, 4)
    bins_vwm = stratify_bins(vwm, 4)
    folds_age = outer_folds(n=n, bins=bins_age, seeds=seeds, n_outer=n_outer)
    folds_vwm = outer_folds(n=n, bins=bins_vwm, seeds=seeds, n_outer=n_outer)
    folds_vwm2 = outer_folds(n=n, bins=bins_vwm, seeds=seeds, n_outer=n_outer)
    # identical bins -> identical folds (determinism)
    assert [t for t, _ in folds_vwm] == [t for t, _ in folds_vwm2]
    # different bins -> at least one fold differs (proves the override is necessary)
    assert [s for _, s in folds_age] != [s for _, s in folds_vwm]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_splits_stratify_override.py -v`
Expected: PASS already (this asserts existing `splits.py` behaviour — it documents *why* the override matters and guards against future regressions). Keep it.

- [ ] **Step 3: Add `stratify_labels` to `NestedCrossValidator.run`**

In `src/training/nested_cross_validation.py`, add the keyword to the `run` signature (after `labels`):

```python
        labels: np.ndarray,
        stratify_labels: Optional[np.ndarray] = None,
```

Change the binning line (currently `bins = self._stratify_bins(labels)` at ~line 293):

```python
        # Bin on stratify_labels when provided (e.g. age-source runs stratified
        # on VWM so their outer folds match every VWM run), else on the target.
        strat = stratify_labels if stratify_labels is not None else labels
        if stratify_labels is not None and len(stratify_labels) != len(labels):
            raise ValueError(
                f"stratify_labels length {len(stratify_labels)} != labels length {len(labels)}"
            )
        bins = self._stratify_bins(strat)
```

- [ ] **Step 4: Write the fold-index manifest after `all_folds` is built**

Immediately after `all_folds = _shared_outer_folds(...)` (~line 295-297) and before the fold loop, write the manifest:

```python
        manifest = {
            "n": len(dataset),
            "n_outer": n_outer,
            "n_repetitions": self.cfg.n_repetitions,
            "outer_seeds": outer_seeds,
            "folds": [
                {
                    "rep": fi // n_outer,
                    "fold": fi % n_outer,
                    "train_val_idx": sorted(map(int, tv)),
                    "test_idx": sorted(map(int, te)),
                }
                for fi, (tv, te) in enumerate(all_folds)
            ],
        }
        ckpt_root.mkdir(parents=True, exist_ok=True)
        with open(ckpt_root / "fold_indices.json", "w") as f:
            json.dump(manifest, f)
        log.info("Wrote fold-index manifest: %s", ckpt_root / "fold_indices.json")
```

(`json` is already imported at the top of the module.)

- [ ] **Step 5: Integration check — manifest is written and self-consistent**

```python
# tests/test_splits_stratify_override.py  (append)
def test_manifest_roundtrip_shape(tmp_path):
    # minimal manifest contract used by SourceBackboneProvider
    manifest = {
        "folds": [
            {"rep": 0, "fold": 0, "train_val_idx": [1, 2], "test_idx": [0]},
        ]
    }
    p = tmp_path / "fold_indices.json"
    p.write_text(json.dumps(manifest))
    loaded = json.loads(p.read_text())
    rec = loaded["folds"][0]
    assert (rec["rep"], rec["fold"]) == (0, 0)
    assert rec["train_val_idx"] == [1, 2]
```

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_splits_stratify_override.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/training/nested_cross_validation.py tests/test_splits_stratify_override.py
git commit -m "feat(nested-cv): stratify_labels override + fold_indices.json manifest"
```

---

## Task 5: `SourceBackboneProvider`

**Files:**
- Create: `src/finetuning/transfer_nested.py`
- Test: `tests/test_transfer_nested.py`

Loads the source run's manifest + per-(rep,fold) weights; asserts the consuming run's fold indices match.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_nested.py
import json
import pytest
import torch
from src.finetuning.transfer_nested import SourceBackboneProvider


def _make_source(tmp_path):
    root = tmp_path / "src_ckpt"
    (root / "rep_0" / "fold_0").mkdir(parents=True)
    torch.save({"backbone.weight": torch.ones(2, 2)},
               root / "rep_0" / "fold_0" / "model_best.pt")
    manifest = {"folds": [
        {"rep": 0, "fold": 0, "train_val_idx": [1, 2, 3], "test_idx": [0]},
    ]}
    (root / "fold_indices.json").write_text(json.dumps(manifest))
    return root


def test_assert_aligned_passes_on_match(tmp_path):
    root = _make_source(tmp_path)
    p = SourceBackboneProvider(root)
    p.assert_aligned(rep=0, fold=0, train_val_idx=[3, 1, 2], test_idx=[0])  # order-independent


def test_assert_aligned_raises_on_mismatch(tmp_path):
    root = _make_source(tmp_path)
    p = SourceBackboneProvider(root)
    with pytest.raises(ValueError, match="fold-index mismatch"):
        p.assert_aligned(rep=0, fold=0, train_val_idx=[1, 2], test_idx=[0, 3])


def test_state_dict_for_loads_weights(tmp_path):
    root = _make_source(tmp_path)
    p = SourceBackboneProvider(root)
    sd = p.state_dict_for(rep=0, fold=0)
    assert "backbone.weight" in sd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_nested.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the provider**

```python
# src/finetuning/transfer_nested.py
"""Inject per-(rep,fold) age-pretrained backbones into a nested-CV VWM run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

log = logging.getLogger(__name__)


class SourceBackboneProvider:
    """Serve source (age) backbone weights for each outer fold, alignment-checked.

    Parameters
    ----------
    source_root : str | Path
        Root of a completed source nested run: holds ``fold_indices.json`` and
        ``rep_<R>/fold_<F>/model_<variant>.pt``.
    variant : str
        ``"best"`` or ``"last"``.
    """

    def __init__(self, source_root, variant: str = "best") -> None:
        self.source_root = Path(source_root)
        self.variant = variant
        manifest_path = self.source_root / "fold_indices.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Source run has no fold_indices.json at {manifest_path}; "
                "regenerate the source run with the manifest-writing nested CV."
            )
        manifest = json.loads(manifest_path.read_text())
        self._folds: Dict[Tuple[int, int], Dict[str, List[int]]] = {
            (rec["rep"], rec["fold"]): {
                "train_val_idx": sorted(rec["train_val_idx"]),
                "test_idx": sorted(rec["test_idx"]),
            }
            for rec in manifest["folds"]
        }

    def assert_aligned(
        self, *, rep: int, fold: int, train_val_idx, test_idx
    ) -> None:
        """Raise unless the consuming run's fold (rep,fold) matches the source."""
        key = (rep, fold)
        if key not in self._folds:
            raise ValueError(
                f"fold-index mismatch: source manifest has no (rep={rep}, fold={fold})"
            )
        src = self._folds[key]
        if (
            sorted(map(int, train_val_idx)) != src["train_val_idx"]
            or sorted(map(int, test_idx)) != src["test_idx"]
        ):
            raise ValueError(
                f"fold-index mismatch at (rep={rep}, fold={fold}): the VWM run's "
                "outer split differs from the age-source run's. Stratification is "
                "not aligned — refusing to transfer (would leak test subjects)."
            )

    def state_dict_for(self, *, rep: int, fold: int) -> dict:
        """Load the source backbone weights for one outer fold."""
        path = self.source_root / f"rep_{rep}" / f"fold_{fold}" / f"model_{self.variant}.pt"
        return torch.load(path, map_location="cpu", weights_only=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_nested.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/finetuning/transfer_nested.py tests/test_transfer_nested.py
git commit -m "feat(transfer): SourceBackboneProvider with fold-alignment guard"
```

---

## Task 6: Wire the provider into nested CV's per-fold factory

**Files:**
- Modify: `src/training/nested_cross_validation.py` (`__init__` ~line 191-200; `_run_outer_fold` ~line 355-416)
- Test: `tests/test_nested_transfer_integration.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_nested_transfer_integration.py
import torch
import torch.nn as nn
from src.finetuning.transfer_nested import SourceBackboneProvider
from src.training.nested_cross_validation import NestedCrossValidator
from src.training.transfer_ops import freeze_layers, reinit_head


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.head = nn.Linear(3, 1)


def test_wrap_factory_loads_freezes_and_reinits(tmp_path, monkeypatch):
    # Build a fake source provider whose weights are all 7.0 on the backbone.
    root = tmp_path / "src"
    (root / "rep_0" / "fold_2").mkdir(parents=True)
    sd = {"backbone.weight": torch.full((3, 3), 7.0), "backbone.bias": torch.full((3,), 7.0)}
    torch.save(sd, root / "rep_0" / "fold_2" / "model_best.pt")
    import json
    (root / "fold_indices.json").write_text(json.dumps(
        {"folds": [{"rep": 0, "fold": 2, "train_val_idx": [0, 1], "test_idx": [2]}]}
    ))
    provider = SourceBackboneProvider(root)

    from src.configs.trainer_config import TrainerConfig
    ncv = NestedCrossValidator(cfg=TrainerConfig(), source_provider=provider,
                               frozen_layers=["backbone"])
    base_factory = lambda cfg=None: _Tiny()
    wrapped = ncv._wrap_factory_for_fold(
        base_factory, rep=0, fold=2, train_val_idx=[1, 0], test_idx=[2],
    )
    model = wrapped(None)
    # backbone loaded from source (==7.0) and frozen; head reinit'd & trainable
    assert torch.allclose(model.backbone.weight, torch.full((3, 3), 7.0))
    assert not model.backbone.weight.requires_grad
    assert model.head.weight.requires_grad


def test_wrap_factory_raises_on_misalignment(tmp_path):
    import json
    root = tmp_path / "src"
    (root / "rep_0" / "fold_0").mkdir(parents=True)
    torch.save({"backbone.weight": torch.zeros(3, 3)},
               root / "rep_0" / "fold_0" / "model_best.pt")
    (root / "fold_indices.json").write_text(json.dumps(
        {"folds": [{"rep": 0, "fold": 0, "train_val_idx": [0, 1], "test_idx": [2]}]}
    ))
    from src.configs.trainer_config import TrainerConfig
    ncv = NestedCrossValidator(cfg=TrainerConfig(),
                               source_provider=SourceBackboneProvider(root),
                               frozen_layers=[])
    wrapped = ncv._wrap_factory_for_fold(
        lambda cfg=None: _Tiny(), rep=0, fold=0, train_val_idx=[0, 9], test_idx=[2],
    )
    import pytest
    with pytest.raises(ValueError, match="fold-index mismatch"):
        wrapped(None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_nested_transfer_integration.py -v`
Expected: FAIL — `__init__` rejects `source_provider`; `_wrap_factory_for_fold` missing.

- [ ] **Step 3: Extend `__init__`**

In `src/training/nested_cross_validation.py`, add params to `NestedCrossValidator.__init__` (after `search_space_path`):

```python
    def __init__(
        self,
        cfg: TrainerConfig,
        search_space_path: Optional[str | Path] = None,
        source_provider: Optional["SourceBackboneProvider"] = None,
        frozen_layers: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.search_space_path = (
            Path(search_space_path) if search_space_path is not None else None
        )
        self._search_specs: Optional[List[SearchSpec]] = None
        self.source_provider = source_provider
        self.frozen_layers = frozen_layers or []
```

Add the import near the top:

```python
from src.finetuning.transfer_nested import SourceBackboneProvider  # noqa: F401  (typing)
```

- [ ] **Step 4: Add `_wrap_factory_for_fold`**

Add this method to `NestedCrossValidator` (e.g. just above `_run_outer_fold`):

```python
    def _wrap_factory_for_fold(
        self, model_factory, *, rep: int, fold: int, train_val_idx, test_idx,
    ):
        """Return a factory that loads the source backbone for (rep,fold).

        No-op (returns ``model_factory``) when no source provider is set.
        """
        if self.source_provider is None:
            return model_factory

        from src.interfaces.adapters import load_partial_state_dict
        from src.training.transfer_ops import freeze_layers, reinit_head

        provider = self.source_provider
        frozen = self.frozen_layers

        def transfer_factory(trial_model_cfg):
            provider.assert_aligned(
                rep=rep, fold=fold, train_val_idx=train_val_idx, test_idx=test_idx,
            )
            model = model_factory(trial_model_cfg)
            state_dict = provider.state_dict_for(rep=rep, fold=fold)
            loaded, _skipped = load_partial_state_dict(model, state_dict)
            backbone_keys = [k for k in model.state_dict() if k.startswith("backbone")]
            missing = [k for k in backbone_keys if k not in loaded]
            if backbone_keys and missing:
                raise RuntimeError(
                    f"Source backbone did not fully load at (rep={rep}, fold={fold}): "
                    f"{len(missing)}/{len(backbone_keys)} backbone keys unmatched "
                    f"(e.g. {missing[:3]}). The VWM model's backbone shape must equal "
                    "the source run's — pin the source architecture (spec §10)."
                )
            reinit_head(model)
            freeze_layers(model, frozen)
            return model

        return transfer_factory
```

- [ ] **Step 5: Use the wrapped factory inside `_run_outer_fold`**

In `_run_outer_fold`, immediately after `train_idx, val_idx = self._inner_split(...)` (~line 378-380), insert:

```python
        fold_factory = self._wrap_factory_for_fold(
            model_factory, rep=rep, fold=fold,
            train_val_idx=train_val_idx, test_idx=test_idx,
        )
```

Then replace the two downstream `model_factory=model_factory` arguments — in the `_run_inner_hpo(...)` call (~line 408-416) and both `_refit_on_trainval(...)` calls (~line 456-463 and 471-478) — with `model_factory=fold_factory`. (In the `inner_hpo_trials == 0` branch, also pass `model_factory=fold_factory` to `_train_inner_trial`.)

- [ ] **Step 6: Run the integration test to verify it passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_nested_transfer_integration.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Regression — non-transfer nested CV is unchanged**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/ -k "nested or sklearn_nested" -v`
Expected: PASS (provider defaults to `None` → `_wrap_factory_for_fold` returns the base factory).

- [ ] **Step 8: Commit**

```bash
git add src/training/nested_cross_validation.py tests/test_nested_transfer_integration.py
git commit -m "feat(nested-cv): per-fold source-backbone injection with alignment guard"
```

---

## Task 7: `TransferConfig` + config presets

**Files:**
- Create: `src/configs/transfer_config.py`
- Create: `configs/transfer/none.yaml`, `configs/transfer/from_age_ft.yaml`, `configs/transfer/from_age_frozen.yaml`
- Modify: `configs/experiment.yaml`
- Test: `tests/test_transfer_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_config.py
import pytest
from src.configs.transfer_config import TransferConfig


def test_disabled_by_default():
    assert TransferConfig().enabled is False


def test_enabled_requires_source_root():
    with pytest.raises(ValueError, match="source_checkpoint_root"):
        TransferConfig(enabled=True).validate_runtime()


def test_frozen_arm_roundtrip():
    c = TransferConfig(enabled=True, source_checkpoint_root="ck/age_id",
                       frozen_layers=["backbone"])
    c.validate_runtime()
    assert c.frozen_layers == ["backbone"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_config.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the schema**

```python
# src/configs/transfer_config.py
"""Nested-CV transfer config (age-pretrained backbone → VWM)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TransferConfig(BaseModel):
    """Inject a per-fold source backbone into a nested-CV run.

    Distinct from FinetuningConfig (flat CV). When ``enabled``, the run stays
    on the nested runner but loads source weights per outer fold.
    """

    enabled: bool = Field(default=False)
    source_checkpoint_root: Optional[str] = Field(
        default=None,
        description="Root of a completed source nested run (holds fold_indices.json).",
    )
    checkpoint_variant: Literal["best", "last"] = Field(default="best")
    frozen_layers: List[str] = Field(
        default_factory=list,
        description="Param-name prefixes to freeze. [] = full fine-tune (B1/B3); "
        "['backbone'] = head-only (B2/B4).",
    )

    def validate_runtime(self) -> None:
        if self.enabled and not self.source_checkpoint_root:
            raise ValueError(
                "transfer.enabled=true requires transfer.source_checkpoint_root"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_config.py -v`
Expected: PASS.

- [ ] **Step 5: Write the YAML presets**

```yaml
# configs/transfer/none.yaml
enabled: false
source_checkpoint_root: null
checkpoint_variant: best
frozen_layers: []
```

```yaml
# configs/transfer/from_age_ft.yaml   (B1/B3: full fine-tune)
enabled: true
source_checkpoint_root: ???   # set on CLI: transfer.source_checkpoint_root=checkpoints/<source-run-name>
checkpoint_variant: best
frozen_layers: []
```

```yaml
# configs/transfer/from_age_frozen.yaml   (B2/B4: head-only)
enabled: true
source_checkpoint_root: ???
checkpoint_variant: best
frozen_layers:
  - backbone
```

- [ ] **Step 6: Add to `configs/experiment.yaml` defaults**

In the `defaults:` list add `- transfer: none`, and at top level add `stratify_target: null`.

- [ ] **Step 7: Commit**

```bash
git add src/configs/transfer_config.py configs/transfer/ configs/experiment.yaml tests/test_transfer_config.py
git commit -m "feat(config): TransferConfig + transfer/{none,from_age_ft,from_age_frozen} presets"
```

---

## Task 8: Wire transfer + stratify_target into the runner

**Files:**
- Modify: `scripts/run_experiment.py` (config parse ~line 129-141; label load ~line 256; nested route ~line 537-572)
- Test: covered by Task 10 smoke run (runner wiring is glue; unit-tested pieces are Tasks 4-7).

- [ ] **Step 1: Parse `TransferConfig` alongside the others**

Near `ft_cfg = FinetuningConfig(...)` (~line 141), add:

```python
    from src.configs.transfer_config import TransferConfig
    transfer_cfg = TransferConfig(**OmegaConf.to_container(cfg.transfer, resolve=True))
    transfer_cfg.validate_runtime()
```

- [ ] **Step 2: Build the stratification vector when requested**

After `labels: np.ndarray = dataset.get_labels()` (~line 256), add:

```python
    stratify_labels = None
    stratify_target = cfg.get("stratify_target")
    if stratify_target:
        stratify_labels = dataset.get_label_column(stratify_target)
        log.info("Stratifying folds on '%s' (target remains '%s')",
                 stratify_target, label_cfg.target)
```

- [ ] **Step 3: Pass provider + stratify_labels into the nested route**

In the nested branch (~line 537-572), build the provider and thread both args:

```python
        source_provider = None
        if transfer_cfg.enabled:
            from src.finetuning.transfer_nested import SourceBackboneProvider
            source_provider = SourceBackboneProvider(
                transfer_cfg.source_checkpoint_root,
                variant=transfer_cfg.checkpoint_variant,
            )
            log.info("Transfer enabled — source=%s frozen=%s",
                     transfer_cfg.source_checkpoint_root, transfer_cfg.frozen_layers)

        ncv = NestedCrossValidator(
            cfg=trainer_cfg, search_space_path=trainer_cfg.search_space,
            source_provider=source_provider, frozen_layers=transfer_cfg.frozen_layers,
        )
        ...
        nested_result = ncv.run(
            model_factory=nested_model_factory,
            dataset=graphs,
            labels=labels,
            stratify_labels=stratify_labels,
            base_model_cfg=model_cfg,
            ...
        )
```

- [ ] **Step 4: Smoke-import the runner**

Run: `PYTHONPATH=$(pwd) .venv/bin/python -c "import scripts.run_experiment"`
Expected: no ImportError.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat(runner): wire transfer provider + stratify_target into nested route"
```

---

## Task 9: Restricted transfer sweeper + source-run pinned config

**Files:**
- Create: `configs/sweeper/transfer_finetune.yaml`
- Create: `configs/sweeper/source_age_pinned.yaml`
- Test: `tests/test_transfer_sweeper.py`

The transfer sweeper must **not** include backbone-architecture params (backbone shape is frozen by the loaded weights). The source-run sweeper must pin the backbone (single shape across folds, spec §10).

- [ ] **Step 1: Write the failing test (guards the no-backbone-arch rule)**

```python
# tests/test_transfer_sweeper.py
from pathlib import Path
from src.training.search_space import load_sweeper_params, parse_search_space

_FORBIDDEN = {"model.embedding_dim", "model.hidden_dim", "model.num_layers",
              "model.pooling", "model.jk_mode"}


def test_transfer_sweeper_has_no_backbone_arch_params():
    params = load_sweeper_params(Path("configs/sweeper/transfer_finetune.yaml"))
    specs = parse_search_space(params)
    names = {s.name for s in specs}
    assert not (names & _FORBIDDEN), f"backbone-arch params leak into transfer HPO: {names & _FORBIDDEN}"
    # must still tune optimiser + head
    assert "trainer.lr" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_sweeper.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Write the transfer sweeper**

```yaml
# configs/sweeper/transfer_finetune.yaml
# @package hydra.sweeper
# Restricted HPO for transfer arms: backbone shape is frozen by the loaded
# source weights, so ONLY optimiser + the reinit'd head are tuned.
sampler:
  _target_: optuna.samplers.TPESampler
  seed: 42
direction: maximize
n_trials: 20
params:
  trainer.lr: choice(0.0001, 0.0005, 0.001, 0.005)
  trainer.weight_decay: choice(0.0, 0.0001, 0.001, 0.01)
  model.dropout: choice(0.0, 0.1, 0.2, 0.3)
  model.head_hidden_dim: choice(16, 32, 64, 128)
  model.head_num_layers: choice(2)
```

- [ ] **Step 4: Write the pinned source sweeper**

```yaml
# configs/sweeper/source_age_pinned.yaml
# @package hydra.sweeper
# Source age pretraining: pin the backbone architecture so EVERY fold's refit
# checkpoint shares one shape (transfer requires identical backbone dims).
# Only optimiser + head vary during the source run's own HPO.
sampler:
  _target_: optuna.samplers.TPESampler
  seed: 42
direction: maximize
n_trials: 20
params:
  trainer.lr: choice(0.0001, 0.0005, 0.001, 0.005)
  trainer.weight_decay: choice(0.0, 0.0001, 0.001, 0.01)
  model.dropout: choice(0.0, 0.1, 0.2, 0.3)
  model.head_hidden_dim: choice(16, 32, 64, 128)
  model.head_num_layers: choice(2)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/test_transfer_sweeper.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/sweeper/transfer_finetune.yaml configs/sweeper/source_age_pinned.yaml tests/test_transfer_sweeper.py
git commit -m "feat(config): restricted transfer + pinned-source HPO sweepers"
```

---

## Task 10: End-to-end smoke run (fast preset) + runbook

**Files:**
- Create: `docs/runbooks/age-vwm-transfer.md`
- No test file — this is a CLI smoke check with the fast preset (1 rep × 2 folds × 0 trials).

- [ ] **Step 1: Generate a tiny source age run (ID carrier), stratified on VWM**

Use the fast preset (set `trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=0` to keep it seconds, not hours) on whatever local PNC subset is available. Record the run name (= `<experiment_name>-<uid>`) and confirm `checkpoints/<run-name>/fold_indices.json` and `checkpoints/<run-name>/rep_0/fold_0/model_best.pt` exist.

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc model=gcn features=identity \
  labels.target=<AGE_COL> stratify_target=<VWM_COL> \
  experiment_name=src_age_id_smoke \
  trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=0 \
  logging.enabled=false
```

Expected: log line `Wrote fold-index manifest: .../fold_indices.json`.

- [ ] **Step 2: Run the VWM transfer arm (B2 frozen) off that source**

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc model=gcn features=identity \
  labels.target=<VWM_COL> \
  transfer=from_age_frozen \
  transfer.source_checkpoint_root=checkpoints/src_age_id_smoke-<uid> \
  experiment_name=b2_smoke \
  trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=0 \
  logging.enabled=false
```

Expected: completes; logs `Transfer enabled — source=...`; no `fold-index mismatch`; outer-test metrics printed. (The fast preset stratifies VWM the same way the source did, so the guard passes.)

- [ ] **Step 3: Verify the guard fires on deliberate misalignment**

Re-run Step 2 but point at a source generated **without** `stratify_target` (i.e. age-stratified). Expected: hard crash with `fold-index mismatch ... refusing to transfer`.

- [ ] **Step 4: Write the runbook**

Create `docs/runbooks/age-vwm-transfer.md` documenting: the two source runs (ID + GLM, both `stratify_target=<VWM_COL>`, `sweeper=source_age_pinned`), then the four B-arms (B1/B3 = `transfer=from_age_ft`; B2/B4 = `transfer=from_age_frozen`; ID source for B1/B2, GLM source for B3/B4), each with `sweeper=transfer_finetune` and the paper preset (`n_repetitions=10 n_outer_folds=5 inner_hpo_trials=20`). Note that B-arm `features=` must match its source carrier so `node_feat_dim` matches the loaded backbone.

- [ ] **Step 5: Commit**

```bash
git add docs/runbooks/age-vwm-transfer.md
git commit -m "docs: age→VWM transfer runbook (source runs + B1-B4)"
```

---

## Task 11: A4 + A5 baselines

**Files:**
- Modify: `docs/runbooks/age-vwm-transfer.md` (append A4 + A5 sections).
- A5 reuses the existing classical-ML nested harness (no new GNN code).

- [ ] **Step 1: A4 (GLM→age) — config-only nested run**

Append to the runbook the command (age-stratified — A4 is a standalone reported number, distinct from the VWM-stratified GLM-age *source* run):

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc model=gcn features=glm_diagonal labels.target=<AGE_COL> \
  sweeper=source_age_pinned experiment_name=a4_glm_to_age \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20
```

- [ ] **Step 2: A5 (age→VWM, no graph) — classical 1-feature baseline**

Append the command using the existing classical-ML nested harness (ElasticNet) with a single input feature = age. Confirm the classical runner accepts a scalar-covariate input mode; if the classical harness reads node/edge features only, A5 is computed as the closed-form `r2 = corr(age, vwm)**2` under the same outer folds and recorded directly in EXPERIMENTS.md. Document whichever path the classical harness supports — this is the trivial developmental floor, not a model.

- [ ] **Step 3: Commit**

```bash
git add docs/runbooks/age-vwm-transfer.md
git commit -m "docs: A4 (GLM->age) + A5 (age->VWM floor) baseline runbook"
```

---

## Final verification

- [ ] Run the full suite: `PYTHONPATH=$(pwd) .venv/bin/pytest tests/ -q` — all green.
- [ ] Confirm a plain nested run (no `transfer=`, no `stratify_target`) reproduces a prior A2/A3 number within fold noise (the changes are inert when off).
- [ ] Finish the branch via `superpowers:finishing-a-development-branch` (PR to `main`, per repo git-workflow).

---

## Notes for the implementer

- **Why the backbone-fully-loaded assertion (Task 6 Step 4) matters:** if the VWM model's backbone shape differs from the source's, `load_partial_state_dict` silently skips those keys and transfer degrades to from-scratch with no error. The assertion converts that into a loud failure. This is the runtime half of spec §10; the pinned source sweeper (Task 9) is the other half.
- **Carrier/source pairing:** B1/B2 use the **ID** source; B3/B4 use the **GLM** source. The B-arm's `features=` must equal its source carrier (same `node_feat_dim`), else Task 6's assertion fires.
- **Out of scope here (Plan 2):** C1/C2 @head covariate injection. D1 deferred entirely.
```
