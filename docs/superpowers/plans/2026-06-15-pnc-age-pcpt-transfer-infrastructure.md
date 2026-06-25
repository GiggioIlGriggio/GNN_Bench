# PNC age→PCPT transfer — Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three infra pieces that the identity→age→PCPT transfer experiment needs on top of the (merged) age→VWM transfer infra: drop the divergent lr=5e-3 from the transfer sweeper, add a `frozen_random` control arm (freeze a randomly-initialised backbone, no checkpoint), and extend the cohort generator to require multiple non-NaN targets (TRITASK cohort).

**Architecture:** Reuse the existing nested-CV transfer machinery (`SourceBackboneProvider`, per-fold backbone injection, fail-closed fold-alignment guard) unchanged. All three changes are small, isolated, and TDD'd with pure-Python unit tests (no GPU/data). End-to-end correctness is validated by a **cluster smoke run on gpunode03**, not locally. See spec: `docs/superpowers/specs/2026-06-15-pnc-age-pcpt-transfer-design.md`.

**Tech Stack:** Python 3.10, PyTorch 2.10 + PyG 2.7, Hydra/OmegaConf configs, Optuna inner HPO, pytest. Cluster: Slurm + Singularity via the `cluster-helper` tooling.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `configs/sweeper/transfer_finetune.yaml` | modify | Drop `0.005` from `trainer.lr` grid |
| `tests/test_transfer_finetune_lr_grid.py` | create | Guard: grid never re-introduces 5e-3 |
| `configs/transfer/frozen_random.yaml` | create | Frozen-random control config |
| `scripts/run_experiment.py` | modify | Extract `require_nested_for_transfer()` pure guard; call it |
| `tests/test_run_experiment_routing.py` | modify | Add tests for the new guard |
| `src/training/nested_cross_validation.py` | modify | Freeze-only factory branch in `_wrap_factory_for_fold` |
| `tests/test_frozen_random_backbone.py` | create | Backbone frozen + head trainable, no checkpoint read |
| `scripts/make_pnc_vwm_cohort.py` | modify | 5th-positional `also_require`; `intersect_cohorts()`; multi-load main |
| `tests/test_make_pnc_vwm_cohort.py` | modify | Update to 5-tuple `parse_args`; add `also_require` + intersect tests |
| `slurm/make_cohort.sh` | modify | Report the actual OUT_PATH (not the hardcoded VWM file) |

All work on branch `feature/age-pcpt-transfer` (worktree `../GNN_Bench-age-pcpt-transfer`). Run pytest with `.venv/bin/python -m pytest`.

---

## Task 1: Drop the divergent lr=5e-3 from the transfer sweeper

**Why:** `age_vwm_transfer_dynamics` — identity *full*-FT diverged in 7/50 folds at lr=5e-3 (test r²≈−0.51); 1e-4/5e-4/1e-3 were stable. This is the headline arm's known failure. Remove the value and lock it with a guard test.

**Files:**
- Modify: `configs/sweeper/transfer_finetune.yaml:10`
- Test: `tests/test_transfer_finetune_lr_grid.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_transfer_finetune_lr_grid.py`:

```python
"""Guard: the transfer fine-tune sweeper must not re-introduce lr=5e-3.

Per docs/superpowers/specs/2026-06-13-pnc-age-vwm-transfer-design.md follow-up
(age_vwm_transfer_dynamics): identity full-FT diverged in 7/50 folds at
lr=5e-3 on the dense-SC/identity backbone. The grid is pinned to 1e-4/5e-4/1e-3.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

SWEEPER = Path("configs/sweeper/transfer_finetune.yaml")


def test_lr_grid_excludes_5e3():
    cfg = OmegaConf.load(SWEEPER)
    lr_spec = str(cfg.params["trainer.lr"])
    assert "0.005" not in lr_spec, (
        f"lr=5e-3 diverges identity full-FT (7/50 folds); must stay dropped, got {lr_spec}"
    )


def test_lr_grid_keeps_the_three_stable_values():
    cfg = OmegaConf.load(SWEEPER)
    lr_spec = str(cfg.params["trainer.lr"])
    for v in ("0.0001", "0.0005", "0.001"):
        assert v in lr_spec, f"expected {v} in transfer lr grid, got {lr_spec}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_finetune_lr_grid.py -q`
Expected: `test_lr_grid_excludes_5e3` FAILS (0.005 currently present).

- [ ] **Step 3: Edit the sweeper**

In `configs/sweeper/transfer_finetune.yaml`, change line 10 from:

```yaml
  trainer.lr: choice(0.0001, 0.0005, 0.001, 0.005)
```

to:

```yaml
  # lr=0.005 dropped: diverged identity full-FT in 7/50 folds (age_vwm_transfer_dynamics).
  trainer.lr: choice(0.0001, 0.0005, 0.001)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_finetune_lr_grid.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add configs/sweeper/transfer_finetune.yaml tests/test_transfer_finetune_lr_grid.py
git commit -m "fix(transfer): drop divergent lr=5e-3 from transfer_finetune sweeper"
```

---

## Task 2: Frozen-random control

**Why:** Isolates whether a *frozen* transfer gain comes from the age-trained representation or from any random nonlinear graph projection. Needs a path that freezes the backbone **without** loading a source checkpoint — which the current infra cannot express (freezing only runs inside `transfer_factory`, which requires a `SourceBackboneProvider`).

### Task 2a: Extract a pure runner-compatibility guard

**Files:**
- Modify: `scripts/run_experiment.py` (near `select_runner`, ~line 42; and the inline guard ~line 390)
- Test: `tests/test_run_experiment_routing.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_run_experiment_routing.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_routing.py -q`
Expected: FAIL — `ImportError: cannot import name 'require_nested_for_transfer'`.

- [ ] **Step 3: Add the pure function** in `scripts/run_experiment.py`, immediately after `select_runner()` (after line ~76, before the next def):

```python
def require_nested_for_transfer(*, transfer_enabled: bool, frozen_layers, runner: str) -> None:
    """Fail loud if backbone manipulation is requested outside the nested runner.

    Both source-checkpoint transfer (``transfer.enabled``) and the frozen-random
    control (``transfer.frozen_layers`` non-empty with no source) only take effect
    inside the nested-CV route; any other runner would silently train from scratch.
    """
    if (transfer_enabled or bool(frozen_layers)) and runner != "nested":
        raise ValueError(
            f"transfer / frozen backbone requires the nested runner (runner=nested), "
            f"but the resolved runner is {runner!r}. The age-pretrained (or frozen-random) "
            "backbone is only applied in the nested-CV route — any other runner would "
            "silently train from scratch. Set runner=nested or transfer=none."
        )
```

- [ ] **Step 4: Replace the inline guard** in `scripts/run_experiment.py` (~lines 390-396). Change:

```python
    if transfer_cfg.enabled and runner != "nested":
        raise ValueError(
            f"transfer.enabled=true requires the nested runner (runner=nested), but "
            f"the resolved runner is {runner!r}. The age-pretrained backbone is only "
            "injected in the nested-CV route — any other runner would silently train "
            "from scratch. Set runner=nested or transfer=none."
        )
```

to:

```python
    require_nested_for_transfer(
        transfer_enabled=transfer_cfg.enabled,
        frozen_layers=transfer_cfg.frozen_layers,
        runner=runner,
    )
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_routing.py -q`
Expected: PASS (all prior + 4 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.py tests/test_run_experiment_routing.py
git commit -m "refactor(run): extract require_nested_for_transfer guard (covers frozen_layers)"
```

### Task 2b: Freeze-only factory branch in nested CV

**Files:**
- Modify: `src/training/nested_cross_validation.py:404-405` (`_wrap_factory_for_fold`)
- Test: `tests/test_frozen_random_backbone.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_frozen_random_backbone.py`:

```python
"""Frozen-random control: freeze a randomly-initialised backbone, no checkpoint.

_wrap_factory_for_fold must, when there is NO source provider but frozen_layers
is non-empty, return a factory that builds a fresh model and freezes the backbone
(head stays trainable) — without reading any checkpoint.
"""

from __future__ import annotations

import torch

from src.training.nested_cross_validation import NestedCrossValidator


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 1)


def _ncv(frozen_layers):
    # cfg is stored but unused by _wrap_factory_for_fold; pass None to avoid
    # constructing a full TrainerConfig.
    return NestedCrossValidator(cfg=None, source_provider=None, frozen_layers=frozen_layers)


def test_no_provider_no_frozen_returns_original_factory():
    ncv = _ncv([])
    orig = lambda cfg: _FakeModel()
    wrapped = ncv._wrap_factory_for_fold(orig, rep=0, fold=0, train_val_idx=[1, 2], test_idx=[0])
    assert wrapped is orig  # identity passthrough — from-scratch (transfer=none)


def test_frozen_random_freezes_backbone_only():
    ncv = _ncv(["backbone"])
    wrapped = ncv._wrap_factory_for_fold(
        lambda cfg: _FakeModel(), rep=0, fold=0, train_val_idx=[1, 2], test_idx=[0]
    )
    model = wrapped(None)
    bb = {n: p.requires_grad for n, p in model.named_parameters() if n.startswith("backbone")}
    hd = {n: p.requires_grad for n, p in model.named_parameters() if n.startswith("head")}
    assert bb and not any(bb.values()), f"backbone must be frozen, got {bb}"
    assert hd and all(hd.values()), f"head must stay trainable, got {hd}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_frozen_random_backbone.py -q`
Expected: `test_frozen_random_freezes_backbone_only` FAILS — current code returns `orig` whenever `source_provider is None`, so the backbone is **not** frozen.

- [ ] **Step 3: Edit `_wrap_factory_for_fold`** in `src/training/nested_cross_validation.py`. Replace lines 404-405:

```python
        if self.source_provider is None:
            return model_factory
```

with:

```python
        if self.source_provider is None:
            if not self.frozen_layers:
                return model_factory
            # Frozen-random control: no source checkpoint, but freeze a
            # randomly-initialised backbone so only the head trains. Isolates the
            # age-pretraining effect from a generic frozen graph projection.
            from src.training.transfer_ops import freeze_layers

            frozen = self.frozen_layers

            def frozen_random_factory(trial_model_cfg):
                model = model_factory(trial_model_cfg)
                freeze_layers(model, frozen)
                return model

            return frozen_random_factory
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_frozen_random_backbone.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/nested_cross_validation.py tests/test_frozen_random_backbone.py
git commit -m "feat(transfer): freeze-only factory path for frozen-random control"
```

### Task 2c: Frozen-random config

**Files:**
- Create: `configs/transfer/frozen_random.yaml`
- Test: `tests/test_frozen_random_backbone.py` (append a config-shape assertion)

- [ ] **Step 1: Write the failing test** — append to `tests/test_frozen_random_backbone.py`:

```python
from pathlib import Path
from omegaconf import OmegaConf


def test_frozen_random_config_shape():
    cfg = OmegaConf.load(Path("configs/transfer/frozen_random.yaml"))
    # enabled=False => run_experiment builds NO SourceBackboneProvider.
    assert cfg.enabled is False
    assert cfg.source_checkpoint_root is None
    # non-empty frozen_layers => triggers the freeze-only path.
    assert list(cfg.frozen_layers) == ["backbone"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_frozen_random_backbone.py::test_frozen_random_config_shape -q`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Create the config** `configs/transfer/frozen_random.yaml`:

```yaml
# Frozen-random control (age-pretraining specificity).
# Freeze a RANDOMLY-INITIALISED backbone (no source checkpoint), train only the
# head. enabled=false => run_experiment builds NO SourceBackboneProvider; the
# non-empty frozen_layers triggers the freeze-only path in NestedCrossValidator
# (_wrap_factory_for_fold). Pin the SAME architecture as the source so the only
# difference vs from_age_frozen is trained-vs-random backbone weights.
enabled: false
source_checkpoint_root: null
checkpoint_variant: best
frozen_layers:
  - backbone
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_frozen_random_backbone.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add configs/transfer/frozen_random.yaml tests/test_frozen_random_backbone.py
git commit -m "feat(transfer): add frozen_random control config"
```

---

## Task 3: Multi-target cohort generator (TRITASK)

**Why:** `make_pnc_vwm_cohort.py` filters on a single target (loads the dataset, which NaN-filters on the current target). TRITASK = `graph ∩ age ∩ VWM ∩ PCPT_precision ∩ GLM-map` needs VWM **and** PCPT both non-NaN. Cleanest reuse: load the dataset once per required target (each load applies that target's NaN filter) and intersect the subject-id sets. No CSV parsing or SUBJID-mapping logic. Verified result: VWM-940 ∩ PCPT = **937**.

### Task 3a: parse_args gains `also_require` (5th positional)

**Files:**
- Modify: `scripts/make_pnc_vwm_cohort.py` (`parse_args`)
- Test: `tests/test_make_pnc_vwm_cohort.py`

- [ ] **Step 1: Update existing tests + add new** — rewrite `tests/test_make_pnc_vwm_cohort.py` to the 5-tuple signature:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_make_pnc_vwm_cohort.py -q`
Expected: FAIL — `parse_args` returns a 4-tuple (ValueError unpacking 5) and `intersect_cohorts` does not exist.

- [ ] **Step 3: Update `parse_args` + add `intersect_cohorts`** in `scripts/make_pnc_vwm_cohort.py`. Replace the `parse_args` function (lines 62-79) with:

```python
def parse_args(argv: list[str]) -> tuple[Path, str, str, list[str], list[str]]:
    """Split CLI args into positionals + extra Hydra overrides.

    Positionals are ``[OUT_PATH] [LABELS] [FEATURES] [ALSO_REQUIRE]``.
    ALSO_REQUIRE is a comma-separated list of extra label configs whose targets
    must ALSO be non-NaN (e.g. ``pnc_PCPT_accuracy`` for the TRITASK cohort).
    Any arg containing ``=`` is treated as a Hydra override (e.g.
    ``dataset.root=/cluster/PNC``) and forwarded verbatim. Order-independent.
    """
    extra_overrides = [a for a in argv if "=" in a]
    positional = [a for a in argv if "=" not in a]
    out_path = Path(positional[0]) if len(positional) > 0 else (
        REPO_ROOT / "configs" / "subject_lists" / "pnc_vwm_cohort.txt"
    )
    labels = positional[1] if len(positional) > 1 else "pnc_VWMdprime"
    features = positional[2] if len(positional) > 2 else "glm_diagonal"
    also_require = (
        [s for s in positional[3].split(",") if s] if len(positional) > 3 else []
    )
    return out_path, labels, features, also_require, extra_overrides


def intersect_cohorts(base: list[str], extra_sets: list[list[str]]) -> list[str]:
    """Sorted intersection of a base subject list with zero or more extra sets."""
    result = set(base)
    for s in extra_sets:
        result &= set(s)
    return sorted(result)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_make_pnc_vwm_cohort.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/make_pnc_vwm_cohort.py tests/test_make_pnc_vwm_cohort.py
git commit -m "feat(cohort): parse_args also_require + intersect_cohorts helper"
```

### Task 3b: Wire main() to load-per-target and intersect

**Files:**
- Modify: `scripts/make_pnc_vwm_cohort.py` (`main`, refactor load into `_loadable_sub_ids`)

No new unit test (requires real PNC data — covered by the cluster smoke). This is a mechanical refactor of existing logic plus the intersection loop.

- [ ] **Step 1: Refactor `main`** in `scripts/make_pnc_vwm_cohort.py`. Replace the body of `main()` (lines 82-140, from `out_path, labels, ... = parse_args(...)` through the `out_path.write_text(...)` / final print) with:

```python
def _loadable_sub_ids(cfg) -> list[str]:
    """Bare 'sub-XXXXXXXXXX' ids loadable under a composed config (NaN-filtered
    on that config's target). Composed cfg must already carry dataset/labels/features.
    """
    from src.configs.dataset_config import DatasetConfig
    from src.configs.feature_config import FeatureConfig
    from src.configs.label_config import LabelConfig
    from src.datasets.registry import get_dataset

    dataset_cfg = DatasetConfig(**OmegaConf.to_container(cfg.dataset, resolve=True))
    feature_cfg = FeatureConfig(**OmegaConf.to_container(cfg.features, resolve=True))
    label_cfg = LabelConfig(**OmegaConf.to_container(cfg.labels, resolve=True))
    print(
        f"Loading PNC with target={label_cfg.target} to determine the loadable cohort...",
        flush=True,
    )
    dataset = get_dataset(
        name=dataset_cfg.name, cfg=dataset_cfg,
        feature_cfg=feature_cfg, label_cfg=label_cfg,
    )
    dataset.load_raw()
    sub_ids, seen = [], set()
    for sid in dataset._subject_ids:
        m = _SUB_RE.search(sid)
        if not m:
            continue
        bare = m.group(1)
        if bare not in seen:
            seen.add(bare)
            sub_ids.append(bare)
    return sub_ids


def main() -> None:
    out_path, labels, features, also_require, extra_overrides = parse_args(sys.argv[1:])

    config_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        def _compose(label):
            return compose(
                config_name="experiment",
                overrides=[
                    "dataset=pnc", f"labels={label}", f"features={features}",
                    *extra_overrides,
                ],
            )

        base_ids = _loadable_sub_ids(_compose(labels))
        extra_sets = [_loadable_sub_ids(_compose(lbl)) for lbl in also_require]

    sub_ids = intersect_cohorts(base_ids, extra_sets)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    require_desc = f" AND non-NaN {','.join(also_require)}" if also_require else ""
    header = (
        f"# PNC cohort (target={labels}, features={features}{require_desc}): "
        f"{len(sub_ids)} subjects with a graph AND a non-NaN target{require_desc}.\n"
        f"# Generated by scripts/make_pnc_vwm_cohort.py — pass as "
        f"dataset.subject_list_file to ALL arms so folds align.\n"
    )
    out_path.write_text(header + "\n".join(sub_ids) + "\n")
    print(f"Wrote {len(sub_ids)} subject ids to {out_path}", flush=True)
```

(Keep the module-level imports `from hydra import compose, initialize_config_dir` and `from omegaconf import OmegaConf` as they are.)

- [ ] **Step 2: Sanity check — import + arg-parse still pass (no data)**

Run: `.venv/bin/python -m pytest tests/test_make_pnc_vwm_cohort.py -q`
Expected: PASS (8 passed — import-level refactor didn't break parsing).

- [ ] **Step 3: Sanity check — module imports cleanly**

Run: `.venv/bin/python -c "import scripts.make_pnc_vwm_cohort as m; print(m._loadable_sub_ids.__name__, m.intersect_cohorts.__name__)"`
Expected: prints `_loadable_sub_ids intersect_cohorts` with no ImportError.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_pnc_vwm_cohort.py
git commit -m "feat(cohort): multi-target intersection in cohort generator main"
```

### Task 3c: make_cohort.sh reports the actual cohort file

**Files:**
- Modify: `slurm/make_cohort.sh:52` (hardcoded `COHORT_FILE`)

Untestable locally (bash on cluster); validated by the smoke cohort job. Mechanical.

- [ ] **Step 1: Edit the report block** in `slurm/make_cohort.sh`. Replace lines 52-56:

```bash
COHORT_FILE="configs/subject_lists/pnc_vwm_cohort.txt"
if [[ -f "$COHORT_FILE" ]]; then
    N=$(grep -cvE '^\s*(#|$)' "$COHORT_FILE")
    echo "[cohort] $COHORT_FILE has $N subject ids"
fi
```

with (derive the OUT_PATH from the first non-`=` token of RUN_ARGS, fall back to the VWM default):

```bash
COHORT_FILE="configs/subject_lists/pnc_vwm_cohort.txt"
for tok in ${RUN_ARGS:-}; do
    if [[ "$tok" != *"="* ]]; then COHORT_FILE="$tok"; break; fi
done
if [[ -f "$COHORT_FILE" ]]; then
    N=$(grep -cvE '^\s*(#|$)' "$COHORT_FILE")
    echo "[cohort] $COHORT_FILE has $N subject ids"
fi
```

- [ ] **Step 2: Lint the script**

Run: `bash -n slurm/make_cohort.sh`
Expected: no output (syntax OK).

- [ ] **Step 3: Commit**

```bash
git add slurm/make_cohort.sh
git commit -m "fix(cohort): make_cohort.sh reports the actual OUT_PATH not the VWM default"
```

---

## Final: full local test suite

- [ ] Run the whole suite to confirm nothing regressed:

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (pre-existing + new). Investigate any failure before pushing.

- [ ] Push the branch:

```bash
git push -u origin feature/age-pcpt-transfer
```

---

## Validation: cluster smoke on gpunode03 (NOT local)

End-to-end smoke via `cluster-helper` on **gpunode03**, fast preset, to prove every new code path runs on real data before the full launch. Each step gates the next.

1. **Cohort smoke (CPU node).** Submit `slurm/make_cohort.sh` with
   `RUN_ARGS="configs/subject_lists/pnc_tritask_cohort.txt pnc_VWMdprime glm_diagonal pnc_PCPT_accuracy"`.
   - **Gate:** confirms `PNC_ALL_SCORES_PCPT.csv` exists on the cluster (the `pnc_PCPT_accuracy` load), the multi-target intersection works, and the count is **~937**. If the PCPT CSV is missing on the cluster, repoint `configs/labels/pnc_PCPT_accuracy.yaml:metadata_file` to `Tabular_data/PNC_ALL_SCORES.csv` (verified byte-identical locally) and resubmit.
2. **Source smoke (gpunode03).** identity→age on the tritask cohort, fast preset (`trainer.n_repetitions=1`, `trainer.inner_hpo_trials=0`, `trainer.epochs=5`, ~2 outer folds), `runner=nested`, `stratify_target=PCPT_precision`, `logging.project=age_pcpt_transfer`. Produces `fold_indices.json` + per-fold checkpoints under `<exp_name>-<jobid>`.
3. **B-frozen smoke (gpunode03).** `transfer=from_age_frozen`, `transfer.source_checkpoint_root=checkpoints/<source-run-name>`, same cohort, `inner_hpo_trials=2` (exercises the lr-fixed `transfer_finetune` grid), fast preset, `target=PCPT_precision`, `stratify_target=PCPT_precision`.
   - **Gate:** the fold-alignment guard must PASS (proves source/transfer partitions align) and HPO must only sample lr ∈ {1e-4, 5e-4, 1e-3}.
4. **frozen-random smoke (gpunode03).** `transfer=frozen_random`, same cohort/preset/target.
   - **Gate:** runs to completion, logs "Froze N parameters", produces a finite test r² (proves the freeze-only path builds a real GNN, freezes the backbone, trains the head).
5. **A3 from-scratch smoke (gpunode03).** `transfer=none`, same cohort/preset/target — confirms the unchanged from-scratch path still runs on this cohort.

If all five are green, the infra is validated; proceed to the handoff for the full 10×5×20 launch. **Do not** run the full experiment in this session — that is the handoff's job.

---

## Self-review notes

- **Spec coverage:** lr-fix (§4.1)→Task 1; frozen-random code path (§4.2)→Task 2a/2b/2c; multi-column cohort generator (§4.3)→Task 3a/3b/3c; A3=`transfer=none` needs no code (noted §3); A5 closed-form needs no code (computed=0.284). Cluster execution (§5) + cluster CSV verification→Validation. ✓
- **No placeholders:** every code/test step shows full content and exact commands. ✓
- **Type/name consistency:** `parse_args` 5-tuple `(out_path, labels, features, also_require, extra)` used identically in script + tests; `intersect_cohorts(base, extra_sets)`, `_loadable_sub_ids(cfg)`, `require_nested_for_transfer(*, transfer_enabled, frozen_layers, runner)`, `frozen_random_factory` consistent across tasks. ✓
