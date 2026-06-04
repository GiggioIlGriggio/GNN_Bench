# Configurable GNN Normalization (`model.norm`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the per-layer GNN normalization a validated, configurable choice (`model.norm` ∈ `batch`|`layer`|`none`, default `batch`) wired into all three backbones, with default behavior byte-identical to today.

**Architecture:** Add a `norm` field to `ModelConfig` (Pydantic-validated, mirroring `pooling`/`jk_mode`). Add a module-level `build_norm(kind, dim)` factory in `base_backbone.py`. Replace the hardcoded `nn.BatchNorm1d(cfg.hidden_dim)` in each of `gcn.py`/`gat.py`/`gin.py` with `build_norm(cfg.norm, cfg.hidden_dim)`. `forward` is unchanged — all three norm kinds are shape-preserving drop-ins at `base_backbone.py:95`.

**Tech Stack:** Python, Pydantic v2, PyTorch, PyTorch Geometric, pytest.

**Source spec:** `docs/superpowers/specs/2026-06-04-configurable-gnn-norm-design.md` (approved).

**Test command prefix (symlinked venv — verify PyG imports first):**
```bash
cd /home/compa/Documents/working_dir/GNN_Bench-gnn-norm-layernorm
PYTHONPATH=$(pwd) .venv/bin/python -m pytest <args>
```

---

## File Structure

- **Modify** `src/configs/model_config.py` — add the `norm` field (validation comes free from the `Literal` type, as with `pooling`/`jk_mode`).
- **Modify** `src/models/backbones/base_backbone.py` — add the module-level `build_norm` factory.
- **Modify** `src/models/backbones/gcn.py`, `gat.py`, `gin.py` — swap the hardcoded `nn.BatchNorm1d(...)` for `build_norm(cfg.norm, cfg.hidden_dim)`.
- **Modify** `configs/model/gcn.yaml`, `gat.yaml`, `gin.yaml` — add explicit `norm: batch`.
- **Create** `tests/test_backbone_norm.py` — config-validation + per-(backbone × kind) module-type and forward-smoke tests.

---

### Task 1: `model.norm` config field + validation

**Files:**
- Modify: `src/configs/model_config.py` (add field after `jk_mode`, ~line 51)
- Test: `tests/test_backbone_norm.py`

- [ ] **Step 1: Write the failing config tests**

Create `tests/test_backbone_norm.py` with:

```python
"""Tests for the configurable `model.norm` knob and the build_norm factory.

All tests use synthetic data — no real datasets.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch_geometric.data

from src.configs.model_config import ModelConfig
from src.models.backbones.base_backbone import build_norm
from src.models.backbones.gcn import GCNBackbone
from src.models.backbones.gat import GATBackbone
from src.models.backbones.gin import GINBackbone

BACKBONES = {"gcn": GCNBackbone, "gat": GATBackbone, "gin": GINBackbone}
NORM_TYPES = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "none": nn.Identity}


def _cfg(norm: str = "batch") -> ModelConfig:
    """Small backbone config; hidden_dim divisible by heads for GAT."""
    return ModelConfig(
        name="test",
        hidden_dim=16,
        num_layers=2,
        heads=4,
        dropout=0.0,
        norm=norm,
    )


def _make_synthetic_data(
    num_nodes: int = 20,
    node_feat_dim: int = 3,
    num_edges: int = 60,
    edge_feat_dim: int = 1,
) -> torch_geometric.data.Data:
    return torch_geometric.data.Data(
        x=torch.randn(num_nodes, node_feat_dim),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, edge_feat_dim),
        y=torch.tensor([1.0], dtype=torch.float32),
    )


# --- Config validation -----------------------------------------------------

def test_norm_defaults_to_batch() -> None:
    assert ModelConfig(name="test").norm == "batch"


def test_norm_accepts_known_kinds() -> None:
    for kind in ("batch", "layer", "none"):
        assert ModelConfig(name="test", norm=kind).norm == kind


def test_norm_rejects_unknown_kind() -> None:
    with pytest.raises(Exception):  # pydantic.ValidationError
        ModelConfig(name="test", norm="bogus")
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run:
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/test_backbone_norm.py -k "norm_defaults or norm_accepts or norm_rejects" -v
```
Expected: FAIL — `ModelConfig` has no `norm` field (`test_norm_defaults_to_batch` fails on missing attribute / the import of `build_norm` may also error; that's fixed in Task 2). To isolate config behavior here, the three named tests fail because `norm` is unknown/ignored.

- [ ] **Step 3: Add the `norm` field**

In `src/configs/model_config.py`, after the `jk_mode` field (line 48-51) add:

```python
    norm: Literal["batch", "layer", "none"] = Field(
        default="batch",
        description="Per-layer normalization in GNN backbones "
        "(batch=BatchNorm1d, layer=LayerNorm, none=Identity).",
    )
```

(`Literal` is already imported at line 5; validation/rejection of unknown values is automatic, matching `pooling`/`jk_mode`.)

- [ ] **Step 4: Run the config tests to verify they pass**

Run:
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/test_backbone_norm.py -k "norm_defaults or norm_accepts or norm_rejects" -v
```
Expected: 3 passed (the `build_norm` import at module top must already resolve — if Task 2 not yet done, collection errors; do Task 2 then re-run). To keep slices independent, commit config + factory together is acceptable; otherwise stub `build_norm` is unnecessary because Task 2 follows immediately.

- [ ] **Step 5: Commit**

```bash
git add src/configs/model_config.py
git commit -m "$(cat <<'EOF'
feat(config): add validated model.norm field (batch|layer|none)

Default `batch` keeps every prior result byte-identical. Mirrors the
existing pooling/jk_mode Literal-validated style.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `build_norm` factory + wire all three backbones + tests

**Files:**
- Modify: `src/models/backbones/base_backbone.py` (add module-level helper)
- Modify: `src/models/backbones/gcn.py:33`, `gat.py:37`, `gin.py:43`
- Test: `tests/test_backbone_norm.py` (add backbone tests)

- [ ] **Step 1: Add the backbone norm tests**

Append to `tests/test_backbone_norm.py`:

```python
# --- build_norm factory ----------------------------------------------------

def test_build_norm_returns_expected_modules() -> None:
    assert isinstance(build_norm("batch", 16), nn.BatchNorm1d)
    assert isinstance(build_norm("layer", 16), nn.LayerNorm)
    assert isinstance(build_norm("none", 16), nn.Identity)


def test_build_norm_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_norm("bogus", 16)


# --- backbone wiring: norm module type per (backbone, kind) ----------------

@pytest.mark.parametrize("backbone_name", list(BACKBONES))
@pytest.mark.parametrize("kind", list(NORM_TYPES))
def test_backbone_norm_module_type(backbone_name: str, kind: str) -> None:
    cfg = _cfg(norm=kind)
    backbone = BACKBONES[backbone_name](cfg, in_channels=3)
    assert len(backbone.norms) == cfg.num_layers
    for norm_layer in backbone.norms:
        assert isinstance(norm_layer, NORM_TYPES[kind])


# --- default-unchanged regression -----------------------------------------

@pytest.mark.parametrize("backbone_name", list(BACKBONES))
def test_default_norm_is_batchnorm(backbone_name: str) -> None:
    """Omitting `norm` must reproduce today's BatchNorm1d (prior results valid)."""
    cfg = ModelConfig(name="test", hidden_dim=16, num_layers=2, heads=4)
    backbone = BACKBONES[backbone_name](cfg, in_channels=3)
    assert isinstance(backbone.norms[0], nn.BatchNorm1d)


# --- forward smoke per (backbone, kind) -----------------------------------

@pytest.mark.parametrize("backbone_name", list(BACKBONES))
@pytest.mark.parametrize("kind", list(NORM_TYPES))
def test_backbone_forward_smoke(backbone_name: str, kind: str) -> None:
    cfg = _cfg(norm=kind)
    backbone = BACKBONES[backbone_name](cfg, in_channels=3)
    backbone.eval()
    data = _make_synthetic_data(node_feat_dim=3)
    with torch.no_grad():
        out = backbone(data)
    assert out.shape == (data.x.size(0), backbone.get_output_dim())
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run the backbone tests to verify they fail**

Run:
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/test_backbone_norm.py -v
```
Expected: FAIL/ERROR — `build_norm` does not exist in `base_backbone` (ImportError at collection), and backbones still hardcode `BatchNorm1d` so `layer`/`none` type asserts would fail.

- [ ] **Step 3: Add the `build_norm` factory**

In `src/models/backbones/base_backbone.py`, after the imports / before `_apply_adjacency_type` (around line 12), add:

```python
def build_norm(kind: str, dim: int) -> nn.Module:
    """Construct the per-layer normalization module for a GNN backbone.

    - ``"batch"``: :class:`torch.nn.BatchNorm1d` (the historical default).
    - ``"layer"``: plain :class:`torch.nn.LayerNorm` (per-node, across channels).
    - ``"none"``:  :class:`torch.nn.Identity` (no normalization).

    ``ModelConfig.norm`` already validates the value; the ``raise`` is
    defense-in-depth for direct callers.
    """
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r}")
```

- [ ] **Step 4: Wire the factory into all three backbones**

In `src/models/backbones/gcn.py` replace lines 32-33:
```python
        for _ in range(cfg.num_layers):
            self.norms.append(nn.BatchNorm1d(cfg.hidden_dim))
```
with:
```python
        for _ in range(cfg.num_layers):
            self.norms.append(build_norm(cfg.norm, cfg.hidden_dim))
```
and add `build_norm` to the base import at the top:
```python
from src.models.backbones.base_backbone import GNNBackbone, build_norm
```

Apply the identical change in `src/models/backbones/gat.py` (lines 36-37) and `src/models/backbones/gin.py` (lines 42-43), each adding `build_norm` to its `from src.models.backbones.base_backbone import GNNBackbone` line.

- [ ] **Step 5: Run the full new test file to verify it passes**

Run:
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/test_backbone_norm.py -v
```
Expected: all passed (3 config + 2 factory + 9 module-type + 3 default-regression + 9 forward-smoke).

- [ ] **Step 6: Commit**

```bash
git add src/models/backbones/base_backbone.py src/models/backbones/gcn.py src/models/backbones/gat.py src/models/backbones/gin.py tests/test_backbone_norm.py
git commit -m "$(cat <<'EOF'
feat(backbones): build per-layer norm via build_norm(cfg.norm, dim)

Adds a shared build_norm factory in base_backbone and wires gcn/gat/gin
to it. Default `batch` reproduces nn.BatchNorm1d exactly. Adds TDD
coverage for all three norm kinds across all three backbones.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Document the default in the model YAMLs

**Files:**
- Modify: `configs/model/gcn.yaml`, `gat.yaml`, `gin.yaml`

- [ ] **Step 1: Add explicit `norm: batch`**

In each of `configs/model/gcn.yaml`, `gat.yaml`, `gin.yaml`, add after the `jk_mode` line:
```yaml
norm: batch                   # batch | layer | none
```

- [ ] **Step 2: Verify Hydra resolves the field**

Run:
```bash
PYTHONPATH=$(pwd) .venv/bin/python -c "
from src.configs.model_config import ModelConfig
import yaml
for m in ('gcn','gat','gin'):
    d = yaml.safe_load(open(f'configs/model/{m}.yaml'))
    print(m, ModelConfig(**d).norm)
"
```
Expected: `gcn batch` / `gat batch` / `gin batch`.

- [ ] **Step 3: Commit**

```bash
git add configs/model/gcn.yaml configs/model/gat.yaml configs/model/gin.yaml
git commit -m "$(cat <<'EOF'
docs(config): make model.norm=batch explicit in gcn/gat/gin YAMLs

Self-documents the resolved config; default is unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Full-regression verification

- [ ] **Step 1: Run the model + training test suites**

Run:
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/test_models.py tests/test_backbone_norm.py tests/test_training.py -v
```
Expected: new norm tests + `test_models.py` pass. The **2 pre-existing** failures `tests/test_training.py::TestNestedCrossValidator::{test_fast_preset_smoke,test_hpo_preset_smoke}` (stale ADR-0012 path assertion — see `reports/2026-06-04-identity-vwm-r2-decomposition.md` §8) remain and are **unrelated** — leave them.

- [ ] **Step 2: Confirm no *new* failures vs the known baseline**

Only the 2 stale `test_training.py` smoke failures are acceptable. Any other red is a regression to fix before pushing.

---

## Experiment (after merge-ready; see spec §Experiment)

Push `feature/gnn-norm-layernorm`, then submit 3 jobs to **gpunode02 / rtx6000** (chosen 2026-06-04; all 3 seeds run in parallel) with names `gcn-pnc-sc-vwm-identity-layernorm-seed{42,100,200}`:

```
experiment_name=gcn-pnc-sc-vwm-identity-layernorm-seed<S> features=identity dataset=pnc model=gcn labels=pnc_VWMdprime model.norm=layer logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=<S>
```

Comparator = the already-finished E3 BatchNorm jobs (360854/360855/360856) — do **not** re-run BatchNorm. ~16 h/job. On COMPLETED: `cluster-fetch` (root-relative paths) → `scripts/pooled_vs_meanfolds.py` → update report §9, `EXPERIMENTS.md`, memory `identity_vwm_r2_investigation.md`.

---

## Self-Review

- **Spec coverage:** config field+validation (Task 1) ✓; build_norm factory (Task 2 §3) ✓; 3-backbone wiring (Task 2 §4) ✓; LayerNorm flavor = plain `nn.LayerNorm` (Task 2 factory) ✓; YAML defaults (Task 3) ✓; test plan items 1-4 (Task 1+2 tests) ✓; regression run incl. known-stale failures (Task 4) ✓; experiment recipe + node/names (Experiment section) ✓.
- **Placeholders:** none — all steps carry real code/commands.
- **Type consistency:** `build_norm(kind: str, dim: int)` used identically in factory, tests, and all three backbones; `NORM_TYPES`/`BACKBONES` dicts drive the parametrized tests.
