# Classical ML Connectivity/GLM Baselines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add XGBoost + ElasticNet classical-ML baselines (and reuse the existing torch MLP) that learn from the GNN's exact thresholded connectivity and GLM activations, under the same nested-CV protocol, so results are directly comparable to the GNN.

**Architecture:** A new `SklearnNestedCrossValidator` (epoch-free fold loop) reuses the GNN pipeline's leaf components — extracted split logic, `FoldBarrier` label/GLM normalization, `TrialOverrides`+search-space DSL, Optuna, `compute_metrics`, `NestedCVResult`. Feature vectors are flattened from the same PyG `Data` the GNN consumes. The torch MLP runs unchanged through the existing runner. `run_experiment.py` gains one early branch for `model.kind == "sklearn"`.

**Tech Stack:** Python, PyTorch Geometric, scikit-learn (present), xgboost (added), Optuna (present), Hydra/pydantic configs, Slurm/Singularity on the cluster.

**Spec:** `docs/superpowers/specs/2026-06-10-classical-ml-connectivity-glm-baselines-design.md`

---

## File structure

**New:**
- `src/training/splits.py` — stratified binning + outer/inner fold index generation, shared by both runners (guarantees identical folds).
- `src/models/flatten.py` — numpy feature-vector extraction from PyG `Data` (adjacency upper-triangle + node-feature flatten).
- `src/models/sklearn_baselines.py` — estimator registry: `build_estimator(name, params, seed)`.
- `src/training/sklearn_nested_cv.py` — `SklearnNestedCrossValidator`.
- `configs/model/xgboost.yaml`, `configs/model/elasticnet.yaml`.
- `configs/sweeper/xgboost.yaml`, `configs/sweeper/elasticnet.yaml`.
- `slurm/train_sklearn.sh` — CPU (no-GPU) Singularity launch.
- Tests: `tests/test_splits.py`, `tests/test_flatten.py`, `tests/test_sklearn_baselines.py`, `tests/test_sklearn_nested_cv.py`, `tests/test_sklearn_configs.py`.

**Modified (additive):**
- `requirements.txt` — add `xgboost`.
- `src/configs/model_config.py` — add `kind` field.
- `src/training/nested_cross_validation.py` — call `splits.py` (behavior-preserving).
- `scripts/run_experiment.py` — early `kind == "sklearn"` dispatch + `run_sklearn` helper.

**Not modified:** `MLPBrainModel` (its batched torch vectorization is pinned to `flatten.py` by an equivalence test instead — lower risk than rewriting working code; honors "don't rewrite what exists").

---

## Task 1: Add xgboost dependency

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_sklearn_baselines.py` (import smoke added in Task 5)

- [ ] **Step 1: Add the dependency**

Add to `requirements.txt` (near `scikit-learn==1.7.2`, line ~71):

```
xgboost==2.1.4
```

- [ ] **Step 2: Install locally and verify import**

Run: `pip install xgboost==2.1.4 && python -c "import xgboost; print(xgboost.__version__)"`
Expected: prints `2.1.4` (if the exact pin is unavailable, bump to the nearest installable `2.1.x` and update the line).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add xgboost for classical-ML baselines"
```

> **Container note:** the cluster container must be rebuilt before sklearn jobs run — `cluster-push-container` (Task 12). scikit-learn is already in the image.

---

## Task 2: Extract fold-split logic to `src/training/splits.py`

**Files:**
- Create: `src/training/splits.py`
- Modify: `src/training/nested_cross_validation.py` (use the shared helpers)
- Test: `tests/test_splits.py`

- [ ] **Step 1: Write the failing test (parity with current NestedCrossValidator splits)**

```python
# tests/test_splits.py
import numpy as np
from src.training.splits import stratify_bins, outer_folds, inner_split


def _labels(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n)


def test_stratify_bins_matches_qcut():
    import pandas as pd
    labels = _labels()
    got = stratify_bins(labels, n_bins=5)
    exp = pd.qcut(labels, q=5, labels=False, duplicates="drop")
    assert np.array_equal(got, exp)


def test_outer_folds_deterministic_and_partition():
    labels = _labels()
    bins = stratify_bins(labels, n_bins=5)
    folds_a = outer_folds(n=len(labels), bins=bins, seeds=[42, 43], n_outer=5)
    folds_b = outer_folds(n=len(labels), bins=bins, seeds=[42, 43], n_outer=5)
    # Deterministic
    assert folds_a == folds_b
    # 2 reps x 5 folds
    assert len(folds_a) == 10
    # Each fold: test indices partition the dataset within a rep
    for rep in range(2):
        rep_folds = folds_a[rep * 5:(rep + 1) * 5]
        test_union = sorted(i for _, te in rep_folds for i in te)
        assert test_union == list(range(len(labels)))


def test_inner_split_is_subset_and_3to1():
    labels = _labels()
    bins = stratify_bins(labels, n_bins=5)
    train_val = list(range(0, 100))
    tr, va = inner_split(train_val, bins, inner_seed=42 * 1000 + 0)
    assert set(tr).issubset(train_val) and set(va).issubset(train_val)
    assert set(tr).isdisjoint(va)
    assert len(tr) + len(va) == len(train_val)
    # StratifiedKFold(4) first fold → ~3:1
    assert 2.0 < len(tr) / len(va) < 4.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.splits'`

- [ ] **Step 3: Implement `src/training/splits.py`**

```python
# src/training/splits.py
"""Stratified fold-index generation shared by the GNN and sklearn nested-CV runners.

Extracted verbatim from NestedCrossValidator so BOTH runners produce
byte-identical folds for the same labels + seeds (the basis for
baseline-vs-GNN comparability).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def stratify_bins(labels: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin continuous labels for stratified splitting."""
    return pd.qcut(labels, q=n_bins, labels=False, duplicates="drop")


def outer_folds(
    *, n: int, bins: np.ndarray, seeds: List[int], n_outer: int,
) -> List[Tuple[List[int], List[int]]]:
    """Enumerate (train_val_idx, test_idx) for every (rep, fold), rep-major order.

    Mirrors NestedCrossValidator.run: one StratifiedKFold per outer seed.
    """
    out: List[Tuple[List[int], List[int]]] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for train_val_idx, test_idx in skf.split(np.arange(n), bins):
            out.append((train_val_idx.tolist(), test_idx.tolist()))
    return out


def inner_split(
    train_val_idx: List[int], bins: np.ndarray, inner_seed: int,
) -> Tuple[List[int], List[int]]:
    """Stratified 4-way split → first fold's 3:1 train:val (matches ADR-0003)."""
    idx_arr = np.asarray(train_val_idx)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=inner_seed)
    first_train, first_val = next(skf.split(idx_arr, bins[idx_arr]))
    return idx_arr[first_train].tolist(), idx_arr[first_val].tolist()
```

- [ ] **Step 4: Refactor `NestedCrossValidator` to call the shared helpers**

In `src/training/nested_cross_validation.py`:

Add to imports (near line 50):
```python
from src.training.splits import inner_split as _shared_inner_split
from src.training.splits import outer_folds as _shared_outer_folds
from src.training.splits import stratify_bins as _shared_stratify_bins
```

Replace the outer loop body in `run()` (lines ~293–322) — instead of the nested `for rep / for fold_idx` over `StratifiedKFold`, enumerate via the helper while preserving rep/fold indices:
```python
        all_folds = _shared_outer_folds(
            n=len(dataset), bins=bins, seeds=outer_seeds, n_outer=n_outer,
        )
        for flat_idx, (train_val_idx, test_idx) in enumerate(all_folds):
            rep = flat_idx // n_outer
            fold_idx = flat_idx % n_outer
            outer_seed = outer_seeds[rep]
            fr = self._run_outer_fold(
                rep=rep,
                fold=fold_idx,
                outer_seed=outer_seed,
                train_val_idx=train_val_idx,
                test_idx=test_idx,
                dataset=dataset,
                labels=labels,
                bins=bins,
                base_model_cfg=base_model_cfg,
                model_factory=model_factory,
                logger=logger,
                device=device,
                label_builder=label_builder,
                label_components=label_components,
                glm_col_range=glm_col_range,
                glm_normalize=glm_normalize,
                feature_config=feature_config,
                ckpt_root=ckpt_root,
            )
            fold_results.append(fr)
            log.info(
                "rep=%d fold=%d outer-test: %s",
                rep, fold_idx, fr.outer_test_metrics,
            )
```

Replace `_stratify_bins` body (line ~688) to delegate:
```python
    def _stratify_bins(self, labels: np.ndarray) -> np.ndarray:
        return _shared_stratify_bins(labels, self.cfg.stratify_bins)
```

Replace `_inner_split` body (line ~693) to delegate:
```python
    def _inner_split(
        self, train_val_idx, bins, inner_seed,
    ):
        return _shared_inner_split(train_val_idx, bins, inner_seed)
```

- [ ] **Step 5: Run split tests + existing nested-CV tests**

Run: `pytest tests/test_splits.py tests/test_nested_cross_validation.py -v`
(If `tests/test_nested_cross_validation.py` does not exist, run `pytest tests/ -k "nested or cross_validation" -v`.)
Expected: PASS (refactor is behavior-preserving).

- [ ] **Step 6: Commit**

```bash
git add src/training/splits.py src/training/nested_cross_validation.py tests/test_splits.py
git commit -m "refactor: extract shared fold-split logic into src/training/splits.py"
```

---

## Task 3: Feature-vector extraction `src/models/flatten.py`

**Files:**
- Create: `src/models/flatten.py`
- Test: `tests/test_flatten.py` (includes equivalence vs MLP's batched output)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flatten.py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.flatten import adjacency_vector, node_feature_vector, build_feature_matrix


def _toy_graph(n=4):
    # Upper-tri edges (0-1 w=2.0), (1-3 w=5.0), made symmetric
    ei = torch.tensor([[0, 1, 1, 3], [1, 0, 3, 1]], dtype=torch.long)
    ea = torch.tensor([[2.0], [2.0], [5.0], [5.0]])
    x = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
    return Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=n)


def test_adjacency_vector_upper_tri_weighted():
    g = _toy_graph(4)
    v = adjacency_vector(g, num_nodes=4, weighted=True)
    # tri order for N=4: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
    assert v.shape == (6,)
    assert np.allclose(v, [2.0, 0.0, 0.0, 0.0, 5.0, 0.0])


def test_node_feature_vector_flattens_x():
    g = _toy_graph(4)
    v = node_feature_vector(g)
    assert v.shape == (8,)
    assert np.allclose(v, np.arange(8))


def test_build_feature_matrix_shapes():
    graphs = [_toy_graph(4) for _ in range(5)]
    X_adj = build_feature_matrix(graphs, input_mode="adjacency", num_nodes=4, weighted=True)
    X_nf = build_feature_matrix(graphs, input_mode="node_features", num_nodes=4, weighted=True)
    assert X_adj.shape == (5, 6)
    assert X_nf.shape == (5, 8)


def test_adjacency_matches_mlp_batched_output():
    """Pin flatten.adjacency_vector to the existing MLP's batched vectorization."""
    from src.configs.model_config import ModelConfig
    from src.models.mlp_model import MLPBrainModel

    graphs = [_toy_graph(4) for _ in range(3)]
    batch = next(iter(DataLoader(graphs, batch_size=3)))
    cfg = ModelConfig(name="mlp", mlp_input="adjacency", mlp_adjacency_type="weighted")
    mlp = MLPBrainModel(cfg, node_feat_dim=2, edge_feat_dim=1, num_nodes=4)
    mlp_vecs = mlp._build_adjacency_vector(batch).detach().cpu().numpy()
    ours = np.stack([adjacency_vector(g, num_nodes=4, weighted=True) for g in graphs])
    assert np.allclose(mlp_vecs, ours)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_flatten.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.flatten'`

- [ ] **Step 3: Implement `src/models/flatten.py`**

```python
# src/models/flatten.py
"""Flatten PyG Data into numpy feature vectors for classical-ML baselines.

The adjacency vector reconstructs the upper triangle from the SAME sparse
edge_index/edge_attr the GNN consumes (post top-k thresholding), so a
sklearn baseline and the GNN see identical connectivity. Pinned to the
existing MLP's batched vectorization by tests/test_flatten.py.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch_geometric.data


def adjacency_vector(
    data: torch_geometric.data.Data, *, num_nodes: int, weighted: bool = True,
) -> np.ndarray:
    """Upper-triangular adjacency as a flat vector of length N*(N-1)/2."""
    N = num_nodes
    tri = N * (N - 1) // 2
    # flat index for (i, j), i < j
    flat_map = np.zeros((N, N), dtype=np.int64)
    iu = np.triu_indices(N, k=1)
    flat_map[iu] = np.arange(tri)

    out = np.zeros(tri, dtype=np.float64)
    ei = data.edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    upper = src < dst
    su, du = src[upper], dst[upper]
    if weighted and getattr(data, "edge_attr", None) is not None:
        ea = data.edge_attr.cpu().numpy()
        vals = ea[:, 0] if ea.ndim > 1 else ea
        vals = vals[upper]
    else:
        vals = np.ones(su.shape[0], dtype=np.float64)
    out[flat_map[su, du]] = vals
    return out


def node_feature_vector(data: torch_geometric.data.Data) -> np.ndarray:
    """Flatten node features [N, F] → [N*F]."""
    return data.x.cpu().numpy().reshape(-1).astype(np.float64)


def build_feature_matrix(
    graphs: List[torch_geometric.data.Data],
    *,
    input_mode: str,
    num_nodes: int,
    weighted: bool = True,
) -> np.ndarray:
    """Stack per-graph vectors into [n_graphs, dim] for the given input mode."""
    rows: List[np.ndarray] = []
    for g in graphs:
        if input_mode == "adjacency":
            rows.append(adjacency_vector(g, num_nodes=num_nodes, weighted=weighted))
        elif input_mode == "node_features":
            rows.append(node_feature_vector(g))
        elif input_mode == "both":
            rows.append(np.concatenate([
                adjacency_vector(g, num_nodes=num_nodes, weighted=weighted),
                node_feature_vector(g),
            ]))
        else:
            raise ValueError(
                f"Unknown input_mode {input_mode!r}; expected "
                f"'adjacency', 'node_features', or 'both'."
            )
    return np.stack(rows, axis=0)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_flatten.py -v`
Expected: PASS (all four tests, including MLP equivalence).

- [ ] **Step 5: Commit**

```bash
git add src/models/flatten.py tests/test_flatten.py
git commit -m "feat: numpy feature-vector extraction pinned to MLP vectorization"
```

---

## Task 4: Add `kind` field to `ModelConfig`

**Files:**
- Modify: `src/configs/model_config.py`
- Test: `tests/test_sklearn_configs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sklearn_configs.py
from src.configs.model_config import ModelConfig


def test_kind_defaults_to_gnn():
    assert ModelConfig(name="gcn").kind == "gnn"


def test_kind_accepts_sklearn():
    assert ModelConfig(name="xgboost", kind="sklearn").kind == "sklearn"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sklearn_configs.py -v`
Expected: FAIL — `AttributeError`/validation error (no `kind` field).

- [ ] **Step 3: Implement**

In `src/configs/model_config.py`, add after the `name` field (line ~19):
```python
    kind: Literal["gnn", "sklearn"] = Field(
        default="gnn",
        description="Estimator family. 'gnn' (default) routes through the torch "
        "Trainer/NestedCrossValidator; 'sklearn' routes through "
        "SklearnNestedCrossValidator (XGBoost/ElasticNet).",
    )
```
(`Literal` is already imported.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sklearn_configs.py -v`
Expected: PASS (these two tests; config-load tests added in Task 8).

- [ ] **Step 5: Commit**

```bash
git add src/configs/model_config.py tests/test_sklearn_configs.py
git commit -m "feat: add model.kind to route gnn vs sklearn estimators"
```

---

## Task 5: Estimator registry `src/models/sklearn_baselines.py`

**Files:**
- Create: `src/models/sklearn_baselines.py`
- Test: `tests/test_sklearn_baselines.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sklearn_baselines.py
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models.sklearn_baselines import build_estimator, SKLEARN_ESTIMATORS


def test_registry_keys():
    assert set(SKLEARN_ESTIMATORS) == {"xgboost", "elasticnet"}


def test_elasticnet_built_with_params():
    pipe = build_estimator("elasticnet", {"alpha": 0.5, "l1_ratio": 0.3}, seed=0)
    assert isinstance(pipe, Pipeline)
    enet = pipe.steps[-1][1]
    assert enet.alpha == 0.5 and enet.l1_ratio == 0.3


def test_xgboost_built_with_params():
    pipe = build_estimator("xgboost", {"n_estimators": 50, "max_depth": 3}, seed=0)
    xgb = pipe.steps[-1][1]
    assert xgb.n_estimators == 50 and xgb.max_depth == 3


def test_unknown_estimator_raises():
    with pytest.raises(ValueError):
        build_estimator("randomforest", {}, seed=0)


def test_estimator_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)); y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    pipe = build_estimator("elasticnet", {"alpha": 0.01, "l1_ratio": 0.5}, seed=0)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (60,)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sklearn_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.sklearn_baselines'`

- [ ] **Step 3: Implement `src/models/sklearn_baselines.py`**

```python
# src/models/sklearn_baselines.py
"""Classical-ML estimator registry for the sklearn baseline runner.

Each builder returns a leakage-safe sklearn Pipeline:
StandardScaler (fit on train fold) → estimator. Required for ElasticNet;
harmless for XGBoost. Hyperparameters arrive as a flat dict (the model's
``model_params`` after TrialOverrides) so the search-space DSL can sweep
``model.model_params.<hp>`` exactly like the GNN sweeps ``model.<field>``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_elasticnet(params: Dict[str, Any], seed: int):
    p = {"max_iter": 5000, "random_state": seed}
    p.update(params)
    return ElasticNet(**p)


def _build_xgboost(params: Dict[str, Any], seed: int):
    from xgboost import XGBRegressor

    p = {
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        "objective": "reg:squarederror",
    }
    p.update(params)
    return XGBRegressor(**p)


SKLEARN_ESTIMATORS: Dict[str, Callable[[Dict[str, Any], int], Any]] = {
    "elasticnet": _build_elasticnet,
    "xgboost": _build_xgboost,
}


def build_estimator(name: str, params: Dict[str, Any], *, seed: int) -> Pipeline:
    """Return a StandardScaler→estimator Pipeline for ``name`` with ``params``."""
    if name not in SKLEARN_ESTIMATORS:
        raise ValueError(
            f"Unknown sklearn estimator {name!r}; "
            f"expected one of {sorted(SKLEARN_ESTIMATORS)}."
        )
    estimator = SKLEARN_ESTIMATORS[name](params, seed)
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sklearn_baselines.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/sklearn_baselines.py tests/test_sklearn_baselines.py
git commit -m "feat: sklearn estimator registry (xgboost, elasticnet) with scaler pipeline"
```

---

## Task 6: `SklearnNestedCrossValidator`

**Files:**
- Create: `src/training/sklearn_nested_cv.py`
- Test: `tests/test_sklearn_nested_cv.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sklearn_nested_cv.py
import numpy as np
import torch
from torch_geometric.data import Data

from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from src.training.nested_cross_validation import NestedCVResult
from src.training.sklearn_nested_cv import SklearnNestedCrossValidator


class _NullLogger:
    class _Cfg:
        enabled = False
    def __init__(self): self.cfg = self._Cfg()
    def __getattr__(self, name):
        def _noop(*a, **k): return None
        return _noop


def _signal_dataset(n=80, nodes=6, seed=0):
    """Node features carry a linear signal so r2 should be clearly positive."""
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    ea = torch.ones(ei.shape[1], 1)
    for _ in range(n):
        x = torch.tensor(rng.normal(size=(nodes, 1)), dtype=torch.float32)
        y = float(x.sum().item() + rng.normal(scale=0.05))
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=nodes))
        labels.append(y)
    return graphs, np.asarray(labels)


def test_sklearn_nested_cv_runs_and_recovers_signal(tmp_path):
    graphs, labels = _signal_dataset()
    trainer_cfg = TrainerConfig(
        n_repetitions=1, n_outer_folds=2, inner_hpo_trials=2,
        hpo_metric="val_r2", stratify_bins=4, seed=42,
        checkpoint_dir=str(tmp_path),
    )
    model_cfg = ModelConfig(
        name="elasticnet", kind="sklearn", mlp_input="node_features",
        model_params={"alpha": 0.1, "l1_ratio": 0.5},
    )
    scv = SklearnNestedCrossValidator(
        cfg=trainer_cfg,
        search_space_path="configs/sweeper/elasticnet.yaml",
    )
    result = scv.run(
        estimator_name="elasticnet",
        model_cfg=model_cfg,
        dataset=graphs,
        labels=labels,
        logger=_NullLogger(),
        run_name="test_run",
        glm_col_range=None,
        glm_normalize=False,
        num_nodes=6,
    )
    assert isinstance(result, NestedCVResult)
    assert len(result.fold_results) == 2
    assert "r2" in result.mean_metrics
    # node-feature sum is highly learnable → pooled-ish mean r2 clearly positive
    assert result.mean_metrics["r2"] > 0.5
    # result JSON persisted + reloadable
    saved = NestedCVResult.load(tmp_path / "test_run" / "nested_cv_result.json")
    assert saved.model_name == "elasticnet"
```

(Task 8 creates `configs/sweeper/elasticnet.yaml`; if running tasks out of order, create it first.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sklearn_nested_cv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.sklearn_nested_cv'`

- [ ] **Step 3: Implement `src/training/sklearn_nested_cv.py`**

```python
# src/training/sklearn_nested_cv.py
"""Epoch-free nested cross-validation for classical-ML baselines.

Reuses the GNN pipeline's leaf components — shared fold splits
(src.training.splits), FoldBarrier label/GLM normalization, the
search-space DSL + TrialOverrides, Optuna, compute_metrics, and the
NestedCVResult/FoldResult schema — so sklearn baselines are directly
comparable to the GNN runs (same folds, same label handling, same output).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from src.datasets.label_builder import LabelBuilder
from src.models.flatten import build_feature_matrix
from src.models.sklearn_baselines import build_estimator
from src.training.fold_barrier import FoldBarrier
from src.training.metrics import compute_metrics
from src.training.nested_cross_validation import (
    FoldResult, NestedCVResult, _aggregate,
)
from src.training.search_space import (
    TrialOverrides, load_sweeper_params, parse_search_space,
)
from src.training.splits import inner_split, outer_folds, stratify_bins

log = logging.getLogger(__name__)


class SklearnNestedCrossValidator:
    """Nested CV for sklearn estimators. Mirrors the GNN protocol without epochs."""

    def __init__(
        self,
        cfg: TrainerConfig,
        search_space_path: Optional[str | Path] = None,
    ) -> None:
        self.cfg = cfg
        self.search_space_path = (
            Path(search_space_path) if search_space_path is not None else None
        )

    def run(
        self,
        *,
        estimator_name: str,
        model_cfg: ModelConfig,
        dataset: List[Any],
        labels: np.ndarray,
        logger: Any,
        run_name: str,
        num_nodes: int,
        glm_col_range: Optional[Tuple[int, int]] = None,
        glm_normalize: bool = True,
        label_builder: Optional[LabelBuilder] = None,
        label_components: Optional[pd.DataFrame] = None,
        feature_config: Optional[Dict[str, Any]] = None,
    ) -> NestedCVResult:
        n_outer = self.cfg.effective_n_outer_folds
        if n_outer < 2:
            raise ValueError(f"effective_n_outer_folds must be >= 2, got {n_outer}")

        specs = None
        if self.cfg.inner_hpo_trials > 0:
            if self.search_space_path is None:
                raise ValueError("inner_hpo_trials > 0 but no search_space_path given")
            specs = parse_search_space(load_sweeper_params(self.search_space_path))

        input_mode = model_cfg.mlp_input
        weighted = model_cfg.mlp_adjacency_type == "weighted"
        outer_seeds = self.cfg.resolved_outer_seeds()
        bins = stratify_bins(labels, self.cfg.stratify_bins)
        all_folds = outer_folds(
            n=len(dataset), bins=bins, seeds=outer_seeds, n_outer=n_outer,
        )

        ckpt_root = Path(self.cfg.checkpoint_dir) / run_name
        fold_results: List[FoldResult] = []

        for flat_idx, (train_val_idx, test_idx) in enumerate(all_folds):
            rep = flat_idx // n_outer
            fold = flat_idx % n_outer
            outer_seed = outer_seeds[rep]
            inner_seed = outer_seed * 1000 + fold
            fr = self._run_fold(
                estimator_name=estimator_name, model_cfg=model_cfg,
                dataset=dataset, labels=labels, bins=bins,
                train_val_idx=train_val_idx, test_idx=test_idx,
                rep=rep, fold=fold, inner_seed=inner_seed,
                input_mode=input_mode, weighted=weighted, num_nodes=num_nodes,
                specs=specs, glm_col_range=glm_col_range, glm_normalize=glm_normalize,
                label_builder=label_builder, label_components=label_components,
                logger=logger,
            )
            fold_results.append(fr)
            log.info("rep=%d fold=%d outer-test: %s", rep, fold, fr.outer_test_metrics)

        mean_metrics, std_metrics = _aggregate(fold_results)
        result = NestedCVResult(
            run_name=run_name, model_name=estimator_name,
            n_repetitions=self.cfg.n_repetitions, n_outer_folds=n_outer,
            inner_hpo_trials=self.cfg.inner_hpo_trials, hpo_metric=self.cfg.hpo_metric,
            outer_seeds=outer_seeds, fold_results=fold_results,
            mean_metrics=mean_metrics, std_metrics=std_metrics,
        )
        result.save(ckpt_root / "nested_cv_result.json")
        try:
            logger.log_nested_final_summary(mean_metrics, std_metrics)
        except Exception as exc:  # noqa: BLE001
            log.warning("log_nested_final_summary failed (continuing): %s", exc)
        return result

    # ------------------------------------------------------------------

    def _barrier_and_labels(
        self, *, dataset, labels, train_idx, eval_idx,
        glm_col_range, glm_normalize, label_builder, label_components,
    ):
        """Fit a FoldBarrier on train_idx; return (barrier, y_train, y_eval)."""
        barrier = FoldBarrier(
            label_norm_strategy=self.cfg.label_norm_strategy,
            glm_col_range=glm_col_range, glm_normalize=glm_normalize,
            label_builder=label_builder,
        )
        train_graphs = [dataset[i] for i in train_idx]
        if label_builder is not None and label_components is not None:
            barrier.fit(train_graphs, label_components.iloc[train_idx])
            y_train = barrier.transform_labels(label_components.iloc[train_idx])
            y_eval = barrier.transform_labels(label_components.iloc[eval_idx])
        else:
            barrier.fit(train_graphs, labels[train_idx])
            y_train = barrier.transform_labels(labels[train_idx])
            y_eval = barrier.transform_labels(labels[eval_idx])
        return barrier, y_train, y_eval

    def _matrix(self, barrier, dataset, idx, input_mode, weighted, num_nodes):
        graphs = barrier.transform_graphs([dataset[i] for i in idx])
        return build_feature_matrix(
            graphs, input_mode=input_mode, num_nodes=num_nodes, weighted=weighted,
        )

    def _score(self, name, params, X_tr, y_tr, X_ev, y_ev, barrier, seed):
        """Fit on train, predict eval, return metrics on the original scale."""
        pipe = build_estimator(name, params, seed=seed)
        pipe.fit(X_tr, y_tr)
        y_pred = barrier.inverse_transform_labels(pipe.predict(X_ev))
        y_true = barrier.inverse_transform_labels(y_ev)
        return y_true, y_pred, pipe

    def _run_fold(
        self, *, estimator_name, model_cfg, dataset, labels, bins,
        train_val_idx, test_idx, rep, fold, inner_seed, input_mode, weighted,
        num_nodes, specs, glm_col_range, glm_normalize,
        label_builder, label_components, logger,
    ) -> FoldResult:
        # ---- Inner HPO (or base params) -------------------------------
        if specs is not None and self.cfg.inner_hpo_trials > 0:
            tr_idx, va_idx = inner_split(train_val_idx, bins, inner_seed)
            barrier_in, y_tr, y_va = self._barrier_and_labels(
                dataset=dataset, labels=labels, train_idx=tr_idx, eval_idx=va_idx,
                glm_col_range=glm_col_range, glm_normalize=glm_normalize,
                label_builder=label_builder, label_components=label_components,
            )
            X_tr = self._matrix(barrier_in, dataset, tr_idx, input_mode, weighted, num_nodes)
            X_va = self._matrix(barrier_in, dataset, va_idx, input_mode, weighted, num_nodes)
            best_hparams, best_trial = self._inner_hpo(
                estimator_name, model_cfg, specs, X_tr, y_tr, X_va, y_va,
                barrier_in, inner_seed,
            )
        else:
            best_hparams, best_trial = {}, 0

        # ---- Refit on outer-trainval, evaluate on outer-test ----------
        trial_model_cfg, _ = TrialOverrides(values=best_hparams).apply(
            model_cfg=model_cfg, trainer_cfg=self.cfg,
        )
        params = dict(trial_model_cfg.model_params)
        barrier_out, y_trv, y_te = self._barrier_and_labels(
            dataset=dataset, labels=labels, train_idx=train_val_idx, eval_idx=test_idx,
            glm_col_range=glm_col_range, glm_normalize=glm_normalize,
            label_builder=label_builder, label_components=label_components,
        )
        X_trv = self._matrix(barrier_out, dataset, train_val_idx, input_mode, weighted, num_nodes)
        X_te = self._matrix(barrier_out, dataset, test_idx, input_mode, weighted, num_nodes)
        y_true, y_pred, _ = self._score(
            estimator_name, params, X_trv, y_trv, X_te, y_te, barrier_out, inner_seed,
        )
        outer_metrics = compute_metrics(y_true, y_pred)

        try:
            logger.log_nested_outer_test(rep, fold, outer_metrics)
            logger.log_nested_best_hparams(rep, fold, best_hparams, best_trial, 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb log rep=%d fold=%d failed: %s", rep, fold, exc)

        return FoldResult(
            rep=rep, fold=fold, outer_test_metrics=outer_metrics,
            best_hparams=best_hparams, best_trial=best_trial, refit_epochs=0,
            n_train=len(train_val_idx), n_val=0, n_test=len(test_idx),
            y_true=y_true.tolist(), y_pred=y_pred.tolist(),
        )

    def _inner_hpo(
        self, name, model_cfg, specs, X_tr, y_tr, X_va, y_va, barrier, inner_seed,
    ) -> Tuple[Dict[str, Any], int]:
        import optuna

        maximize = self.cfg.hpo_metric == "val_r2"
        metric_key = self.cfg.hpo_metric.replace("val_", "")
        study = optuna.create_study(
            direction="maximize" if maximize else "minimize",
            sampler=optuna.samplers.TPESampler(seed=inner_seed),
        )

        def objective(trial: "optuna.Trial") -> float:
            overrides = TrialOverrides(values={s.name: s.suggest(trial) for s in specs})
            trial_cfg, _ = overrides.apply(model_cfg=model_cfg, trainer_cfg=self.cfg)
            params = dict(trial_cfg.model_params)
            y_true, y_pred, _ = self._score(
                name, params, X_tr, y_tr, X_va, y_va, barrier, inner_seed,
            )
            return compute_metrics(y_true, y_pred)[metric_key]

        study.optimize(objective, n_trials=self.cfg.inner_hpo_trials)
        # Reconstruct chosen overrides from the best trial's sampled params.
        best = study.best_trial
        best_hparams = {s.name: best.params[s.name] for s in specs}
        return best_hparams, int(best.number)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sklearn_nested_cv.py -v`
Expected: PASS — runs end to end, `mean_metrics["r2"] > 0.5`, JSON reloads.

- [ ] **Step 5: Commit**

```bash
git add src/training/sklearn_nested_cv.py tests/test_sklearn_nested_cv.py
git commit -m "feat: SklearnNestedCrossValidator reusing GNN folds/barrier/metrics"
```

---

## Task 7: Wire sklearn dispatch into `run_experiment.py`

**Files:**
- Modify: `scripts/run_experiment.py`
- Test: `tests/test_run_sklearn_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_sklearn_dispatch.py
import numpy as np
import torch
from torch_geometric.data import Data

from src.configs.model_config import ModelConfig
from src.configs.trainer_config import TrainerConfig
from scripts.run_experiment import run_sklearn


class _NullLogger:
    class _Cfg:
        enabled = False
    def __init__(self): self.cfg = self._Cfg()
    def __getattr__(self, name):
        def _noop(*a, **k): return None
        return _noop


def test_run_sklearn_helper(tmp_path):
    rng = np.random.default_rng(0)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    ea = torch.ones(3, 1)
    graphs, labels = [], []
    for _ in range(40):
        x = torch.tensor(rng.normal(size=(4, 1)), dtype=torch.float32)
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=4))
        labels.append(float(x.sum()))
    labels = np.asarray(labels)
    trainer_cfg = TrainerConfig(
        n_repetitions=1, n_outer_folds=2, inner_hpo_trials=0,
        hpo_metric="val_r2", stratify_bins=4, checkpoint_dir=str(tmp_path),
    )
    model_cfg = ModelConfig(
        name="elasticnet", kind="sklearn", mlp_input="node_features",
        model_params={"alpha": 0.05, "l1_ratio": 0.5},
    )
    result = run_sklearn(
        model_cfg=model_cfg, trainer_cfg=trainer_cfg, graphs=graphs, labels=labels,
        logger=_NullLogger(), run_name="t", glm_col_range=None, glm_normalize=False,
        label_builder=None, label_components=None, feature_config={},
    )
    assert "r2" in result.mean_metrics
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run_sklearn_dispatch.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_sklearn'`

- [ ] **Step 3: Implement the helper + branch**

In `scripts/run_experiment.py`, add this helper at module level (after `select_runner`, ~line 81):
```python
def run_sklearn(
    *, model_cfg, trainer_cfg, graphs, labels, logger, run_name,
    glm_col_range, glm_normalize, label_builder, label_components, feature_config,
):
    """Dispatch a classical-ML (sklearn) nested-CV run. Returns NestedCVResult."""
    from src.training.sklearn_nested_cv import SklearnNestedCrossValidator

    scv = SklearnNestedCrossValidator(
        cfg=trainer_cfg, search_space_path=trainer_cfg.search_space,
    )
    return scv.run(
        estimator_name=model_cfg.name,
        model_cfg=model_cfg,
        dataset=graphs,
        labels=labels,
        logger=logger,
        run_name=run_name,
        num_nodes=graphs[0].num_nodes if graphs else 0,
        glm_col_range=glm_col_range,
        glm_normalize=glm_normalize,
        label_builder=label_builder,
        label_components=label_components,
        feature_config=feature_config,
    )
```

In `main()`, insert the branch immediately AFTER the dataset-stats logging block (after line ~290, before "6) Define model factory"):
```python
    # ------------------------------------------------------------------
    # 5b) Classical-ML (sklearn) baselines bypass the torch model factory,
    #     model summary, and select_runner — they have no torch module.
    # ------------------------------------------------------------------
    if model_cfg.kind == "sklearn":
        from src.training.run_identity import build_run_name

        log.info("Classical-ML runner — estimator=%s input=%s",
                 model_cfg.name, model_cfg.mlp_input)
        sk_result = run_sklearn(
            model_cfg=model_cfg, trainer_cfg=trainer_cfg, graphs=graphs, labels=labels,
            logger=logger, run_name=build_run_name(cfg.experiment_name),
            glm_col_range=glm_col_range, glm_normalize=feature_cfg.glm_normalize,
            label_builder=label_builder, label_components=label_components,
            feature_config=feature_cfg.model_dump(),
        )
        log.info("Classical-ML nested CV complete — mean=%s  std=%s",
                 sk_result.mean_metrics, sk_result.std_metrics)
        logger.finish()
        return
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run_sklearn_dispatch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py tests/test_run_sklearn_dispatch.py
git commit -m "feat: route model.kind=sklearn to SklearnNestedCrossValidator"
```

---

## Task 8: Model + sweeper config files

**Files:**
- Create: `configs/model/xgboost.yaml`, `configs/model/elasticnet.yaml`
- Create: `configs/sweeper/xgboost.yaml`, `configs/sweeper/elasticnet.yaml`
- Test: extend `tests/test_sklearn_configs.py`

- [ ] **Step 1: Write the failing test (append to tests/test_sklearn_configs.py)**

```python
import yaml
from src.configs.model_config import ModelConfig
from src.training.search_space import load_sweeper_params, parse_search_space


def _load_model(path):
    with open(path) as f:
        return ModelConfig(**yaml.safe_load(f))


def test_model_configs_load_as_sklearn():
    for name in ("xgboost", "elasticnet"):
        cfg = _load_model(f"configs/model/{name}.yaml")
        assert cfg.kind == "sklearn" and cfg.name == name


def test_sweeper_configs_parse():
    for name in ("xgboost", "elasticnet"):
        specs = parse_search_space(load_sweeper_params(f"configs/sweeper/{name}.yaml"))
        assert specs and all(s.name.startswith("model.model_params.") for s in specs)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sklearn_configs.py -k "sklearn or sweeper" -v`
Expected: FAIL — files not found.

- [ ] **Step 3: Create the config files**

`configs/model/elasticnet.yaml`:
```yaml
name: elasticnet
kind: sklearn
mlp_input: adjacency          # adjacency (SC/FC) | node_features (GLM); set per run
mlp_adjacency_type: weighted
model_params:
  alpha: 1.0
  l1_ratio: 0.5
```

`configs/model/xgboost.yaml`:
```yaml
name: xgboost
kind: sklearn
mlp_input: adjacency
mlp_adjacency_type: weighted
model_params:
  n_estimators: 400
  max_depth: 4
  learning_rate: 0.1
  subsample: 0.9
  colsample_bytree: 0.6
  min_child_weight: 3
  reg_lambda: 1.0
```

`configs/sweeper/elasticnet.yaml`:
```yaml
# @package hydra.sweeper
sampler:
  _target_: optuna.samplers.TPESampler
  seed: 42
direction: maximize
n_trials: 20
params:
  model.model_params.alpha: tag(log, interval(0.001, 10.0))
  model.model_params.l1_ratio: interval(0.05, 0.95)
```

`configs/sweeper/xgboost.yaml`:
```yaml
# @package hydra.sweeper
sampler:
  _target_: optuna.samplers.TPESampler
  seed: 42
direction: maximize
n_trials: 20
params:
  model.model_params.n_estimators: choice(200, 400, 600, 800, 1000)
  model.model_params.max_depth: range(2, 7)
  model.model_params.learning_rate: tag(log, interval(0.01, 0.3))
  model.model_params.subsample: interval(0.6, 1.0)
  model.model_params.colsample_bytree: interval(0.3, 1.0)
  model.model_params.min_child_weight: range(1, 11)
  model.model_params.reg_lambda: tag(log, interval(0.01, 10.0))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sklearn_configs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/model/xgboost.yaml configs/model/elasticnet.yaml \
        configs/sweeper/xgboost.yaml configs/sweeper/elasticnet.yaml \
        tests/test_sklearn_configs.py
git commit -m "feat: model + sweeper configs for xgboost and elasticnet baselines"
```

---

## Task 9: CPU Slurm launch script

**Files:**
- Create: `slurm/train_sklearn.sh`

- [ ] **Step 1: Create `slurm/train_sklearn.sh`** (GPU-free; partition/qos injected by `cluster-submit --node <cpunode>`)

```bash
#!/bin/bash
#SBATCH --job-name=gnn_bench_sklearn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# NOTE: no --gres / no --partition / no --qos here on purpose.
# Submit via `cluster-submit --node <cpunode> slurm/train_sklearn.sh ...`,
# which injects --partition/--qos for the chosen CPU node and caps cpus.

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

case "${RUN_ARGS:-}" in
    *"dataset=orbit"*) DATASET_ROOT_OVERRIDE="dataset.root=/data/bdip_ssd/al5165/GNNBenchV2/data/ORBIT" ;;
    *"dataset=pnc"*)   DATASET_ROOT_OVERRIDE="dataset.root=/data/bdip_ssd/al5165/GNNBenchV2/data/PNC" ;;
    *)                 DATASET_ROOT_OVERRIDE="" ;;
esac

echo "[run] SHA=$(git rev-parse HEAD)"
echo "[run] host=$(hostname)"
echo "[run] container=$SIF"
echo "[run] RUN_ARGS=${RUN_ARGS:-<none>}"

# CPU-only: singularity WITHOUT --nv
PYTHONPATH="$(pwd)" singularity exec \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    --bind /data/bdip_ssd/al5165/GNNBenchV2/data:/data/bdip_ssd/al5165/GNNBenchV2/data \
    "$SIF" \
    python scripts/run_experiment.py \
        logging.entity=teampolpetta \
        logging.project=baselines \
        ${RUN_ARGS:-} \
        ${DATASET_ROOT_OVERRIDE:-}
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x slurm/train_sklearn.sh
git add slurm/train_sklearn.sh
git commit -m "feat: CPU-only Slurm launch script for sklearn baselines"
```

---

## Task 10: Full test sweep + local pipeline smoke

**Files:** none (validation)

- [ ] **Step 1: Run the whole new test suite**

Run: `pytest tests/test_splits.py tests/test_flatten.py tests/test_sklearn_baselines.py tests/test_sklearn_nested_cv.py tests/test_sklearn_configs.py tests/test_run_sklearn_dispatch.py -v`
Expected: all PASS.

- [ ] **Step 2: Run the existing suite to confirm no regressions from the splits refactor**

Run: `pytest tests/ -q`
Expected: no new failures vs the pre-change baseline (the `splits.py` refactor is behavior-preserving). Note any pre-existing failures unrelated to this change.

- [ ] **Step 3: Local end-to-end pipeline smoke (one cell, if the local venv has PyG libs — see memory `llm_venv_pyg_companion_libs`)**

Run:
```bash
PYTHONPATH=$(pwd) python scripts/run_experiment.py \
  dataset=pnc model=elasticnet features=default labels=pnc_default \
  trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=2 \
  trainer.hpo_metric=val_r2 trainer.search_space=configs/sweeper/elasticnet.yaml \
  logging.enabled=false experiment_name=smoke-enet-pnc-sc-age
```
Expected: completes; logs "Classical-ML nested CV complete — mean=..."; writes `checkpoints/<run>/nested_cv_result.json`. If the local venv lacks PyG companion libs, skip and run this as the cluster smoke in Task 11.

- [ ] **Step 4: Commit (if any fixes were needed)** — otherwise no-op.

---

## Task 11: Cluster — container, smoke, r²-sanity gate, full launch

**Files:** none (execution; uses the `cluster-helper` skill)

- [ ] **Step 1: Rebuild + push the container (xgboost added)**

Run: `cluster-push-container`
Expected: `CONTAINER_PUSHED=...`.

- [ ] **Step 2: Discover the CPU node** (CPU support was added to the skill)

Run: `cluster-gpus` (and/or the new `cluster-cpus` listing).
Pick a CPU node with `USABLE=yes`. Record its name as `<cpunode>`.

- [ ] **Step 3: Pipeline smoke — all 6 (estimator × input) paths on both datasets, tiny budget**

For each `model ∈ {xgboost, elasticnet}` and each cell below, submit with `RUN_ARGS` overriding to `trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=2 logging.enabled=false`, names prefixed `smoke-`:
- SC: `dataset=pnc model=<m> features=default labels=pnc_default model.mlp_input=adjacency`
- FC: `dataset=pnc_fc model=<m> features=default labels=pnc_default model.mlp_input=adjacency`
- GLM: `dataset=pnc model=<m> features=glm_scalar labels=pnc_VWMdprime model.mlp_input=node_features trainer.search_space=configs/sweeper/<m>.yaml`
- Repeat for `dataset=orbit` / `orbit_fc` with `labels=default` (age) / `labels=orbit_mri_VWM_HL_p` (VWM).

Submit pattern (CPU node, sklearn script):
```bash
RUN_ARGS="<the overrides above> trainer.search_space=configs/sweeper/<m>.yaml" \
  cluster-submit --node <cpunode> slurm/train_sklearn.sh -J smoke-<m>-<cell>
```
Expected: every job `COMPLETED`. This proves all estimator×input×dataset paths run and the container is fresh.

- [ ] **Step 4: r²-sanity gate (the user's requirement — §8a of the spec)**

Run these anchor cells at a real-ish budget (`trainer.n_repetitions=3 trainer.n_outer_folds=5 trainer.inner_hpo_trials=10 logging.project=baselines`), named (no `smoke-` prefix):
- **FC→age PNC, elasticnet** and **xgboost** — `dataset=pnc_fc features=default labels=pnc_default model.mlp_input=adjacency`
- **SC→age PNC, elasticnet** — `dataset=pnc features=default labels=pnc_default model.mlp_input=adjacency`
- **GLM→VWM PNC, elasticnet** — `dataset=pnc features=glm_scalar labels=pnc_VWMdprime model.mlp_input=node_features`

After they finish, pull pooled r² (use the `backfill-experiment-results` skill or read each `nested_cv_result.json`). **Gate:**
  - FC→age PNC pooled r² **clearly positive** (≳0.3 expected). If ≈0 or negative → STOP and debug (label alignment / feature vector / leakage) before any further runs.
  - SC→age positive (typically < FC→age).
  - GLM→VWM PNC mildly positive (~0.05–0.2 ballpark; compare to GNN GLM cells ~0.18–0.22).
ORBIT anchors are direction-only (N≈95, high variance) — judge on pooled r² + Pearson.

- [ ] **Step 5: Launch the full 30-run matrix** (only after Step 4 passes)

For each (dataset∈{pnc,orbit} via sc cfg and fc cfg, target via labels, estimator∈{xgboost,elasticnet}) submit connectivity runs (`model.mlp_input=adjacency`), and the GLM runs (`features=glm_scalar model.mlp_input=node_features`, VWM only). Plus the MLP cells via the **GPU** path:
```bash
# sklearn (CPU)
RUN_ARGS="dataset=<ds> model=<m> features=<feat> labels=<lab> model.mlp_input=<mode> \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  trainer.hpo_metric=val_r2 trainer.search_space=configs/sweeper/<m>.yaml logging.project=baselines" \
  cluster-submit --node <cpunode> slurm/train_sklearn.sh -J <m>-<ds>-<feat>-<target>

# torch MLP (GPU) — existing path
RUN_ARGS="dataset=<ds> model=mlp features=<feat> labels=<lab> model.mlp_input=<mode> \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  trainer.hpo_metric=val_r2 trainer.search_space=configs/sweeper/mlp.yaml logging.project=baselines" \
  cluster-submit --node <gpunode> slurm/train.sh -J mlp-<ds>-<feat>-<target>
```
Naming `<m>-<ds>-<sc|fc|glm>-<age|vwm>`. Confirm node/GPU availability with `cluster-gpus` and follow `cluster-helper` rule 4/5 (ask the user to confirm the node + names before official submits).

- [ ] **Step 6: Backfill + report**

Use `backfill-experiment-results` to write pooled + mean-of-folds r² and wandb links into `EXPERIMENTS.md` (new batch entry), then run `scripts/compare_models.py` for baseline-vs-GNN significance and write a `reports/2026-06-xx-classical-ml-baselines.md`.

---

## Self-review

**Spec coverage:**
- Estimators (XGBoost/ElasticNet new, MLP reused) → Tasks 5, 8, 9, 11 (MLP via existing path in Task 11 Step 5). ✓
- Top-20% thresholded weighted connectivity vector → Task 3 (`adjacency_vector` from edge_index/edge_attr) + equivalence test. ✓
- Approach B separate runner reusing leaf components → Task 6. ✓
- Identical folds → Task 2 (shared `splits.py`) + parity test. ✓
- StandardScaler fit-on-train → Task 5 (Pipeline). ✓
- Label/GLM normalization reuse → Task 6 (`FoldBarrier`). ✓
- Protocol 10×5×20, r² → Task 11 Step 5; smaller for smoke/gate. ✓
- `baselines` wandb project → Tasks 9, 11. ✓
- CPU compute → Task 9 (script) + Task 11 (CPU node). ✓
- ORBIT GLM contrast = `contrast-2back_vs_0back` → reuses `features=glm_scalar` (Task 11). ✓
- NestedCVResult / compare_models compatibility → Task 6 (reuses schema). ✓
- 30-run matrix + r²-sanity pre-launch gate → Task 11. ✓

**Placeholder scan:** no TBD/TODO; all code blocks complete; xgboost pin carries a verify-and-bump step (Task 1 Step 2). ✓

**Type consistency:** `FoldResult`/`NestedCVResult`/`_aggregate` imported from `nested_cross_validation` and constructed with matching fields; `build_estimator(name, params, *, seed)`, `build_feature_matrix(..., input_mode, num_nodes, weighted)`, `splits.outer_folds(*, n, bins, seeds, n_outer)` used consistently across Tasks 2/3/5/6/7. ✓
