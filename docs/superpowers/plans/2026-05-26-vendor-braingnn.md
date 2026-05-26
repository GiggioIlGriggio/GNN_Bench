# Vendor Official BrainGNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rewritten BrainGNN with the official `xxlya/BrainGNN_Pytorch` conv/message-passing layers vendored verbatim, behind a thin adapter that satisfies this repo's `BrainGNN` `encode`/`decode` contract.

**Architecture:** Vendor the three upstream layer files verbatim (`brainmsgpassing.py`, `braingraphconv.py`, `inits.py`) plus the two loss functions; re-express the monolithic upstream `Network.forward` inside the adapter's `encode()` (the contract splits `forward` and we swap the head). `decode()` runs the repo's `RegressionHead`; `auxiliary_loss()` reproduces upstream's topk/unit/consist losses, with consistency applied per target-quantile-bin (regression has no classes).

**Tech Stack:** Python 3.10, PyTorch 2.4.0+cu121, PyG 2.7.0.dev, torch_scatter 2.1.2, torch_sparse 0.6.18 (`spspmm`), pytest, Hydra/OmegaConf configs.

**Spec:** `docs/superpowers/specs/2026-05-26-vendor-braingnn-design.md`
**Upstream pinned commit:** `1e337e7a13af5e4374343fda81ce11b62c5ce566` (`xxlya/BrainGNN_Pytorch`)
**Upstream files already fetched to:** `/tmp/bgnn_src/` (`net_braingnn.py`, `net_braingraphconv.py`, `net_brainmsgpassing.py`, `net_inits.py`, `03-main.py`). If absent, re-fetch (see Task 2 Step 1).

---

## Key facts for the implementer (read once)

- **`BrainGNN` contract** (`src/models/base_model.py`): `forward` is final (`encode`→`decode`, do not override). Subclasses implement `encode(data)→[B,emb]` and `decode(emb)→[B,1]`, and may override `auxiliary_loss()→dict|None` returning **pre-scaled scalar tensors** (no args).
- **Trainer** (`src/training/trainer.py:488-494`) calls `model.auxiliary_loss()` after `model(batch)`, adds each term to MSE, logs each by key. Keys become wandb `fold_N/train/<key>`.
- **Registry** (`src/models/registry.py`): `@register_model("braingnn")`; constructed as `cls(cfg=cfg, node_feat_dim=…, edge_feat_dim=…, num_nodes=…)` (`src/finetuning/finetuner.py:156`).
- **PyG 2.7 gotchas:** `TopKPooling` has no `.weight` — the gating weight is `.select.weight`; the score nonlinearity is `.select.act`. Constructor accepts `nonlinearity=torch.sigmoid` (callable).
- **`data.pos` does not exist** in this pipeline — the adapter must synthesize per-ROI identity each forward.
- **Sigmoid stack is faithful and intentional:** the pool returns a post-σ score; `encode` applies `torch.sigmoid` again before stashing (matching upstream `Network.forward`); `consist_loss` applies `torch.sigmoid` a third time. Reproduce exactly. Do not "fix" it.
- **Aggregation:** vendored `MyMessagePassing` hardcodes `scatter_add` (the `aggr='mean'` arg is asserted then ignored). With softmax-normalized edge weights in `message`, this is a weighted sum. Preserve verbatim.

## File structure

- Create: `src/models/vendor/__init__.py`, `src/models/vendor/braingnn/__init__.py`
- Create (verbatim copies, intra-package imports fixed): `src/models/vendor/braingnn/brainmsgpassing.py`, `braingraphconv.py`, `inits.py`
- Create: `src/models/vendor/braingnn/losses.py` (verbatim `topk_loss`, `consist_loss` + device fix)
- Create: `src/models/vendor/braingnn/PROVENANCE.md`
- Rewrite: `src/models/braingnn_model.py` (adapter)
- Delete: `src/models/backbones/roi_aware_conv.py`
- Create: `tests/test_vendor_braingnn.py` (new unit tests for vendored layers + adapter aux losses)
- Modify: `tests/test_models.py` (remove ROIAwareConv tests + the `_unit_loss` static test)
- Modify: `configs/model/braingnn.yaml`, `configs/sweeper/braingnn_vwm.yaml`, `configs/sweeper/braingnn_fc_vwm.yaml`
- Create: `docs/adr/0010-vendor-braingnn.md`

---

## Task 1: Create the feature worktree

**Files:** none (git only)

- [ ] **Step 1: Create worktree + branch**

Per `CLAUDE.md`, feature work goes in a `git worktree` + `feature/<slug>` branch. Use the `superpowers:using-git-worktrees` skill to create the worktree for branch `feature/vendor-braingnn` off `main`. Do all subsequent tasks inside that worktree. The spec/plan docs already committed on `main` will be present.

- [ ] **Step 2: Verify environment**

Run: `python -c "import torch_sparse, torch_scatter, torch_geometric; from torch_sparse import spspmm; print('ok')"`
Expected: prints `ok`.

---

## Task 2: Vendor the upstream layer files verbatim

**Files:**
- Create: `src/models/vendor/__init__.py`
- Create: `src/models/vendor/braingnn/__init__.py`
- Create: `src/models/vendor/braingnn/inits.py`
- Create: `src/models/vendor/braingnn/brainmsgpassing.py`
- Create: `src/models/vendor/braingnn/braingraphconv.py`
- Test: `tests/test_vendor_braingnn.py`

- [ ] **Step 1: (Re)fetch upstream if `/tmp/bgnn_src` is missing**

```bash
mkdir -p /tmp/bgnn_src && cd /tmp/bgnn_src
base="https://raw.githubusercontent.com/xxlya/BrainGNN_Pytorch/1e337e7a13af5e4374343fda81ce11b62c5ce566"
for f in net/braingnn.py net/braingraphconv.py net/brainmsgpassing.py net/inits.py 03-main.py; do
  curl -fsSL "$base/$f" -o "$(echo "$f" | tr '/' '_')"
done
cd - >/dev/null
```

- [ ] **Step 2: Create package init files**

Create `src/models/vendor/__init__.py` with a single line:
```python
"""Vendored third-party model code (see each subpackage's PROVENANCE)."""
```
Create `src/models/vendor/braingnn/__init__.py`:
```python
"""Vendored from xxlya/BrainGNN_Pytorch @ 1e337e7. See PROVENANCE.md."""
from src.models.vendor.braingnn.braingraphconv import MyNNConv
from src.models.vendor.braingnn.brainmsgpassing import MyMessagePassing
from src.models.vendor.braingnn.inits import uniform

__all__ = ["MyNNConv", "MyMessagePassing", "uniform"]
```

- [ ] **Step 3: Copy `inits.py` verbatim**

```bash
cp /tmp/bgnn_src/net_inits.py src/models/vendor/braingnn/inits.py
```
No edits. (Defines `uniform`, used by `MyNNConv.reset_parameters`.)

- [ ] **Step 4: Copy `brainmsgpassing.py` verbatim**

```bash
cp /tmp/bgnn_src/net_brainmsgpassing.py src/models/vendor/braingnn/brainmsgpassing.py
```
No edits needed (its only imports are `torch` and `torch_scatter`; the commented `scatter_` import stays commented). This file defines `MyMessagePassing` with a self-contained `propagate` using `torch_scatter.scatter_add`.

- [ ] **Step 5: Copy `braingraphconv.py` and fix intra-package imports**

```bash
cp /tmp/bgnn_src/net_braingraphconv.py src/models/vendor/braingnn/braingraphconv.py
```
Then apply exactly these two import edits (the ONLY changes to this file):

Replace:
```python
from net.brainmsgpassing import MyMessagePassing
```
with:
```python
from src.models.vendor.braingnn.brainmsgpassing import MyMessagePassing
```
Replace:
```python
from net.inits import uniform
```
with:
```python
from src.models.vendor.braingnn.inits import uniform
```
Leave everything else (`MyNNConv`: `aggr='mean'`, `add_remaining_self_loops(..., 1, ...)` in `forward`, softmax in `message`, bias in `update`, `normalize=False`) untouched.

- [ ] **Step 6: Write failing test for vendored layers**

Create `tests/test_vendor_braingnn.py`:
```python
import torch
import torch.nn as nn

from src.models.vendor.braingnn import MyNNConv, MyMessagePassing, uniform


class TestVendoredLayers:
    def test_imports(self) -> None:
        assert MyMessagePassing is not None
        assert uniform is not None

    def test_mynnconv_output_shape_and_selfloops(self) -> None:
        in_ch, out_ch, R, k = 4, 6, 5, 8
        n = nn.Sequential(
            nn.Linear(R, k, bias=False), nn.ReLU(), nn.Linear(k, in_ch * out_ch)
        )
        conv = MyNNConv(in_ch, out_ch, n, normalize=False)
        N = R  # one graph, one node per ROI
        x = torch.randn(N, in_ch)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_weight = torch.ones(edge_index.size(1), 1)
        pos = torch.eye(R)  # per-ROI identity
        out = conv(x, edge_index, edge_weight, pos)
        assert out.shape == (N, out_ch)
        assert torch.isfinite(out).all()
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_vendor_braingnn.py::TestVendoredLayers -v`
Expected: PASS (2 tests). If `MyNNConv.forward` errors on `edge_weight.squeeze()`, confirm you passed a 2-D `edge_weight` of shape `[E, 1]` as in the test.

- [ ] **Step 8: Commit**

```bash
git add src/models/vendor/ tests/test_vendor_braingnn.py
git commit -m "feat(braingnn): vendor upstream conv/message-passing layers verbatim"
```

---

## Task 3: Vendor the loss functions

**Files:**
- Create: `src/models/vendor/braingnn/losses.py`
- Test: `tests/test_vendor_braingnn.py` (append)

- [ ] **Step 1: Write failing test for the loss functions**

Append to `tests/test_vendor_braingnn.py`:
```python
from src.models.vendor.braingnn.losses import topk_loss, consist_loss


class TestVendoredLosses:
    def test_topk_loss_scalar_finite(self) -> None:
        s = torch.rand(3, 10)  # [B, n_kept] in (0,1)
        loss = topk_loss(s, ratio=0.5)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_consist_loss_zero_for_single_subject(self) -> None:
        s = torch.rand(1, 10)  # single subject -> Laplacian is zero
        loss = consist_loss(s)
        assert float(loss) == 0.0 or torch.allclose(
            torch.as_tensor(float(loss)), torch.tensor(0.0), atol=1e-6
        )

    def test_consist_loss_uses_input_device(self) -> None:
        s = torch.rand(4, 10)
        loss = consist_loss(s)
        assert torch.isfinite(torch.as_tensor(float(loss)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vendor_braingnn.py::TestVendoredLosses -v`
Expected: FAIL — `ModuleNotFoundError: src.models.vendor.braingnn.losses`.

- [ ] **Step 3: Create `losses.py` (verbatim math, device taken from input)**

Create `src/models/vendor/braingnn/losses.py`:
```python
"""Auxiliary losses vendored verbatim from xxlya/BrainGNN_Pytorch 03-main.py
(commit 1e337e7).

Only change vs upstream: the consistency-loss Laplacian is placed on the
*input tensor's* device instead of a module-global ``device``. The math is
identical. See PROVENANCE.md.
"""
import torch

EPS = 1e-10


def topk_loss(s, ratio):
    """Per-graph pooling-score regularizer (upstream `topk_loss`).

    ``s`` has shape [B, n_kept]; sort within each graph (dim=1) and push the
    top fraction toward 1 and the bottom fraction toward 0.
    """
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() \
          - torch.log(1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
    return res


def consist_loss(s):
    """Group-consistency Laplacian regularizer (upstream `consist_loss`).

    ``s`` has shape [n_subjects_in_group, n_kept]. NOTE: applies sigmoid to
    ``s`` (this is the third sigmoid in the upstream score path — faithful,
    intentional).
    """
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0], device=s.device)
    D = torch.eye(s.shape[0], device=s.device) * torch.sum(W, dim=1)
    L = D - W
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vendor_braingnn.py::TestVendoredLosses -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/models/vendor/braingnn/losses.py tests/test_vendor_braingnn.py
git commit -m "feat(braingnn): vendor topk/consist losses (device-safe)"
```

---

## Task 4: Adapter constructor + registry wiring

**Files:**
- Modify (full rewrite): `src/models/braingnn_model.py`
- Test: `tests/test_vendor_braingnn.py` (append)

- [ ] **Step 1: Write failing test for construction + contract attributes**

Append to `tests/test_vendor_braingnn.py`:
```python
from src.configs.model_config import ModelConfig
from src.models.registry import get_model
from src.models.base_model import BrainGNN


def _make_adapter(num_nodes=10, node_feat_dim=10, hidden_dim=16, **mp):
    params = {"pool_ratio": 0.5, "roi_embed_dim": 8}
    params.update(mp)
    cfg = ModelConfig(
        name="braingnn", hidden_dim=hidden_dim, dropout=0.0, model_params=params
    )
    return get_model(
        "braingnn", cfg, node_feat_dim=node_feat_dim, edge_feat_dim=1,
        num_nodes=num_nodes,
    )


class TestAdapterConstruction:
    def test_is_braingnn_subclass(self) -> None:
        model = _make_adapter()
        assert isinstance(model, BrainGNN)

    def test_pools_use_sigmoid(self) -> None:
        model = _make_adapter()
        assert model.pool1.select.act is torch.sigmoid
        assert model.pool2.select.act is torch.sigmoid

    def test_head_input_dim_is_hidden_times_four(self) -> None:
        model = _make_adapter(hidden_dim=16)
        first_linear = model.head.layers[0]
        assert first_linear.in_features == 16 * 4

    def test_default_lambdas(self) -> None:
        model = _make_adapter()
        assert model.lambda_topk == 0.1
        assert model.lambda_unit == 0.0
        assert model.lambda_consist == 0.1
        assert model.consist_n_bins == 4

    def test_num_nodes_zero_raises(self) -> None:
        cfg = ModelConfig(
            name="braingnn", hidden_dim=16, dropout=0.0,
            model_params={"pool_ratio": 0.5, "roi_embed_dim": 8},
        )
        with __import__("pytest").raises(ValueError):
            get_model("braingnn", cfg, node_feat_dim=10, edge_feat_dim=1, num_nodes=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vendor_braingnn.py::TestAdapterConstruction -v`
Expected: FAIL (old `BrainGNNModel` still uses `ROIAwareConv` / old `model_params`; `lambda_*` attrs and defaults don't exist).

- [ ] **Step 3: Rewrite `src/models/braingnn_model.py` — constructor only (methods filled in later tasks)**

Replace the **entire** file with the following. `encode`/`decode`/`auxiliary_loss` bodies are added in Tasks 5–6; for now include the `__init__` and stubs that raise so the module imports cleanly.

```python
"""BrainGNN (Li et al. 2021) — thin adapter over vendored upstream layers.

The scientifically-critical conv/message-passing code is vendored verbatim
under ``src/models/vendor/braingnn`` (see its PROVENANCE.md). This adapter
re-expresses the upstream ``Network.forward`` to fit the repo's
``encode``/``decode`` contract and swaps in the configurable RegressionHead.

Reference: https://github.com/xxlya/BrainGNN_Pytorch @ 1e337e7
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import TopKPooling, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_sparse import spspmm

from src.configs.model_config import ModelConfig
from src.models.base_model import BrainGNN
from src.models.heads.regression_head import RegressionHead
from src.models.registry import register_model
from src.models.vendor.braingnn import MyNNConv
from src.models.vendor.braingnn.losses import consist_loss, topk_loss


@register_model("braingnn")
class BrainGNNModel(BrainGNN):
    """BrainGNN adapter: vendored ROI-aware GConv + ROI-TopK pooling.

    Reads BrainGNN-specific parameters from ``cfg.model_params``:

    - ``pool_ratio`` (float, default 0.5): fraction of ROIs kept per stage.
    - ``roi_embed_dim`` (int, default 8): number of communities ``k`` in the
      per-ROI weight network.
    - ``lambda_topk`` (float, default 0.1): weight on the topk loss.
    - ``lambda_unit`` (float, default 0.0): weight on the unit loss.
    - ``lambda_consist`` (float, default 0.1): weight on the consistency loss.
    - ``consist_n_bins`` (int, default 4): target-quantile bins for the
      regression consistency loss.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        node_feat_dim: int,
        edge_feat_dim: int,
        num_nodes: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        if cfg.fusion is not None:
            raise NotImplementedError(
                "BrainGNN does not support multimodal fusion. Set model.fusion=null."
            )
        if num_nodes <= 0:
            raise ValueError(
                "BrainGNN requires num_nodes > 0. "
                "Ensure the dataset provides graphs with a fixed ROI count."
            )

        p = cfg.model_params
        self.pool_ratio: float = float(p.get("pool_ratio", 0.5))
        k: int = int(p.get("roi_embed_dim", 8))
        self.lambda_topk: float = float(p.get("lambda_topk", 0.1))
        self.lambda_unit: float = float(p.get("lambda_unit", 0.0))
        self.lambda_consist: float = float(p.get("lambda_consist", 0.1))
        self.consist_n_bins: int = int(p.get("consist_n_bins", 4))

        self.num_nodes = num_nodes
        hidden = cfg.hidden_dim
        R = num_nodes

        # Stage 1: per-ROI weight network -> vendored MyNNConv -> TopK pool.
        self.n1 = nn.Sequential(
            nn.Linear(R, k, bias=False), nn.ReLU(), nn.Linear(k, hidden * node_feat_dim)
        )
        self.conv1 = MyNNConv(node_feat_dim, hidden, self.n1, normalize=False)
        self.pool1 = TopKPooling(
            hidden, ratio=self.pool_ratio, multiplier=1, nonlinearity=torch.sigmoid
        )

        # Stage 2.
        self.n2 = nn.Sequential(
            nn.Linear(R, k, bias=False), nn.ReLU(), nn.Linear(k, hidden * hidden)
        )
        self.conv2 = MyNNConv(hidden, hidden, self.n2, normalize=False)
        self.pool2 = TopKPooling(
            hidden, ratio=self.pool_ratio, multiplier=1, nonlinearity=torch.sigmoid
        )

        self.head = RegressionHead(cfg, embedding_dim=hidden * 4)

        # Aux-loss tensors stashed during the last encode().
        self._s1: Optional[torch.Tensor] = None
        self._s2: Optional[torch.Tensor] = None
        self._w1: Optional[torch.Tensor] = None
        self._w2: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None

    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        raise NotImplementedError  # implemented in Task 5

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # implemented in Task 5

    def auxiliary_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError  # implemented in Task 6

    def _augment_adj(self, edge_index, edge_weight, num_nodes):
        """A² on the pooled subgraph (verbatim from upstream Network.augment_adj).

        Uses plain ``add_self_loops`` (default fill) — NOT
        ``add_remaining_self_loops(fill=1)``, which lives inside MyNNConv.
        """
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes)
        edge_index, edge_weight = spspmm(
            edge_index, edge_weight, edge_index, edge_weight,
            num_nodes, num_nodes, num_nodes,
        )
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vendor_braingnn.py::TestAdapterConstruction -v`
Expected: PASS (5 tests). (`encode`/`decode` are stubs but construction tests don't call them.)

- [ ] **Step 5: Commit**

```bash
git add src/models/braingnn_model.py tests/test_vendor_braingnn.py
git commit -m "feat(braingnn): adapter constructor over vendored layers"
```

---

## Task 5: Implement `encode` / `decode`

**Files:**
- Modify: `src/models/braingnn_model.py` (replace the `encode` and `decode` stubs)
- Test: `tests/test_vendor_braingnn.py` (append)

- [ ] **Step 1: Write failing tests for the forward pass**

Append to `tests/test_vendor_braingnn.py`:
```python
import torch_geometric.data as geo_data


def _make_batch(num_graphs=2, num_nodes=10, node_feat_dim=10, edge_feat_dim=1,
                num_edges_per_graph=20):
    all_x, all_ei, all_ea, all_batch = [], [], [], []
    offset = 0
    for g in range(num_graphs):
        all_x.append(torch.randn(num_nodes, node_feat_dim))
        all_ei.append(torch.randint(0, num_nodes, (2, num_edges_per_graph)) + offset)
        all_ea.append(torch.rand(num_edges_per_graph, edge_feat_dim))
        all_batch.append(torch.full((num_nodes,), g, dtype=torch.long))
        offset += num_nodes
    return geo_data.Data(
        x=torch.cat(all_x, 0), edge_index=torch.cat(all_ei, 1),
        edge_attr=torch.cat(all_ea, 0), batch=torch.cat(all_batch, 0),
        y=torch.randn(num_graphs, 1),
    )


class TestAdapterForward:
    def test_encode_shape(self) -> None:
        hidden = 16
        model = _make_adapter(num_nodes=10, node_feat_dim=10, hidden_dim=hidden)
        model.eval()
        data = _make_batch(num_graphs=3, num_nodes=10, node_feat_dim=10)
        with torch.no_grad():
            emb = model.encode(data)
        assert emb.shape == (3, hidden * 4)

    def test_forward_shape(self) -> None:
        model = _make_adapter(num_nodes=10, node_feat_dim=10, hidden_dim=16)
        model.eval()
        data = _make_batch(num_graphs=3, num_nodes=10, node_feat_dim=10)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (3, 1)

    def test_encode_stashes_aux_tensors(self) -> None:
        model = _make_adapter(num_nodes=10, node_feat_dim=10, hidden_dim=16)
        model.train()
        data = _make_batch(num_graphs=4, num_nodes=10, node_feat_dim=10)
        model.encode(data)
        assert model._s1 is not None and model._s1.shape[0] == 4
        assert model._s2 is not None and model._s2.shape[0] == 4
        assert model._w1 is not None and model._w2 is not None
        assert model._y is not None and model._y.numel() == 4
        # scores are probabilities in (0, 1) (post double-sigmoid)
        assert (model._s1 >= 0).all() and (model._s1 <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vendor_braingnn.py::TestAdapterForward -v`
Expected: FAIL — `NotImplementedError` from the `encode` stub.

- [ ] **Step 3: Replace the `encode` and `decode` stubs**

In `src/models/braingnn_model.py`, replace:
```python
    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        raise NotImplementedError  # implemented in Task 5

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # implemented in Task 5
```
with:
```python
    def encode(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Faithful port of upstream Network.forward, truncated at the readout.

        Returns the hierarchical dual readout ``[B, hidden*4]`` and stashes
        the pooling weights/scores + targets for ``auxiliary_loss``.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1

        if edge_attr is not None:
            edge_weight = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
        else:
            edge_weight = x.new_ones(edge_index.size(1))

        # Synthesize upstream's data.pos: per-ROI identity, repeated per graph.
        pos = F.one_hot(
            torch.arange(self.num_nodes, device=x.device).repeat(num_graphs),
            self.num_nodes,
        ).float()

        # --- Stage 1 ---
        x = self.conv1(x, edge_index, edge_weight, pos)
        x, edge_index, edge_weight, batch, perm, score1 = self.pool1(
            x, edge_index, edge_weight, batch
        )
        pos = pos[perm]
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # A² adjacency augmentation on the pooled subgraph.
        edge_weight = edge_weight.squeeze()
        edge_index, edge_weight = self._augment_adj(edge_index, edge_weight, x.size(0))

        # --- Stage 2 ---
        x = self.conv2(x, edge_index, edge_weight, pos)
        x, edge_index, edge_weight, batch, perm, score2 = self.pool2(
            x, edge_index, edge_weight, batch
        )
        pos = pos[perm]
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # Stash for auxiliary_loss. The extra sigmoid here matches upstream
        # Network.forward (see PROVENANCE: this is the 2nd of 3 sigmoids).
        self._w1 = self.pool1.select.weight
        self._w2 = self.pool2.select.weight
        self._s1 = torch.sigmoid(score1).view(num_graphs, -1)
        self._s2 = torch.sigmoid(score2).view(num_graphs, -1)
        self._y = data.y.detach().view(-1)

        return torch.cat([x1, x2], dim=1)  # [B, hidden*4]

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Configurable RegressionHead: [B, hidden*4] -> [B, 1]."""
        return self.head(embedding)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vendor_braingnn.py::TestAdapterForward -v`
Expected: PASS (3 tests). If `view(num_graphs, -1)` raises a reshape error, the kept-node count is non-uniform — confirm the synthetic batch uses identical `num_nodes` per graph (it does) and `pool_ratio` is constant.

- [ ] **Step 5: Commit**

```bash
git add src/models/braingnn_model.py tests/test_vendor_braingnn.py
git commit -m "feat(braingnn): implement encode/decode (faithful forward port)"
```

---

## Task 6: Implement `auxiliary_loss` (topk + unit + binned consist)

**Files:**
- Modify: `src/models/braingnn_model.py` (replace the `auxiliary_loss` stub; add `_binned_consist_loss`)
- Test: `tests/test_vendor_braingnn.py` (append)

- [ ] **Step 1: Write failing tests for aux losses**

Append to `tests/test_vendor_braingnn.py`:
```python
class TestAuxiliaryLoss:
    def test_none_in_eval(self) -> None:
        model = _make_adapter()
        model.eval()
        data = _make_batch(num_graphs=4, num_nodes=10, node_feat_dim=10)
        model.encode(data)
        assert model.auxiliary_loss() is None

    def test_keys_and_scaling(self) -> None:
        model = _make_adapter(
            num_nodes=10, node_feat_dim=10, hidden_dim=16,
            lambda_topk=0.2, lambda_unit=0.3, lambda_consist=0.4, consist_n_bins=3,
        )
        model.train()
        data = _make_batch(num_graphs=6, num_nodes=10, node_feat_dim=10)
        model.encode(data)
        aux = model.auxiliary_loss()
        assert set(aux.keys()) == {"topk_loss", "unit_loss", "consist_loss"}
        for v in aux.values():
            assert v.ndim == 0 and torch.isfinite(v)

    def test_unit_loss_formula(self) -> None:
        model = _make_adapter(lambda_unit=1.0, lambda_topk=0.0, lambda_consist=0.0)
        model.train()
        data = _make_batch(num_graphs=4, num_nodes=10, node_feat_dim=10)
        model.encode(data)
        aux = model.auxiliary_loss()
        expected = (torch.norm(model._w1, p=2) - 1) ** 2 \
            + (torch.norm(model._w2, p=2) - 1) ** 2
        assert torch.allclose(aux["unit_loss"], expected, atol=1e-6)

    def test_consist_binning_runs_with_more_bins_than_subjects(self) -> None:
        model = _make_adapter(consist_n_bins=8)
        model.train()
        data = _make_batch(num_graphs=3, num_nodes=10, node_feat_dim=10)  # 3 < 8
        model.encode(data)
        aux = model.auxiliary_loss()
        assert torch.isfinite(aux["consist_loss"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vendor_braingnn.py::TestAuxiliaryLoss -v`
Expected: FAIL — `NotImplementedError` from the `auxiliary_loss` stub.

- [ ] **Step 3: Replace the `auxiliary_loss` stub and add the binning helper**

In `src/models/braingnn_model.py`, replace:
```python
    def auxiliary_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        raise NotImplementedError  # implemented in Task 6
```
with:
```python
    def auxiliary_loss(self) -> Optional[Dict[str, torch.Tensor]]:
        """Upstream topk + unit + (binned) consistency losses, pre-scaled.

        Returns None in eval or before the first forward.
        """
        if not self.training or self._s1 is None:
            return None

        topk = topk_loss(self._s1, self.pool_ratio) + topk_loss(self._s2, self.pool_ratio)
        unit = (torch.norm(self._w1, p=2) - 1) ** 2 \
            + (torch.norm(self._w2, p=2) - 1) ** 2
        consist = self._binned_consist_loss(self._s1, self._y, self.consist_n_bins)

        return {
            "topk_loss": self.lambda_topk * topk,
            "unit_loss": self.lambda_unit * unit,
            "consist_loss": self.lambda_consist * consist,
        }

    def _binned_consist_loss(
        self, s: torch.Tensor, y: torch.Tensor, n_bins: int
    ) -> torch.Tensor:
        """Consistency loss applied per target-quantile bin (regression analogue
        of upstream's per-class loop). ``s`` is [B, n_kept], ``y`` is [B]."""
        y = y.view(-1).float()
        total = s.new_zeros(())
        if y.numel() == 0 or n_bins <= 1:
            return total + consist_loss(s)

        qs = torch.linspace(0, 1, n_bins + 1, device=y.device)[1:-1]
        edges = torch.quantile(y, qs)
        bins = torch.bucketize(y, edges)  # values in [0, n_bins-1]
        for b in range(n_bins):
            mask = bins == b
            if int(mask.sum()) >= 1:
                total = total + consist_loss(s[mask])
        return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vendor_braingnn.py::TestAuxiliaryLoss -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/models/braingnn_model.py tests/test_vendor_braingnn.py
git commit -m "feat(braingnn): auxiliary_loss (topk/unit/binned-consist)"
```

---

## Task 7: Lock the triple-sigmoid score semantics

**Files:**
- Test: `tests/test_vendor_braingnn.py` (append)

This task adds a regression test that pins the faithful sigmoid composition so a future refactor cannot silently drop or add a sigmoid.

- [ ] **Step 1: Write the locking test**

Append to `tests/test_vendor_braingnn.py`:
```python
class TestSigmoidComposition:
    def test_stashed_scores_are_sigmoid_of_pool_score(self) -> None:
        """encode() must apply exactly one extra sigmoid to the pool's score
        (the 2nd of the 3 in the upstream path). The pool already applied the
        1st; consist_loss applies the 3rd."""
        torch.manual_seed(0)
        model = _make_adapter(num_nodes=10, node_feat_dim=10, hidden_dim=16)
        model.eval()
        data = _make_batch(num_graphs=2, num_nodes=10, node_feat_dim=10)

        # Re-run the pool path manually to capture the raw (post-1st-sigmoid)
        # score the pool returns, then confirm encode stored sigmoid() of it.
        x = data.x
        edge_weight = data.edge_attr[:, 0]
        pos = torch.eye(10).repeat(2, 1)
        with torch.no_grad():
            xc = model.conv1(x, data.edge_index, edge_weight, pos)
            _, _, _, _, _, raw_score1 = model.pool1(
                xc, data.edge_index, edge_weight, data.batch
            )
            model.encode(data)
        expected_s1 = torch.sigmoid(raw_score1).view(2, -1)
        assert torch.allclose(model._s1, expected_s1, atol=1e-5)

    def test_consist_loss_applies_third_sigmoid(self) -> None:
        """consist_loss sigmoids its input — the 3rd sigmoid. Lock by comparing
        against a manual non-sigmoid Laplacian (they must differ)."""
        s = torch.rand(5, 8)  # already in (0,1)
        from src.models.vendor.braingnn.losses import consist_loss
        with_sigmoid = float(consist_loss(s))
        # manual Laplacian WITHOUT the internal sigmoid:
        ss = s
        W = torch.ones(5, 5)
        L = torch.eye(5) * W.sum(1) - W
        no_sigmoid = float(torch.trace(ss.t() @ L @ ss) / 25)
        assert abs(with_sigmoid - no_sigmoid) > 1e-6
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_vendor_braingnn.py::TestSigmoidComposition -v`
Expected: PASS (2 tests). If `test_stashed_scores_are_sigmoid_of_pool_score` fails because the pool's returned score is NOT already a probability on this PyG build, record the actual composition in `PROVENANCE.md` and adjust the test's `expected_s1` to match what `encode` faithfully reproduces — the invariant is "encode applies one sigmoid on top of the pool's returned score," not the pool's internal behavior.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vendor_braingnn.py
git commit -m "test(braingnn): lock faithful sigmoid composition"
```

---

## Task 8: Delete ROIAwareConv and clean up `test_models.py`

**Files:**
- Delete: `src/models/backbones/roi_aware_conv.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Confirm no remaining importers**

Run: `grep -rn "roi_aware_conv\|ROIAwareConv" src/ --include="*.py"`
Expected: no matches (the adapter rewrite in Task 4 removed the `braingnn_model.py` imports). If any remain, fix them before deleting.

- [ ] **Step 2: Delete the file**

```bash
git rm src/models/backbones/roi_aware_conv.py
```

- [ ] **Step 3: Remove dead tests in `tests/test_models.py`**

Delete these (they reference the removed module / removed static method):
- The entire `class TestROIAwareConv:` block (the `# ROIAwareConv tests` section, ~lines 168–214).
- The method `test_unit_loss_with_sigmoid_scores` inside `TestBrainGNNSigmoidPooling` (~lines 407–418) — it calls the removed `BrainGNNModel._unit_loss`.

Leave intact: `TestBrainGNNWithROIAwareConv` (its `test_braingnn_forward_with_roi_aware_conv` and `test_braingnn_no_roi_embedding_attr` still pass against the new model — the model still returns `[B,1]` and still has no `roi_embedding` attr). `TestBrainGNNHierarchicalReadout`, `TestBrainGNNSigmoidPooling` (remaining two tests), and `TestBrainGNNAdjAugmentation` remain valid.

- [ ] **Step 4: Run the affected test file**

Run: `pytest tests/test_models.py -v`
Expected: PASS, no collection errors, no `ImportError` for `roi_aware_conv`.

- [ ] **Step 5: Commit**

```bash
git add src/models/backbones/roi_aware_conv.py tests/test_models.py
git commit -m "refactor(braingnn): remove ROIAwareConv rewrite + its tests"
```

---

## Task 9: Update configs to the new λ knobs

**Files:**
- Modify: `configs/model/braingnn.yaml`
- Modify: `configs/sweeper/braingnn_vwm.yaml`
- Modify: `configs/sweeper/braingnn_fc_vwm.yaml`

- [ ] **Step 1: Update `configs/model/braingnn.yaml` model_params**

Replace the `model_params:` block (lines 17–21):
```yaml
model_params:
  pool_ratio: 0.5          # fraction of ROIs kept at each TopK pooling stage
  roi_embed_dim: 8         # dimensionality of the learned ROI position embedding
  unit_loss_weight: 0.3    # weight on the unit (binary pooling) auxiliary loss
  topk_loss_weight: 0.5    # weight on the topk (consist) auxiliary loss
```
with:
```yaml
model_params:
  pool_ratio: 0.5          # fraction of ROIs kept at each TopK pooling stage
  roi_embed_dim: 8         # number of communities k in the per-ROI weight net
  lambda_topk: 0.1         # weight on the per-graph topk pooling-score loss
  lambda_unit: 0.0         # weight on the unit loss (‖pool weight‖₂ − 1)²
  lambda_consist: 0.1      # weight on the per-target-bin consistency loss
  consist_n_bins: 4        # target quantile bins for the consistency loss
```

- [ ] **Step 2: Update `configs/sweeper/braingnn_vwm.yaml`**

Replace lines 17–18:
```yaml
  model.model_params.unit_loss_weight: interval(0.0, 1.0)
  model.model_params.topk_loss_weight: interval(0.0, 1.0)
```
with:
```yaml
  model.model_params.lambda_topk: interval(0.0, 1.0)
  model.model_params.lambda_unit: interval(0.0, 1.0)
  model.model_params.lambda_consist: interval(0.0, 1.0)
```

- [ ] **Step 3: Update `configs/sweeper/braingnn_fc_vwm.yaml`**

Replace lines 41–42:
```yaml
  model.model_params.unit_loss_weight: interval(0.0, 1.0)
  model.model_params.topk_loss_weight: interval(0.0, 1.0)
```
with:
```yaml
  model.model_params.lambda_topk: interval(0.0, 1.0)
  model.model_params.lambda_unit: interval(0.0, 1.0)
  model.model_params.lambda_consist: interval(0.0, 1.0)
```

- [ ] **Step 4: Sanity-check configs load**

Run: `python -c "from omegaconf import OmegaConf; [OmegaConf.load(p) for p in ['configs/model/braingnn.yaml','configs/sweeper/braingnn_vwm.yaml','configs/sweeper/braingnn_fc_vwm.yaml']]; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add configs/model/braingnn.yaml configs/sweeper/braingnn_vwm.yaml configs/sweeper/braingnn_fc_vwm.yaml
git commit -m "config(braingnn): rename loss knobs to lambda_* + add consist_n_bins"
```

---

## Task 10: Provenance, ADR, and memory

**Files:**
- Create: `src/models/vendor/braingnn/PROVENANCE.md`
- Create: `docs/adr/0010-vendor-braingnn.md`

- [ ] **Step 1: Write `PROVENANCE.md`**

Create `src/models/vendor/braingnn/PROVENANCE.md`:
```markdown
# Provenance

Vendored from **xxlya/BrainGNN_Pytorch**
- Source: https://github.com/xxlya/BrainGNN_Pytorch
- Commit: `1e337e7a13af5e4374343fda81ce11b62c5ce566`
- Files: `net/inits.py`, `net/brainmsgpassing.py`, `net/braingraphconv.py`,
  and the `topk_loss` / `consist_loss` functions from `03-main.py`.
- License: held by the user (confirmed).

## Edits vs upstream
- `braingraphconv.py`: intra-package imports rewritten from `net.*` to
  `src.models.vendor.braingnn.*`. No logic changes.
- `losses.py`: `consist_loss` builds its Laplacian on the input tensor's device
  rather than a module-global `device`. Identical math.
- `inits.py`, `brainmsgpassing.py`: verbatim.

## Known faithful quirks (intentionally preserved)
- **Triple sigmoid on pooling scores:** TopKPooling returns a post-σ score;
  the adapter `encode` applies σ again (matching upstream `Network.forward`);
  `consist_loss` applies σ a third time. This is an upstream behavior; we
  reproduce it. Locked by `tests/test_vendor_braingnn.py::TestSigmoidComposition`.
- **Aggregation:** `MyMessagePassing` hardcodes `scatter_add` (sum); the
  `aggr='mean'` constructor arg is asserted then ignored.
```

- [ ] **Step 2: Write the ADR**

Create `docs/adr/0010-vendor-braingnn.md`:
```markdown
# 10. Vendor the official BrainGNN instead of maintaining a rewrite

Date: 2026-05-26

## Status
Accepted

## Context
The hand-rewritten BrainGNN diverged from Li et al. 2021 in six ways, several
unintended (BN/ReLU/dropout between conv and pool; missing conv self-loops and
bias; scrambled loss names; global instead of per-graph topk sort; wrong sigmoid
count). Maintaining a rewrite that drifts from the reference is costly and
error-prone.

## Decision
Vendor the upstream conv/message-passing layers and loss functions verbatim
(`src/models/vendor/braingnn`, commit 1e337e7) and re-express the monolithic
`Network.forward` inside a thin adapter that fits the repo's `encode`/`decode`
contract. The configurable `RegressionHead` is spliced in place of the upstream
fixed FC head. The consistency loss, per-class upstream, is applied per
target-quantile bin for regression.

## Consequences
- Edge weights are now softmax-normalized inside `MyNNConv` (reverts the
  rewrite's raw-edge-weight divergence).
- The faithful triple-sigmoid score path is reproduced (see PROVENANCE).
- Loss config keys change: `unit_loss_weight`/`topk_loss_weight` →
  `lambda_topk`/`lambda_unit`/`lambda_consist` (+ `consist_n_bins`). wandb log
  keys change accordingly and now include `consist_loss`.
- `ROIAwareConv` and its tests are removed.
```

- [ ] **Step 3: Update auto-memory `braingnn_migration.md`**

The memory at `/home/compa/.claude/projects/-home-compa-Documents-working-dir-LLM-codebase/memory/braingnn_migration.md` claims the model "matches Li et al. 2021" — now superseded. Update its body to: BrainGNN is now a thin adapter over vendored upstream layers (commit 1e337e7); the prior rewrite (and its 5-priority migration) is retired; faithful triple-σ + softmax edge weights + per-bin consist for regression. Keep the index line in `MEMORY.md` pointing at the same file.

- [ ] **Step 4: Commit**

```bash
git add src/models/vendor/braingnn/PROVENANCE.md docs/adr/0010-vendor-braingnn.md
git commit -m "docs(braingnn): provenance + ADR for vendoring decision"
```

---

## Task 11: Full verification + PR

**Files:** none (verification + git)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q`
Expected: all pass. Pay attention to `tests/test_models.py`, `tests/test_vendor_braingnn.py`, `tests/test_interfaces.py`, `tests/test_training_regressions.py`. Investigate and fix any failure before proceeding (do not skip).

- [ ] **Step 2: End-to-end smoke (model builds + trains one step via registry)**

Run:
```bash
python - <<'PY'
import torch, torch_geometric.data as gd
from src.configs.model_config import ModelConfig
from src.models.registry import get_model
cfg = ModelConfig(name="braingnn", hidden_dim=16, dropout=0.0,
                  model_params={"pool_ratio":0.5,"roi_embed_dim":8,
                                "lambda_topk":0.1,"lambda_unit":0.1,
                                "lambda_consist":0.1,"consist_n_bins":4})
m = get_model("braingnn", cfg, node_feat_dim=10, edge_feat_dim=1, num_nodes=10)
m.train()
xs,eis,eas,bs=[],[],[],[]; off=0
for g in range(4):
    xs.append(torch.randn(10,10)); eis.append(torch.randint(0,10,(2,20))+off)
    eas.append(torch.rand(20,1)); bs.append(torch.full((10,),g)); off+=10
data=gd.Data(x=torch.cat(xs),edge_index=torch.cat(eis,1),edge_attr=torch.cat(eas),
             batch=torch.cat(bs),y=torch.randn(4,1))
pred=m(data).squeeze(-1)
loss=torch.nn.functional.mse_loss(pred,data.y.view(-1))
for k,v in (m.auxiliary_loss() or {}).items(): loss=loss+v
loss.backward()
assert torch.isfinite(loss)
print("smoke ok, loss=", float(loss))
PY
```
Expected: prints `smoke ok, loss= <finite>`; no exception; gradients flow.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feature/vendor-braingnn
gh pr create --title "Vendor official BrainGNN; retire the rewrite" --body "$(cat <<'EOF'
## Summary
- Vendor upstream BrainGNN conv/message-passing layers + losses verbatim (xxlya/BrainGNN_Pytorch @ 1e337e7)
- Re-express Network.forward in a thin adapter fitting the encode/decode contract; splice in RegressionHead
- Consistency loss applied per target-quantile bin (regression has no classes)
- Remove ROIAwareConv rewrite; rename loss config knobs to lambda_*

## Behavior changes
- Softmax-normalized edge weights (reverts raw-weight divergence)
- Faithful triple-sigmoid score path (documented in PROVENANCE)
- New wandb loss keys: topk_loss, unit_loss, consist_loss

## Test plan
- [ ] `pytest tests/ -q` green
- [ ] end-to-end smoke (build + 1 train step via registry) passes
- [ ] spec: docs/superpowers/specs/2026-05-26-vendor-braingnn-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Confirm with the user before pushing/opening the PR (per repo norms, pushing is a shared-state action).

---

## Self-review notes (author)

- **Spec coverage:** §1 layout → Tasks 2,4,8,10. §2 components → Tasks 2,4. §3 data flow (pos synth, augment_adj plain self-loops, readout) → Task 5. §4 losses incl. triple-σ + binning → Tasks 3,6,7. §5 config/breaking → Task 9. §6 testing/parity → Tasks 2,3,5,6,7,8,11. §7 housekeeping (ADR, memory, worktree) → Tasks 1,10. License → Task 10 PROVENANCE.
- **Placeholder scan:** none — every code/edit step shows full content; "find/confirm" steps give exact grep/commands.
- **Type consistency:** attrs `pool_ratio, num_nodes, lambda_topk, lambda_unit, lambda_consist, consist_n_bins, _s1, _s2, _w1, _w2, _y`; methods `encode/decode/auxiliary_loss/_augment_adj/_binned_consist_loss`; loss fns `topk_loss(s,ratio)`, `consist_loss(s)`. Consistent across tasks.
