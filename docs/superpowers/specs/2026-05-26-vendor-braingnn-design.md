# Design: Vendor the official BrainGNN into LLM_codebase

**Date:** 2026-05-26
**Status:** Approved (brainstorming) — pending spec review
**Branch (to be created):** `feature/vendor-braingnn`

## Goal

Stop maintaining a hand-rewritten BrainGNN. Reuse the official
[`xxlya/BrainGNN_Pytorch`](https://github.com/xxlya/BrainGNN_Pytorch) code as
verbatim as possible for the scientifically-critical layers, behind a thin
adapter that satisfies this repo's `BrainGNN` model contract.

### Why vendoring

The current rewrite (`src/models/braingnn_model.py` + `backbones/roi_aware_conv.py`)
carries six divergences from upstream, several of them unintended bugs:

1. BN+ReLU+dropout inserted between conv and pool (upstream has none there).
2. No `fill=1` self-loops inside the conv (data-dependent node self-message).
3. Missing conv bias.
4. **Loss names scrambled:** codebase `"unit_loss"` is upstream's `topk_loss`;
   codebase `"topk_loss"` is upstream's `consist_loss`; upstream's real unit
   loss `(‖w‖₂−1)²` is absent.
5. topk/entropy loss sorts **globally across the batch**; upstream sorts
   **per-graph**.
6. Sigmoid stacking wrong: the rewrite applies a *double* sigmoid where
   faithful upstream is a *triple* sigmoid (see §4) — so even the "bug" was not
   a faithful reproduction.

Vendoring the conv/message-passing layers verbatim erases 1–3 and 6 by
construction; re-expressing the forward pass faithfully fixes 4–5.

## Decisions settled in brainstorming

- **Prediction head:** splice in the existing `RegressionHead`. The adapter's
  `encode` returns the `hidden*4` readout; `decode` runs `RegressionHead`
  (driven by `cfg.head_hidden_dim` / `cfg.head_num_layers`), keeping head
  sizing consistent with the other models in this repo.
- **Consistency loss for regression:** bin the continuous target into quantile
  pseudo-classes (`consist_n_bins`, default 4) and apply the upstream Laplacian
  consistency term per bin. The Laplacian block stays verbatim; only the
  grouping (bins instead of `y == c`) changes.
- **Vendoring scope (Approach B):** vendor the layer files verbatim; re-express
  `Network.forward` inside the adapter. The monolithic upstream `Network` cannot
  be reused intact anyway, because the `BrainGNN` contract splits `forward` into
  `encode`/`decode` and we are swapping the head.

## 1. File layout

```
src/models/vendor/braingnn/
  __init__.py
  brainmsgpassing.py      # verbatim: MyMessagePassing
  braingraphconv.py       # verbatim: MyNNConv
  inits.py                # verbatim: uniform
  PROVENANCE.md           # upstream URL + commit SHA + license + noted quirks
src/models/braingnn_model.py   # rewritten adapter
```

Delete `src/models/backbones/roi_aware_conv.py` after confirming no other
importer remains.

**License:** the user holds the license to use this upstream code. Record the
source URL, pinned commit SHA, and attribution in `PROVENANCE.md`.

## 2. Components

### Vendored layers (verbatim, untouched)

- **`MyMessagePassing`** (`brainmsgpassing.py`): self-contained manual
  `propagate` built on `torch_scatter.scatter_add`. Note: aggregation is
  **hardcoded to `scatter_add` (sum)** — the `aggr='mean'` constructor argument
  is asserted then ignored (the `scatter_(self.aggr, …)` line is commented out).
  Combined with softmax-normalized edge weights in `message`, the effective
  operation is a weighted sum. Faithful porting preserves this exactly.
- **`MyNNConv`** (`braingraphconv.py`): per-ROI weight matrix via `nn(pos)`;
  `add_remaining_self_loops(..., fill_value=1, ...)` **inside `forward`** (this
  is where the `fill=1` self-loops live — not in `augment_adj`);
  `message` applies `softmax(edge_weight, edge_index_i, …)`; `update` adds bias
  and optional L2 normalize (`normalize=False` as used). Traced OK on PyG 2.7.
- **`uniform`** (`inits.py`): required by `MyNNConv.reset_parameters`.

### Adapter `BrainGNNModel(BrainGNN)`

Constructor `(cfg, node_feat_dim, edge_feat_dim, num_nodes, **kwargs)` builds:

- Two `n_k = Sequential(Linear(R, k, bias=False), ReLU, Linear(k, in*out))`
  → two `MyNNConv(in, out, n_k, normalize=False)`.
- Two `TopKPooling(hidden, ratio=pool_ratio, multiplier=1, nonlinearity=torch.sigmoid)`.
- `RegressionHead(cfg, embedding_dim=hidden*4)`.

**Dims:** upstream hardcodes `dim1 = dim2 = 32`. We parametrize both by
`cfg.hidden_dim` (=64 in current configs), as the current rewrite already does.
Readout dimensionality is `hidden*4` (`[max1 | mean1 | max2 | mean2]`).

`@register_model("braingnn")` is preserved. `cfg.fusion is not None` and
`num_nodes <= 0` raise, as today.

## 3. Data flow (`encode` / `decode`)

`encode(data)` ports the upstream `Network.forward` faithfully:

1. `edge_weight = edge_attr[:, 0]` (or ones if absent); synthesize
   `pos = one_hot(arange(R).repeat(B)).float()`.
2. `x = conv1(x, edge_index, edge_weight, pos)`;
   `x, edge_index, edge_weight, batch, perm, score1 = pool1(...)`;
   `pos = pos[perm]`; `x1 = cat([gmp(x,batch), gap(x,batch)])`.
3. **`augment_adj`** (A² on the pooled subgraph): plain
   `add_self_loops` (default `fill_value`), `sort_edge_index`/coalesce,
   `spspmm(...)`, `remove_self_loops`. Uses **plain `add_self_loops`**, NOT
   `add_remaining_self_loops(fill=1)` — that belongs to `MyNNConv`. The current
   rewrite (`braingnn_model.py:143-153`) already matches this correctly and runs
   on PyG 2.7; port it as-is.
4. `x = conv2(x, edge_index, edge_weight, pos)`;
   `x, ..., perm, score2 = pool2(...)`; `pos = pos[perm]`;
   `x2 = cat([gmp, gap])`.
5. `readout = cat([x1, x2])` → `[B, hidden*4]`.
   Stash on `self` for `auxiliary_loss`: `w1 = pool1.select.weight`,
   `w2 = pool2.select.weight`, `s1 = score1`, `s2 = score2` (the pools' returned
   per-graph kept-node scores), and `y = data.y`.

`decode(emb) = RegressionHead(emb)` → `[B, 1]`. `forward` remains the base-class
final `encode → decode` (no override).

**PyG 2.7 note:** `TopKPooling` has no `.weight`; the gating weight is
`.select.weight`. Constructor:
`TopKPooling(in_channels, ratio=0.5, min_score=None, multiplier=1.0, nonlinearity='tanh'|Callable)`;
`nonlinearity=torch.sigmoid` is accepted.

Re-expressing the forward this way fixes the rewrite's global-sort bug (we use
the pools' per-graph kept scores directly) and removes the spurious
BN/ReLU/dropout-between-conv-and-pool.

## 4. Auxiliary losses

Vendor `topk_loss` and `consist_loss` **verbatim from upstream `03-main.py`**;
add the one-line unit loss. `auxiliary_loss()` (no args; returns pre-scaled
scalars; `None` in eval):

- **topk:** `topk_loss(s1, ratio) + topk_loss(s2, ratio)` — per-graph sort
  (`dim=1`). Reshape stashed scores to `[B, n_kept]` (uniform, since the atlas
  size `R` is fixed across subjects).
- **unit:** `(‖w1‖₂ − 1)² + (‖w2‖₂ − 1)²`.
- **consist:** quantize `self._y` (the current batch's targets) into
  `consist_n_bins` quantile bins computed **per forward pass**; apply the
  upstream Laplacian consistency block (`W = ones`, `tr(SᵀLS)`) per bin; average
  over non-empty bins. The Laplacian block is verbatim; only the grouping
  changes. (A bin with a single subject contributes zero, matching the Laplacian
  definition.)

Returns `{"topk_loss": …, "unit_loss": …, "consist_loss": …}`, each scaled by
its λ from `cfg.model_params`.

### Sigmoid stacking (exact — faithful reproduction)

The upstream score path is a **triple sigmoid**, and it must be reproduced
faithfully:

1. `TopKPooling` returns the **post-σ** gating score (it applies
   `nonlinearity=sigmoid` internally).
2. `Network.forward` (`braingnn.py:75`) applies `torch.sigmoid` **again** to the
   returned score before it reaches `topk_loss` → double-σ.
3. `consist_loss` (`03-main.py:107`) applies `torch.sigmoid` a **third** time.

The current rewrite applied only a double sigmoid — i.e. it was *missing* a
layer, not faithfully reproducing upstream. Faithful reuse means reproducing the
full triple-σ stack. This will be locked by a numeric parity test and documented
in `PROVENANCE.md`.

## 5. Config & breaking changes

`cfg.model_params` knobs: `pool_ratio`, `roi_embed_dim` (k), `lambda_topk`,
`lambda_unit`, `lambda_consist`, `consist_n_bins`.

**Breaking changes (confirmed):**

- Loss keys change. The old `unit_loss_weight` / `topk_loss_weight` keys appear
  in `configs/model/braingnn.yaml` and two sweeper files
  (`configs/sweeper/braingnn_vwm.yaml`, `configs/sweeper/braingnn_fc_vwm.yaml`);
  all three need updating to the new λ knobs. wandb log keys
  (`fold_N/train/<name>`) change accordingly (now correctly-named `topk_loss`,
  `unit_loss`, plus new `consist_loss`).
- Edge weights become **softmax-normalized** inside `MyNNConv.message`,
  reverting the rewrite's intentional "raw edge weights" divergence — an
  accepted consequence of vendoring.

## 6. Testing / parity

- **Vendored layers:** import smoke test + a tiny numeric check of per-ROI
  weighting in `MyNNConv`.
- **Contract:** `encode → [B, hidden*4]`, `decode → [B, 1]`, `forward` final;
  construction via `registry.get_model("braingnn", cfg, node_feat_dim=…,
  edge_feat_dim=…, num_nodes=…)`.
- **Aux losses:** per-graph topk sort; unit-loss formula; consist binning groups
  correctly and averages; all terms pre-scaled scalars; `auxiliary_loss()`
  returns `None` in eval.
- **Parity test:** one `conv → pool → readout` pass matches an upstream-faithful
  reference wiring numerically, locking the triple-σ score path.
- **Training smoke** (`tests/test_training_regressions.py`): a step runs; loss
  finite and decreasing on toy data.
- **Existing tests to remove/rewrite:** `tests/test_models.py` has dedicated
  `ROIAwareConv` tests and one asserting the model has no `roi_embedding`
  attribute — these are removed/rewritten when `ROIAwareConv` is deleted. The
  ~33 braingnn test references that assume old loss names / global-sort
  semantics get updated to the new behavior.

## 7. Migration housekeeping

- New ADR under `docs/adr/` documenting the vendoring decision and the accepted
  behavior changes (softmax edge weights, triple-σ, loss renames).
- Update memory `braingnn_migration.md` — its "matches Li et al. 2021" claim is
  too strong given the divergences found.
- Per `CLAUDE.md`: all work in a `git worktree` + `feature/vendor-braingnn`
  branch, merged to `main` via PR, with a commit after each meaningful change.

## Out of scope

- No changes to the dataset/edge-construction pipeline (`pnc_dataset.py`).
- No changes to the trainer's aux-loss consumption (`trainer.py:488-494`) —
  the adapter conforms to the existing contract.
- No new model variants or sweep experiments beyond updating the existing
  braingnn configs to the renamed knobs.
