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
