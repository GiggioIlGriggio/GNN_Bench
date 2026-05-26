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
- The adapter's `encode` stashes the pooling gate weights via PyG 2.x's
  `pool.select.weight` path. Upstream (PyG 1.x) used `pool.weight`; in PyG 2.7
  `TopKPooling` exposes the same `[1, hidden]` gate parameter at
  `.select.weight`. Same tensor, version-adapted attribute path.

## Known faithful quirks (intentionally preserved)
- **Triple sigmoid on pooling scores:** TopKPooling returns a post-σ score;
  the adapter `encode` applies σ again (matching upstream `Network.forward`);
  `consist_loss` applies σ a third time. This is an upstream behavior; we
  reproduce it. Locked by `tests/test_vendor_braingnn.py::TestSigmoidComposition`.
- **Aggregation:** `MyMessagePassing` hardcodes `scatter_add` (sum); the
  `aggr='mean'` constructor arg is asserted then ignored.
