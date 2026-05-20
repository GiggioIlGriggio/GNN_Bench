"""
Instrumented BrainGNN forward pass inspector.
Run from repo root:  python inspect_braingnn.py

Shows tensor shapes and statistics at every architectural stage:
  fc_row features → ROIAwareConv internals → TopK scores → pool1 →
  augment_adj → conv2 → pool2 → dual readout → regression head
"""

from __future__ import annotations
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index
from torch_sparse import spspmm

from src.configs.feature_config import FeatureConfig
from src.configs.model_config import ModelConfig
from src.datasets.base_dataset import RawGraphData
from src.datasets.feature_builder import FeatureBuilder
from src.models.braingnn_model import BrainGNNModel

# ─────────────────────────── settings ────────────────────────────────────────
N      = 20    # ROIs per subject (small so output is readable)
B      = 3     # number of subjects (graphs in the batch)
HIDDEN = 32    # hidden dim
K      = 8     # k for ROIAwareConv (roi_embed_dim in config)
RATIO  = 0.5   # TopK pool ratio
SEED   = 0
torch.manual_seed(SEED)
_EPS   = 1e-8

# ─────────────────────────── helpers ─────────────────────────────────────────
SEP = "─" * 65

def hdr(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def stats(t: torch.Tensor | None, label: str, indent: int = 2) -> None:
    pad = " " * indent
    if t is None:
        print(f"{pad}{label}: None")
        return
    f = t.float()
    print(f"{pad}{label:30s} shape={tuple(t.shape)}"
          f"  mean={f.mean():+.4f}  std={f.std():.4f}"
          f"  min={f.min():+.4f}  max={f.max():+.4f}")


# ─────────────────────────── synthetic data ───────────────────────────────────
def synthetic_fc(n: int) -> torch.Tensor:
    """Symmetric FC correlation matrix with unit diagonal."""
    A = torch.randn(n, n)
    A = (A + A.T) / 2
    A.fill_diagonal_(1.0)
    return A

def fc_to_raw(fc: torch.Tensor, sid: str) -> RawGraphData:
    n = fc.size(0)
    rows, cols = torch.where(fc != 0)
    return RawGraphData(
        subject_id=sid,
        fc_edge_index=torch.stack([rows, cols]),
        fc_edge_attr=fc[rows, cols].unsqueeze(-1),
        num_nodes=n,
    )

feat_cfg = FeatureConfig(
    node_features=["fc_row"],
    edge_features=["weight"],
    node_feat_dim=N,
    edge_feat_dim=1,
)
builder = FeatureBuilder(feat_cfg)

x_list, ei_list, ea_list, batch_list = [], [], [], []
offset = 0
for g in range(B):
    raw = fc_to_raw(synthetic_fc(N), f"sub-{g:03d}")
    x_list.append(builder.build_node_features(raw))
    ei_list.append(raw.fc_edge_index + offset)
    ea_list.append(raw.fc_edge_attr)                      # [E, 1]
    batch_list.append(torch.full((N,), g, dtype=torch.long))
    offset += N

data = torch_geometric.data.Data(
    x=torch.cat(x_list, dim=0),
    edge_index=torch.cat(ei_list, dim=1),
    edge_attr=torch.cat(ea_list, dim=0),
    batch=torch.cat(batch_list, dim=0),
)

# ─────────────────────────── model ───────────────────────────────────────────
cfg = ModelConfig(
    name="braingnn",
    hidden_dim=HIDDEN,
    dropout=0.0,   # disable for reproducible inspection
    model_params={"pool_ratio": RATIO, "roi_embed_dim": K,
                  "unit_loss_weight": 0.3, "topk_loss_weight": 0.5},
)
model = BrainGNNModel(cfg, node_feat_dim=N, edge_feat_dim=1, num_nodes=N)
model.eval()

n_params = sum(p.numel() for p in model.parameters())

# ═══════════════════════════ inspection ══════════════════════════════════════

hdr("MODEL OVERVIEW")
print(f"  N={N} ROIs  B={B} subjects  hidden={HIDDEN}  K={K}  ratio={RATIO}")
print(f"  conv1: ROIAwareConv({N} → {HIDDEN})   [n_k: {N}→{K}→{N*HIDDEN} params]")
print(f"  pool1: TopKPooling(ratio={RATIO}, sigmoid)")
print(f"  conv2: ROIAwareConv({HIDDEN} → {HIDDEN})")
print(f"  pool2: TopKPooling(ratio={RATIO}, sigmoid)")
print(f"  readout: cat([max1,mean1,max2,mean2])  dim = {HIDDEN*4}")
print(f"  head:  Linear({HIDDEN*4}) → Linear({cfg.head_hidden_dim}) → Linear(1)")
print(f"  total params: {n_params:,}")

hdr("RAW INPUT  (fc_row features)")
stats(data.x, "x  [B*N, N]")
stats(data.edge_attr, "edge_attr  [E_total, 1]")
print(f"  total nodes: {data.x.size(0)}  total edges: {data.edge_index.size(1)}")
print(f"  batch: { {int(g): int((data.batch==g).sum()) for g in range(B)} }  (nodes per graph)")

with torch.no_grad():
    x         = data.x
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    batch      = data.batch
    num_graphs = B
    roi_idx    = torch.arange(N).repeat(num_graphs)   # [B*N]

    # ── ROIAwareConv internals ────────────────────────────────────────────────
    hdr("STAGE 1 — ROIAwareConv internals")
    pos = F.one_hot(roi_idx, N).float()                       # [B*N, N]
    raw_weights = model.conv1.n(pos)                          # [B*N, N*HIDDEN]
    W = raw_weights.view(-1, N, HIDDEN)                        # [B*N, N, HIDDEN]
    x_bmm = torch.bmm(x.unsqueeze(1), W).squeeze(1)           # [B*N, HIDDEN]

    stats(pos,        "one-hot ROI encoding  [B*N, N]")
    stats(raw_weights,"n_k output  [B*N, N*HIDDEN]     (raw per-ROI weight vecs)")
    stats(W,          "weight matrices  [B*N, N, HIDDEN]  (reshaped)")
    stats(x_bmm,      "x after bmm (pre message-pass)  [B*N, HIDDEN]")

    # show weight diversity — how different are two ROIs' projection matrices?
    w0 = W[0].flatten()     # ROI 0 weight vector
    w1 = W[1].flatten()     # ROI 1 weight vector
    cos_sim = F.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0)).item()
    print(f"\n  cosine similarity between ROI-0 and ROI-1 weight matrices: {cos_sim:.4f}")
    print(f"  (0 = orthogonal/fully unique, 1 = identical/shared — measures per-ROI diversity)")

    # full conv1 + BN + ReLU
    x_conv1 = model.conv1(x, edge_index, edge_attr[:, 0], roi_idx)
    x_bn1   = model.bn1(x_conv1)
    x_relu1 = F.relu(x_bn1)

    hdr("STAGE 1 — After conv1 + BN + ReLU")
    stats(x_conv1, "after conv1   [B*N, HIDDEN]")
    stats(x_bn1,   "after BN1     [B*N, HIDDEN]")
    stats(x_relu1, "after ReLU1   [B*N, HIDDEN]")

    # ── Pool1 scores ─────────────────────────────────────────────────────────
    hdr("POOL 1 — Score distribution (sigmoid nonlinearity)")
    w_sel  = model.pool1.select.weight                          # [1, HIDDEN]
    raw_sc = (x_relu1 * w_sel).sum(dim=-1)
    scores1 = model.pool1.select.act(raw_sc / (w_sel.norm(p=2, dim=-1) + _EPS))

    stats(scores1, "pool1 scores  [B*N]  (sigmoid → (0,1))")
    k_keep = max(int(N * RATIO), 1)
    print(f"\n  Keeping top-{k_keep} of {N} ROIs per graph:")
    scores_per_graph = scores1.reshape(B, N)
    for g in range(B):
        sc   = scores_per_graph[g]
        topk = sc.topk(k_keep).indices.sort().values.tolist()
        botk = sc.topk(k_keep, largest=False).indices.sort().values.tolist()
        print(f"    graph {g}:  selected ROIs={topk}  (scores {sc[topk].tolist()})")
        print(f"              dropped  ROIs={botk}  (scores {sc[botk].tolist()})")

    x_p1, ei_p1, ea_p1, batch_p1, perm1, _ = model.pool1(x_relu1, edge_index, edge_attr, batch)
    roi_idx_p1 = roi_idx[perm1]

    hdr("AFTER POOL1")
    stats(x_p1,  "x  [B*k1, HIDDEN]")
    stats(ea_p1, "edge_attr  [E_p1, 1]")
    print(f"  nodes: {B*N} → {x_p1.size(0)}  |  edges: {edge_index.size(1)} → {ei_p1.size(1)}")
    print(f"  surviving ROI indices per graph:")
    for g in range(B):
        mask = batch_p1 == g
        print(f"    graph {g}: {sorted(roi_idx_p1[mask].tolist())}")

    # ── Readout x1 ───────────────────────────────────────────────────────────
    x1 = torch.cat([global_max_pool(x_p1, batch_p1), global_mean_pool(x_p1, batch_p1)], dim=1)

    hdr("READOUT x1  (max+mean after pool1)")
    stats(x1, "x1  [B, HIDDEN*2]")

    # ── Augment adjacency ────────────────────────────────────────────────────
    hdr("AUGMENT ADJ  (A² = A@A on pooled subgraph)")
    n1   = x_p1.size(0)
    ew   = ea_p1[:, 0] if ea_p1.dim() > 1 else ea_p1
    ei_a, ew_a = add_self_loops(ei_p1, ew, num_nodes=n1)
    ei_a, ew_a = sort_edge_index(ei_a, ew_a, num_nodes=n1)
    ei_a, ew_a = spspmm(ei_a, ew_a, ei_a, ew_a, n1, n1, n1)
    ei_a, ew_a = remove_self_loops(ei_a, ew_a)
    ea_a = ew_a.unsqueeze(-1)

    print(f"  edges before A²:  {ei_p1.size(1)}")
    print(f"  edges after  A²:  {ei_a.size(1)}  (2-hop neighbours now directly connected)")
    stats(ea_a, "edge weights after A²")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    x_conv2 = model.conv2(x_p1, ei_a, ew_a, roi_idx_p1)
    x_bn2   = model.bn2(x_conv2)
    x_relu2 = F.relu(x_bn2)

    hdr("STAGE 2 — After conv2 + BN + ReLU")
    stats(x_conv2, "after conv2  [B*k1, HIDDEN]")
    stats(x_relu2, "after ReLU2  [B*k1, HIDDEN]")

    # ── Pool2 ────────────────────────────────────────────────────────────────
    hdr("POOL 2 — Score distribution")
    w_sel2  = model.pool2.select.weight
    raw_sc2 = (x_relu2 * w_sel2).sum(dim=-1)
    scores2 = model.pool2.select.act(raw_sc2 / (w_sel2.norm(p=2, dim=-1) + _EPS))

    stats(scores2, "pool2 scores  [B*k1]  (sigmoid → (0,1))")
    k2 = max(int(x_p1.size(0) // B * RATIO), 1)
    print(f"  Keeping ~top-{k2} ROIs per graph from pool1's {x_p1.size(0)//B}:")
    x_p2, ei_p2, ea_p2, batch_p2, perm2, _ = model.pool2(x_relu2, ei_a, ea_a, batch_p1)
    roi_idx_p2 = roi_idx_p1[perm2]

    hdr("AFTER POOL2")
    stats(x_p2, "x  [B*k2, HIDDEN]")
    print(f"  nodes: {x_p1.size(0)} → {x_p2.size(0)}  |  edges: {ei_a.size(1)} → {ei_p2.size(1)}")
    print(f"  surviving ROI indices per graph:")
    for g in range(B):
        mask = batch_p2 == g
        print(f"    graph {g}: {sorted(roi_idx_p2[mask].tolist())}")

    # ── Readout x2 + final embedding ─────────────────────────────────────────
    x2 = torch.cat([global_max_pool(x_p2, batch_p2), global_mean_pool(x_p2, batch_p2)], dim=1)
    embedding = torch.cat([x1, x2], dim=1)

    hdr("FINAL EMBEDDING  [max1 | mean1 | max2 | mean2]")
    stats(x2,        "x2            [B, HIDDEN*2]")
    stats(embedding, "embedding     [B, HIDDEN*4]")
    print(f"\n  Embedding breakdown per subject:")
    for g in range(B):
        e = embedding[g]
        print(f"    subject {g}: max1={e[:HIDDEN].norm():.3f}  mean1={e[HIDDEN:2*HIDDEN].norm():.3f}"
              f"  max2={e[2*HIDDEN:3*HIDDEN].norm():.3f}  mean2={e[3*HIDDEN:].norm():.3f}  (L2 norms of segments)")

    # ── Head ─────────────────────────────────────────────────────────────────
    out = model.decode(embedding)
    hdr("REGRESSION HEAD OUTPUT")
    stats(out, "predictions  [B, 1]")
    print(f"\n  predictions: { {g: round(out[g, 0].item(), 4) for g in range(B)} }")

    # ── Auxiliary losses ─────────────────────────────────────────────────────
    hdr("AUXILIARY LOSSES  (training mode)")
    model.train()
    _ = model.encode(data)   # re-run to populate _pool_scores with gradients
    aux = model.auxiliary_loss()
    if aux:
        for k, v in aux.items():
            print(f"  {k}: {v.item():.6f}")

print(f"\n{'─'*65}")
print("  Inspection complete.")
print(f"{'─'*65}\n")
