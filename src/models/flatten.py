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
