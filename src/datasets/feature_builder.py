"""Declarative node and edge feature assembly.

Design rules
------------
- Each feature is a **named, independently toggled method**.
- Enabling a feature in the YAML and adding one ``_node_feat_<name>`` or
  ``_edge_feat_<name>`` method here is the **only** change needed to add a new
  feature — no other file should change.
"""

from __future__ import annotations

from typing import Dict, Callable, List, Literal

import numpy as np
import torch

from src.configs.feature_config import FeatureConfig
from src.datasets.base_dataset import RawGraphData


class FeatureBuilder:
    """Assembles node and edge features from a declarative config.

    Parameters
    ----------
    cfg : FeatureConfig
        Specifies which features are enabled and expected dimensionalities.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self._node_feat_registry: Dict[str, Callable[..., torch.Tensor]] = (
            self._discover_methods("_node_feat_")
        )
        self._edge_feat_registry: Dict[str, Callable[..., torch.Tensor]] = (
            self._discover_methods("_edge_feat_")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_node_features(self, graph_data: RawGraphData) -> torch.Tensor:
        """Compute all enabled node features and concatenate.

        Parameters
        ----------
        graph_data : RawGraphData
            Per-subject raw graph data.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, node_feat_dim]``.

        Raises
        ------
        ValueError
            If a requested feature name has no matching method.
        """
        feats = []
        for name in self.cfg.node_features:
            if name not in self._node_feat_registry:
                raise ValueError(
                    f"Unknown node feature '{name}'. "
                    f"Available: {list(self._node_feat_registry.keys())}"
                )
            feats.append(self._node_feat_registry[name](graph_data))
        result = torch.cat(feats, dim=-1)

        # Validate node_feat_dim for features whose width equals num_nodes.
        _NODE_SIZED_FEATURES = {"identity", "fc_row", "sc_row"}
        if any(name in _NODE_SIZED_FEATURES for name in self.cfg.node_features):
            actual_dim = result.shape[-1]
            if actual_dim != self.cfg.node_feat_dim:
                raise ValueError(
                    f"node_feat_dim mismatch: config says {self.cfg.node_feat_dim} "
                    f"but actual feature dim is {actual_dim}. "
                    f"node_feat_dim must match num_nodes for features: "
                    f"{_NODE_SIZED_FEATURES & set(self.cfg.node_features)}."
                )

        return result

    def build_edge_features(self, graph_data: RawGraphData) -> torch.Tensor:
        """Compute all enabled edge features and concatenate.

        Parameters
        ----------
        graph_data : RawGraphData
            Per-subject raw graph data.

        Returns
        -------
        torch.Tensor
            Shape ``[num_edges, edge_feat_dim]``.

        Raises
        ------
        ValueError
            If a requested feature name has no matching method.
        """
        feats = []
        for name in self.cfg.edge_features:
            if name not in self._edge_feat_registry:
                raise ValueError(
                    f"Unknown edge feature '{name}'. "
                    f"Available: {list(self._edge_feat_registry.keys())}"
                )
            feats.append(self._edge_feat_registry[name](graph_data))
        return torch.cat(feats, dim=-1)

    def get_node_feat_dim(self) -> int:
        """Return the total node feature dimensionality."""
        return self.cfg.node_feat_dim

    def get_edge_feat_dim(self) -> int:
        """Return the total edge feature dimensionality."""
        return self.cfg.edge_feat_dim

    def get_glm_column_range(self) -> tuple[int, int] | None:
        """Return the ``(start, end)`` column range of GLM features in ``data.x``.

        Scans ``cfg.node_features`` in order and sums the per-feature
        dimensionality to locate where GLM columns begin and end.

        Returns
        -------
        tuple[int, int] | None
            ``(col_start, col_end)`` with ``col_end`` exclusive, or ``None``
            if no GLM features are enabled.
        """
        glm_names = {"glm_scalar", "glm_diagonal"}
        enabled_glm = [f for f in self.cfg.node_features if f in glm_names]
        if not enabled_glm:
            return None

        num_contrasts = len(self.cfg.glm_contrasts)
        if num_contrasts == 0:
            return None

        # Map each non-GLM feature to its dimensionality.
        _SCALAR_FEATURES = {
            "degree", "strength", "betweenness", "clustering",
        }

        offset = 0
        col_start = None
        col_end = None

        for name in self.cfg.node_features:
            if name in _SCALAR_FEATURES:
                dim = 1
            elif name == "glm_scalar":
                dim = num_contrasts
                if col_start is None:
                    col_start = offset
                col_end = offset + dim
            elif name == "glm_diagonal":
                dim = self.cfg.node_feat_dim  # N * num_contrasts — read from cfg
                # For diagonal mode the dimension is num_nodes * num_contrasts.
                # Since num_nodes is fixed (atlas-level), infer from
                # node_feat_dim minus the sum of all other feature dims.
                non_glm_dim = sum(
                    1 for f in self.cfg.node_features if f in _SCALAR_FEATURES
                )
                dim = self.cfg.node_feat_dim - non_glm_dim
                if col_start is None:
                    col_start = offset
                col_end = offset + dim
            else:
                dim = 1  # default assumption for unknown scalar features
            offset += dim

        if col_start is not None and col_end is not None:
            return (col_start, col_end)
        return None

    # ------------------------------------------------------------------
    # Built-in node feature methods
    # ------------------------------------------------------------------

    def _node_feat_degree(self, graph_data: RawGraphData) -> torch.Tensor:
        """Node degree (number of edges per node).

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, 1]``.
        """
        edge_index, _ = self._get_primary_edges(graph_data)
        N = graph_data.num_nodes
        if edge_index is None or edge_index.shape[1] == 0:
            return torch.zeros(N, 1, dtype=torch.float32)
        degree = torch.bincount(edge_index[0], minlength=N).float().unsqueeze(-1)
        return degree

    def _node_feat_strength(self, graph_data: RawGraphData) -> torch.Tensor:
        """Node strength (sum of edge weights per node).

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, 1]``.
        """
        edge_index, edge_attr = self._get_primary_edges(graph_data)
        N = graph_data.num_nodes
        if edge_index is None or edge_attr is None or edge_index.shape[1] == 0:
            return torch.zeros(N, 1, dtype=torch.float32)
        weights = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
        strength = torch.zeros(N, dtype=torch.float32).scatter_add_(0, edge_index[0], weights)
        return strength.unsqueeze(-1)

    def _node_feat_betweenness(self, graph_data: RawGraphData) -> torch.Tensor:
        """Betweenness centrality per node.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, 1]``.
        """
        import networkx as nx
        edge_index, edge_attr = self._get_primary_edges(graph_data)
        N = graph_data.num_nodes
        G = nx.Graph()
        G.add_nodes_from(range(N))
        if edge_index is not None and edge_index.shape[1] > 0:
            edges = edge_index.t().tolist()
            G.add_edges_from(edges)
        centrality = nx.betweenness_centrality(G, normalized=True)
        bc = torch.tensor([centrality[i] for i in range(N)], dtype=torch.float32)
        return bc.unsqueeze(-1)

    def _node_feat_clustering(self, graph_data: RawGraphData) -> torch.Tensor:
        """Clustering coefficient per node.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, 1]``.
        """
        import networkx as nx
        edge_index, _ = self._get_primary_edges(graph_data)
        N = graph_data.num_nodes
        G = nx.Graph()
        G.add_nodes_from(range(N))
        if edge_index is not None and edge_index.shape[1] > 0:
            G.add_edges_from(edge_index.t().tolist())
        clustering = nx.clustering(G)
        cc = torch.tensor([clustering[i] for i in range(N)], dtype=torch.float32)
        return cc.unsqueeze(-1)

    def _node_feat_identity(self, graph_data: RawGraphData) -> torch.Tensor:
        """One-hot identity encoding per node (identity adjacency matrix).

        Each node *i* gets a vector of length ``num_nodes`` that is zero
        everywhere except at position *i*, where it is 1.0.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, num_nodes]``.
        """
        N = graph_data.num_nodes
        return torch.eye(N, dtype=torch.float32)

    def _node_feat_fc_row(self, graph_data: RawGraphData) -> torch.Tensor:
        """FC connectivity-profile: each node's row in the N×N FC matrix.

        Reconstructs the dense N×N functional connectivity matrix from
        ``graph_data.fc_edge_index`` and ``graph_data.fc_edge_attr`` by
        scattering edge weights into the appropriate (src, dst) cells.
        Both directions are filled (symmetric).  If ``fc_edge_index`` is
        ``None`` or empty, returns an all-zero matrix.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, num_nodes]``, dtype float32.
        """
        return self._connectivity_profile(
            graph_data,
            edge_index=graph_data.fc_edge_index,
            edge_attr=graph_data.fc_edge_attr,
        )

    def _node_feat_sc_row(self, graph_data: RawGraphData) -> torch.Tensor:
        """SC connectivity-profile: each node's row in the N×N SC matrix.

        Reconstructs the dense N×N structural connectivity matrix from
        ``graph_data.sc_edge_index`` and ``graph_data.sc_edge_attr`` by
        scattering edge weights into the appropriate (src, dst) cells.
        Both directions are filled (symmetric).  If ``sc_edge_index`` is
        ``None`` or empty, returns an all-zero matrix.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, num_nodes]``, dtype float32.
        """
        return self._connectivity_profile(
            graph_data,
            edge_index=graph_data.sc_edge_index,
            edge_attr=graph_data.sc_edge_attr,
        )

    def _node_feat_cycle_counts(self, graph_data: RawGraphData) -> torch.Tensor:
        """ID-GNN-Fast cycle counts: log1p of [A^l]_vv for l = 2..L.

        Computed on the binarized adjacency. ``L = cfg.cycle_max_length``.
        Returns ``[N, L-1]``. l=1 is omitted (always 0 without self-loops).

        Raises
        ------
        ValueError
            If the binarized graph is fully dense (counts identical -> useless).
        """
        A = self._binary_adjacency(graph_data)
        N = graph_data.num_nodes
        self._check_sparse(A, N, "cycle_counts")
        L = self.cfg.cycle_max_length
        cols: List[torch.Tensor] = []
        A_power = A.clone()           # A^1
        for _ in range(2, L + 1):
            A_power = A_power @ A      # A^l
            cols.append(torch.diagonal(A_power))
        counts = torch.stack(cols, dim=-1)  # [N, L-1]
        return torch.log1p(counts)

    # ------------------------------------------------------------------
    # GLM map node feature methods
    # ------------------------------------------------------------------

    def _node_feat_glm_scalar(self, graph_data: RawGraphData) -> torch.Tensor:
        """Per-node GLM activation value (scalar per contrast).

        For each configured contrast, every node receives its own parcel-level
        GLM value as a single scalar feature.  When multiple contrasts are
        enabled they are concatenated along the feature dimension.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, num_contrasts]``.

        Raises
        ------
        ValueError
            If ``glm_contrasts`` is empty or GLM maps are missing for the
            subject.
        """
        return self._glm_to_tensor(graph_data, mode="scalar")

    def _node_feat_glm_diagonal(self, graph_data: RawGraphData) -> torch.Tensor:
        """Diagonal GLM embedding (one-hot-like with GLM values).

        Each node *i* gets a vector of length ``num_nodes`` that is zero
        everywhere except at position *i*, where it holds the GLM activation
        value for that parcel.  When multiple contrasts are enabled the
        per-contrast diagonal matrices are concatenated along the feature
        dimension.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, num_nodes * num_contrasts]``.

        Raises
        ------
        ValueError
            If ``glm_contrasts`` is empty or GLM maps are missing for the
            subject.
        """
        return self._glm_to_tensor(graph_data, mode="diagonal")

    # ------------------------------------------------------------------
    # Built-in edge feature methods
    # ------------------------------------------------------------------

    def _edge_feat_weight(self, graph_data: RawGraphData) -> torch.Tensor:
        """Raw edge weight.

        Returns
        -------
        torch.Tensor
            Shape ``[num_edges, 1]``.
        """
        _, edge_attr = self._get_primary_edges(graph_data)
        if edge_attr is None:
            return torch.zeros(0, 1, dtype=torch.float32)
        if edge_attr.dim() == 1:
            return edge_attr.float().unsqueeze(-1)
        return edge_attr.float()

    def _edge_feat_zscore(self, graph_data: RawGraphData) -> torch.Tensor:
        """Z-scored edge weight.

        Returns
        -------
        torch.Tensor
            Shape ``[num_edges, 1]``.
        """
        _, edge_attr = self._get_primary_edges(graph_data)
        if edge_attr is None:
            return torch.zeros(0, 1, dtype=torch.float32)
        weights = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
        mean = weights.mean()
        std = weights.std()
        if std == 0:
            std = torch.tensor(1.0)
        return ((weights - mean) / std).float().unsqueeze(-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _glm_to_tensor(
        self,
        graph_data: RawGraphData,
        mode: Literal["scalar", "diagonal"],
    ) -> torch.Tensor:
        """Convert per-subject GLM maps into a node feature tensor.

        Parameters
        ----------
        graph_data : RawGraphData
            Must have ``glm_maps`` populated.
        mode : {"scalar", "diagonal"}
            ``"scalar"``   — each node gets its own GLM value  → ``[N, C]``
            ``"diagonal"`` — one-hot-like with GLM on diagonal → ``[N, N*C]``

        Returns
        -------
        torch.Tensor

        Raises
        ------
        ValueError
            If no GLM contrasts are configured or maps are missing.
        """
        contrasts = self.cfg.glm_contrasts
        if not contrasts:
            raise ValueError(
                "GLM feature requested but no contrasts configured "
                "(set features.glm_contrasts in YAML)."
            )
        if graph_data.glm_maps is None:
            raise ValueError(
                f"GLM maps not loaded for subject '{graph_data.subject_id}'. "
                "Ensure the GLM directory exists and contrasts are correct."
            )

        N = graph_data.num_nodes
        parts: List[torch.Tensor] = []
        for contrast in contrasts:
            if contrast not in graph_data.glm_maps:
                raise ValueError(
                    f"Contrast '{contrast}' missing from GLM maps "
                    f"for subject '{graph_data.subject_id}'."
                )
            vals = torch.tensor(
                graph_data.glm_maps[contrast], dtype=torch.float32
            )  # shape [N]

            if mode == "scalar":
                parts.append(vals.unsqueeze(-1))  # [N, 1]
            else:  # diagonal
                diag = torch.diag(vals)  # [N, N]
                parts.append(diag)

        return torch.cat(parts, dim=-1)

    def _binary_adjacency(self, graph_data: RawGraphData) -> torch.Tensor:
        """Dense 0/1 symmetric adjacency (no self-loops) from primary edges.

        Used **only** by topology-derived features (laplacian_pe, cycle_counts);
        the graph's weighted edge_attr is untouched everywhere else.
        """
        edge_index, _ = self._get_primary_edges(graph_data)
        N = graph_data.num_nodes
        A = torch.zeros(N, N, dtype=torch.float32)
        if edge_index is None or edge_index.shape[1] == 0:
            return A
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0
        A[dst, src] = 1.0
        A.fill_diagonal_(0.0)
        return A

    def _check_sparse(self, A: torch.Tensor, N: int, feature: str) -> None:
        """Raise if the binarized adjacency is fully dense (constant feature)."""
        if N <= 1:
            return
        off_diag_frac = A.sum() / float(N * (N - 1))
        if torch.isclose(off_diag_frac, torch.tensor(1.0)):
            raise ValueError(
                f"'{feature}' requires a thresholded (sparse) graph: the "
                f"binarized adjacency is fully dense, so the feature is "
                f"constant/uninformative. Set dataset.edge_threshold_mode."
            )

    def _connectivity_profile(
        self,
        graph_data: RawGraphData,
        edge_index: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Scatter edge weights into a dense [N, N] connectivity matrix.

        Parameters
        ----------
        graph_data : RawGraphData
            Used only for ``num_nodes``.
        edge_index : torch.Tensor | None
            Shape ``[2, E]``, dtype int64.
        edge_attr : torch.Tensor | None
            Shape ``[E]`` or ``[E, 1]``, dtype float32.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, num_nodes]``, dtype float32.
        """
        N = graph_data.num_nodes
        matrix = torch.zeros(N, N, dtype=torch.float32)
        if edge_index is None or edge_index.shape[1] == 0:
            return matrix

        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        weights = edge_attr
        if weights is not None:
            if weights.dim() > 1:
                weights = weights[:, 0]  # use first column if 2D
            weights = weights.float()
        else:
            weights = torch.ones(edge_index.shape[1], dtype=torch.float32)

        # Fill both (src→dst) and (dst→src) for symmetry
        matrix[src, dst] = weights
        matrix[dst, src] = weights
        return matrix

    def _get_primary_edges(self, graph_data: RawGraphData):
        """Return (edge_index, edge_attr) preferring SC over FC."""
        if graph_data.sc_edge_index is not None:
            return graph_data.sc_edge_index, graph_data.sc_edge_attr
        return graph_data.fc_edge_index, graph_data.fc_edge_attr

    def _discover_methods(self, prefix: str) -> Dict[str, Callable[..., torch.Tensor]]:
        """Discover all methods matching the given prefix and register them.

        Parameters
        ----------
        prefix : str
            Method name prefix (e.g. ``"_node_feat_"``).

        Returns
        -------
        Dict[str, Callable]
            Mapping from feature name to bound method.
        """
        registry: Dict[str, Callable[..., torch.Tensor]] = {}
        for name in dir(self):
            if name.startswith(prefix):
                feat_name = name[len(prefix):]
                registry[feat_name] = getattr(self, name)
        return registry
