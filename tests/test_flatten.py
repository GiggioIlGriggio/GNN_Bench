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
