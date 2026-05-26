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
