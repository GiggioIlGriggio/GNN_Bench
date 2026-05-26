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
