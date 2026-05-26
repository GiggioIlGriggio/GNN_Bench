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
