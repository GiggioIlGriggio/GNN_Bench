from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class ROIAwareConv(MessagePassing):
    """Per-ROI projection conv (MyNNConv from Li et al. 2021).

    Generates a unique [in_channels x out_channels] weight matrix per ROI
    via a small network n: Linear(num_rois, k) -> ReLU -> Linear(k, in*out).
    """

    def __init__(self, in_channels: int, out_channels: int, num_rois: int, k: int = 8):
        super().__init__(aggr='add')
        self.n = nn.Sequential(
            nn.Linear(num_rois, k, bias=False),
            nn.ReLU(),
            nn.Linear(k, in_channels * out_channels),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rois = num_rois

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        roi_idx: torch.Tensor,
    ) -> torch.Tensor:
        pos = F.one_hot(roi_idx, self.num_rois).float()           # [N, num_rois]
        weight = self.n(pos).view(-1, self.in_channels, self.out_channels)  # [N, in, out]
        x = torch.bmm(x.unsqueeze(1), weight).squeeze(1)           # [N, out]
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor | None) -> torch.Tensor:
        if edge_weight is not None:
            return x_j * edge_weight.view(-1, 1)
        return x_j
