"""GNN backbone sub-package."""

from src.models.backbones.base_backbone import GNNBackbone
from src.models.backbones.gcn import GCNBackbone
from src.models.backbones.gat import GATBackbone
from src.models.backbones.gin import GINBackbone
from src.models.backbones.transformer import GraphTransformerBackbone

__all__ = [
    "GNNBackbone",
    "GCNBackbone",
    "GATBackbone",
    "GINBackbone",
    "GraphTransformerBackbone",
]
