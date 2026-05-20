"""Modality fusion sub-package."""

from src.models.fusion.base_fusion import ModalityFusion
from src.models.fusion.concat_fusion import ConcatFusion
from src.models.fusion.attention_fusion import CrossAttentionFusion
from src.models.fusion.gated_fusion import GatedFusion

__all__ = [
    "ModalityFusion",
    "ConcatFusion",
    "CrossAttentionFusion",
    "GatedFusion",
]
