"""Model module — BrainGNN base, backbones, fusion strategies, and prediction heads."""

from src.models.base_model import BrainGNN
from src.models.registry import register_model, get_model, MODEL_REGISTRY

# Force registration of concrete models
import src.models.backbones  # noqa: F401
import src.models.unimodal  # noqa: F401
import src.models.mlp_model  # noqa: F401
import src.models.braingnn_model  # noqa: F401

__all__ = [
    "BrainGNN",
    "register_model",
    "get_model",
    "MODEL_REGISTRY",
]
