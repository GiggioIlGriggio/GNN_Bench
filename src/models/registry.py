"""Model registry — maps string names to :class:`BrainGNN` subclasses."""

from __future__ import annotations

from typing import Any, Dict, Type

from src.configs.model_config import ModelConfig
from src.models.base_model import BrainGNN

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type[BrainGNN]] = {}


def register_model(name: str):
    """Class decorator that registers a :class:`BrainGNN` subclass.

    Usage::

        @register_model("gcn_brain")
        class GCNBrainGNN(BrainGNN):
            ...

    Parameters
    ----------
    name : str
        Unique key (referenced in YAML as ``model.name``).

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """

    def decorator(cls: Type[BrainGNN]) -> Type[BrainGNN]:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered to {MODEL_REGISTRY[name].__name__}."
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, cfg: ModelConfig, **kwargs: Any) -> BrainGNN:
    """Instantiate a registered model by name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"gcn_brain"``).
    cfg : ModelConfig
        Full model configuration.
    **kwargs
        Additional arguments forwarded to the model constructor
        (e.g. ``node_feat_dim``, ``edge_feat_dim``).

    Returns
    -------
    BrainGNN

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](cfg=cfg, **kwargs)
