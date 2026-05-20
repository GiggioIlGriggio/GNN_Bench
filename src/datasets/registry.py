"""Dataset registry — maps string names to :class:`BrainGraphDataset` subclasses."""

from __future__ import annotations

from typing import Dict, Type

from src.configs.dataset_config import DatasetConfig
from src.configs.feature_config import FeatureConfig
from src.configs.label_config import LabelConfig
from src.datasets.base_dataset import BrainGraphDataset

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Type[BrainGraphDataset]] = {}


def register_dataset(name: str):
    """Class decorator that registers a :class:`BrainGraphDataset` subclass.

    Usage::

        @register_dataset("orbit")
        class ORBITDataset(BrainGraphDataset):
            ...

    Parameters
    ----------
    name : str
        Unique key for the registry (referenced in YAML as ``dataset.name``).

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """

    def decorator(cls: Type[BrainGraphDataset]) -> Type[BrainGraphDataset]:
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' is already registered to {DATASET_REGISTRY[name].__name__}."
            )
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(
    name: str,
    cfg: DatasetConfig,
    feature_cfg: FeatureConfig,
    label_cfg: LabelConfig,
) -> BrainGraphDataset:
    """Instantiate a registered dataset by name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"orbit"``).
    cfg : DatasetConfig
        Dataset configuration.
    feature_cfg : FeatureConfig
        Feature configuration.
    label_cfg : LabelConfig
        Label configuration.

    Returns
    -------
    BrainGraphDataset

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](cfg=cfg, feature_cfg=feature_cfg, label_cfg=label_cfg)
