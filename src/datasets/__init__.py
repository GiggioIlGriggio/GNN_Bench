"""Dataset module — brain graph loading, feature assembly, and label extraction."""

from src.datasets.base_dataset import BrainGraphDataset, RawGraphData
from src.datasets.feature_builder import FeatureBuilder
from src.datasets.label_builder import LabelBuilder
from src.datasets.registry import register_dataset, get_dataset, DATASET_REGISTRY

# Force registration of concrete datasets
import src.datasets.orbit_dataset as _orbit  # noqa: F401
import src.datasets.pnc_dataset as _pnc  # noqa: F401

__all__ = [
    "BrainGraphDataset",
    "RawGraphData",
    "FeatureBuilder",
    "LabelBuilder",
    "register_dataset",
    "get_dataset",
    "DATASET_REGISTRY",
]
