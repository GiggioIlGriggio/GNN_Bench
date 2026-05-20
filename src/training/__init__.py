"""Training module — trainer, cross-validation, metrics, and checkpointing."""

from src.training.trainer import Trainer, TrainResult
from src.training.cross_validation import CrossValidator, CVResult
from src.training.label_normalizer import LabelNormalizer
from src.training.metrics import (
    MetricDict,
    AggregatedMetrics,
    compute_metrics,
    aggregate_fold_metrics,
)
from src.training.checkpoint_manager import CheckpointManager, FoldCheckpoint

__all__ = [
    "Trainer",
    "TrainResult",
    "CrossValidator",
    "CVResult",
    "LabelNormalizer",
    "MetricDict",
    "AggregatedMetrics",
    "compute_metrics",
    "aggregate_fold_metrics",
    "CheckpointManager",
    "FoldCheckpoint",
]
