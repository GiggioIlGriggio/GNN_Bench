"""Interface adapters for cross-module incompatibilities.

Typed adapter functions for every known boundary mismatch between modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data

from src.models.base_model import BrainGNN


# ---------------------------------------------------------------------------
# Multimodal data container
# ---------------------------------------------------------------------------

@dataclass
class MultimodalData:
    """Container for paired SC and FC PyG Data objects.

    Used by multimodal :class:`BrainGNN` models that process each modality
    through separate backbones before fusion.

    Attributes
    ----------
    sc_data : Optional[torch_geometric.data.Data]
        Structural connectivity graph.
    fc_data : Optional[torch_geometric.data.Data]
        Functional connectivity graph.
    y : torch.Tensor
        Shared label tensor of shape ``[1]``.
    """

    sc_data: Optional[torch_geometric.data.Data] = None
    fc_data: Optional[torch_geometric.data.Data] = None
    y: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def unimodal_to_multimodal_data(
    data: torch_geometric.data.Data,
    modality: str,
) -> MultimodalData:
    """Wrap a single-modality Data object into the multimodal container.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Unimodal graph (SC or FC).
    modality : str
        ``"sc"`` or ``"fc"`` — determines which slot the data fills.

    Returns
    -------
    MultimodalData

    Raises
    ------
    ValueError
        If ``modality`` is not ``"sc"`` or ``"fc"``.
    """
    raise NotImplementedError(
        "TODO: create MultimodalData with data in the appropriate slot, "
        "other slot as None, y from data.y"
    )


def labels_to_tensor(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """Cast a label numpy array to a float32 tensor on the specified device.

    Parameters
    ----------
    y : np.ndarray
        Labels of shape ``[N]`` or ``[N, 1]``.
    device : torch.device

    Returns
    -------
    torch.Tensor
        Shape ``[N, 1]``, dtype ``float32``.
    """
    raise NotImplementedError(
        "TODO: convert numpy to tensor, reshape to [N, 1], cast to float32, move to device"
    )


def adapt_input_features(
    model: BrainGNN,
    source_feat_dim: int,
    target_feat_dim: int,
) -> BrainGNN:
    """Insert a linear adapter layer when source and target feature dims differ.

    Used by :class:`Finetuner` for cross-dataset transfer where node feature
    dimensions may differ between pretraining and fine-tuning datasets.

    Parameters
    ----------
    model : BrainGNN
        Model to adapt.
    source_feat_dim : int
        Feature dim the model was pretrained with.
    target_feat_dim : int
        Feature dim of the new dataset.

    Returns
    -------
    BrainGNN
        Model with an adapter ``nn.Linear(target_feat_dim, source_feat_dim)``
        prepended to the backbone.
    """
    raise NotImplementedError(
        "TODO: if dims differ, wrap model's first layer input with a Linear adapter"
    )


def load_partial_state_dict(
    model: nn.Module,
    state_dict: dict,
) -> Tuple[List[str], List[str]]:
    """Load matching keys from a state dict, skip mismatched ones.

    Parameters
    ----------
    model : nn.Module
        Target model.
    state_dict : dict
        Source state dict (e.g. from a checkpoint).

    Returns
    -------
    Tuple[List[str], List[str]]
        ``(loaded_keys, skipped_keys)`` for logging.
    """
    model_sd = model.state_dict()
    loaded: List[str] = []
    skipped: List[str] = []

    for key, value in state_dict.items():
        if key in model_sd and model_sd[key].shape == value.shape:
            model_sd[key] = value
            loaded.append(key)
        else:
            skipped.append(key)

    model.load_state_dict(model_sd)
    return loaded, skipped
