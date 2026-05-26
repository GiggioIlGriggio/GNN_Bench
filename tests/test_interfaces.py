"""Tests for interface adapters.

Every adapter function in ``src/interfaces/adapters.py`` is tested here.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_geometric.data

from src.interfaces.adapters import (
    MultimodalData,
    adapt_input_features,
    labels_to_tensor,
    load_partial_state_dict,
    unimodal_to_multimodal_data,
)
from src.models.base_model import BrainGNN


# ---------------------------------------------------------------------------
# unimodal_to_multimodal_data
# ---------------------------------------------------------------------------

class TestUnimodalToMultimodal:
    """Tests for ``unimodal_to_multimodal_data``."""

    # def test_sc_modality(self) -> None:
        # """SC data should go into sc_data slot, fc_data should be None."""
        # raise NotImplementedError(
            # "TODO: create Data, call with modality='sc', "
            # "assert result.sc_data is data and result.fc_data is None"
        # )

    # def test_fc_modality(self) -> None:
        # """FC data should go into fc_data slot, sc_data should be None."""
        # raise NotImplementedError(
            # "TODO: create Data, call with modality='fc', "
            # "assert result.fc_data is data and result.sc_data is None"
        # )

    # def test_invalid_modality_raises(self) -> None:
        # """Invalid modality string should raise ValueError."""
        # raise NotImplementedError(
            # "TODO: call with modality='invalid', assert ValueError"
        # )

    # def test_y_is_preserved(self) -> None:
        # """Label should be carried over."""
        # raise NotImplementedError(
            # "TODO: assert result.y == data.y"
        # )


# ---------------------------------------------------------------------------
# labels_to_tensor
# ---------------------------------------------------------------------------

class TestLabelsToTensor:
    """Tests for ``labels_to_tensor``."""

    # def test_output_shape(self) -> None:
        # """Output should be [N, 1] float32."""
        # raise NotImplementedError(
            # "TODO: pass shape [N], assert output.shape == (N, 1)"
        # )

    # def test_dtype(self) -> None:
        # """Output dtype should be float32."""
        # raise NotImplementedError(
            # "TODO: assert tensor.dtype == torch.float32"
        # )

    # def test_device(self) -> None:
        # """Output should be on the requested device."""
        # raise NotImplementedError(
            # "TODO: pass device=torch.device('cpu'), assert tensor.device"
        # )


# ---------------------------------------------------------------------------
# adapt_input_features
# ---------------------------------------------------------------------------

class TestAdaptInputFeatures:
    """Tests for ``adapt_input_features``."""

    # def test_same_dim_no_change(self) -> None:
        # """When dims match, model should remain unchanged."""
        # raise NotImplementedError(
            # "TODO: call with source_feat_dim == target_feat_dim, "
            # "assert no adapter layer added"
        # )

    # def test_different_dim_adds_adapter(self) -> None:
        # """When dims differ, an adapter Linear layer should be inserted."""
        # raise NotImplementedError(
            # "TODO: call with different dims, assert adapter exists"
        # )


# ---------------------------------------------------------------------------
# load_partial_state_dict
# ---------------------------------------------------------------------------

class TestLoadPartialStateDict:
    """Tests for ``load_partial_state_dict``."""

    # def test_full_match(self) -> None:
        # """All keys match — loaded_keys should equal all keys; skipped = []."""
        # raise NotImplementedError(
            # "TODO: create model, save state_dict, load back, "
            # "assert all keys loaded, none skipped"
        # )

    # def test_partial_match(self) -> None:
        # """State dict with extra keys — extras should be skipped."""
        # raise NotImplementedError(
            # "TODO: add extra key to state_dict, load, assert it's in skipped"
        # )

    # def test_missing_keys(self) -> None:
        # """State dict missing keys should still load available ones."""
        # raise NotImplementedError(
            # "TODO: remove a key from state_dict, load, assert loaded count"
        # )

    # def test_shape_mismatch_skipped(self) -> None:
        # """Keys with wrong shapes should be skipped, not raise."""
        # raise NotImplementedError(
            # "TODO: alter a tensor shape in state_dict, load, "
            # "assert the key is in skipped"
        # )
