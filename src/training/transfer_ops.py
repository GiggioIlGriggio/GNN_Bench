"""Backbone-transfer primitives shared by the flat and nested fine-tuning paths."""

from __future__ import annotations

import logging
from typing import List

import torch

log = logging.getLogger(__name__)


def reinit_head(model: torch.nn.Module) -> None:
    """Reinitialise every ``nn.Linear`` inside ``model.head`` (Kaiming + zero bias)."""
    if not hasattr(model, "head"):
        return
    for m in model.head.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    log.info("Reinitialised prediction head")


def freeze_layers(model: torch.nn.Module, frozen_prefixes: List[str]) -> int:
    """Set ``requires_grad=False`` on params whose name starts with any prefix.

    Returns the number of parameters frozen.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_prefixes):
            param.requires_grad = False
            frozen += 1
    if frozen:
        log.info("Froze %d parameters matching prefixes: %s", frozen, frozen_prefixes)
    return frozen
