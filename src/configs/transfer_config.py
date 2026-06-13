"""Nested-CV transfer config (age-pretrained backbone → VWM)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TransferConfig(BaseModel):
    """Inject a per-fold source backbone into a nested-CV run.

    Distinct from FinetuningConfig (flat CV). When ``enabled``, the run stays
    on the nested runner but loads source weights per outer fold.
    """

    enabled: bool = Field(default=False)
    source_checkpoint_root: Optional[str] = Field(
        default=None,
        description="Root of a completed source nested run (holds fold_indices.json).",
    )
    checkpoint_variant: Literal["best", "last"] = Field(default="best")
    frozen_layers: List[str] = Field(
        default_factory=list,
        description="Param-name prefixes to freeze. [] = full fine-tune (B1/B3); "
        "['backbone'] = head-only (B2/B4).",
    )

    def validate_runtime(self) -> None:
        if self.enabled and not self.source_checkpoint_root:
            raise ValueError(
                "transfer.enabled=true requires transfer.source_checkpoint_root"
            )
