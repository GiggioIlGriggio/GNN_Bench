"""Pydantic v2 schema for the ``logging`` Hydra config group."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Validated configuration for Weights & Biases experiment tracking."""

    enabled: bool = Field(
        default=True,
        description="Whether to log to wandb.",
    )
    project: str = Field(
        default="brain-gnn-benchmark",
        description="wandb project name.",
    )
    entity: Optional[str] = Field(
        default=None,
        description="wandb entity (team or user). None uses the default entity.",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="wandb run name. None lets wandb auto-generate one.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="wandb tags for the run.",
    )
    log_frequency: int = Field(
        default=1,
        description="Log metrics every N epochs.",
    )
