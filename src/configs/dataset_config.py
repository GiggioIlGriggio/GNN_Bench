"""Pydantic v2 schema for the ``dataset`` Hydra config group."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Validated configuration for brain-graph dataset loading.

    Maps 1-to-1 with ``configs/dataset/*.yaml``.
    """

    name: str = Field(
        ...,
        description="Registered dataset name (must match a key in DATASET_REGISTRY).",
    )
    root: str = Field(
        ...,
        description="Filesystem root directory of the dataset.",
    )
    timepoints: List[str] = Field(
        default=["T0"],
        description="Timepoints to load (e.g. ['T0'] or ['T0','T1'] for longitudinal).",
    )
    modality: Literal["sc", "fc", "multimodal"] = Field(
        default="sc",
        description="Connectivity modality: structural, functional, or both.",
    )
    atlas: str = Field(
        default="schaefer400",
        description="Parcellation atlas name (e.g. schaefer400, aicha384, gordon333).",
    )
    sc_type: str = Field(
        default="sift_invnodevol_radius2_count_connectivity",
        description="Structural connectivity variant within the .mat file.",
    )
    fc_task: str = Field(
        default="rest",
        description="Functional task condition (rest, audioWM, visuoWM).",
    )
    edge_threshold_mode: Literal["absolute", "topk_percent", "none"] = Field(
        default="none",
        description="How to threshold dense connectivity into sparse edges.",
    )
    edge_threshold_value: float = Field(
        default=0.0,
        description="Threshold value: absolute cutoff or top-k percentage (0-100).",
    )
    num_workers: int = Field(
        default=4,
        description="Number of DataLoader worker processes.",
    )
    subject_id_pad_width: int = Field(
        default=3,
        description="Zero-padding width for converting integer IDs to sub-XXX format.",
    )
    subject_list_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a text file listing allowed subject IDs (one per line, "
            "e.g. 'sub-0001103037'). When set, only subjects in this list "
            "are loaded; any ID in the file not found on disk raises an error. "
            "If null/omitted, all discoverable subjects are used."
        ),
    )
