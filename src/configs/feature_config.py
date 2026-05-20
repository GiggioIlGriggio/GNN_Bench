"""Pydantic v2 schema for the ``features`` Hydra config group."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    """Validated configuration for node and edge feature assembly.

    Each name in ``node_features`` / ``edge_features`` must correspond to a
    method ``_node_feat_<name>`` / ``_edge_feat_<name>`` in
    :class:`~src.datasets.feature_builder.FeatureBuilder`.
    """

    node_features: List[str] = Field(
        default=["degree", "strength"],
        description="Enabled node feature method names (see FeatureBuilder).",
    )
    edge_features: List[str] = Field(
        default=["weight"],
        description="Enabled edge feature method names (see FeatureBuilder).",
    )
    node_feat_dim: int = Field(
        default=2,
        description="Expected total node feature dimensionality (sum of all enabled features).",
    )
    edge_feat_dim: int = Field(
        default=1,
        description="Expected total edge feature dimensionality (sum of all enabled features).",
    )

    # ------------------------------------------------------------------
    # GLM map node feature settings
    # ------------------------------------------------------------------
    glm_aggregation: str = Field(
        default="mean",
        description=(
            "Aggregation method used when parcellating the GLM map "
            "(must match a subdirectory under schaefer400_glm/)."
        ),
    )
    glm_map_type: str = Field(
        default="zmap",
        description=(
            "Statistical map type to load "
            "(e.g. effect_size, effect_variance, pvalue, stat, zmap)."
        ),
    )
    glm_contrasts: List[str] = Field(
        default_factory=list,
        description=(
            "List of GLM contrasts to use as node features. "
            "When empty, GLM features are disabled even if 'glm_scalar' "
            "or 'glm_diagonal' appears in node_features. "
            "Example: ['contrast-2back_vs_0back']."
        ),
    )
    glm_normalize: bool = Field(
        default=False,
        description=(
            "If True, GLM node features are z-scored per node across "
            "subjects (fit on training set only to prevent data leakage)."
        ),
    )
