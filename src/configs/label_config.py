"""Pydantic v2 schema for the ``labels`` Hydra config group."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class LabelConfig(BaseModel):
    """Validated configuration for label extraction and composite definition.

    Supports:
    - **Single-column target**: set ``target`` to a metadata column name.
    - **Composite target**: set ``composite_columns`` to a list of column
      names, ``composite_method`` to a registered composite name (e.g.
      ``"weighted"``, ``"ies"``, ``"pca"``), and optionally
      ``composite_params`` for method-specific keyword arguments.

    Adding a new composite type requires only:
    1. A class decorated with ``@register_composite("name")`` in
       ``src/datasets/composite_registry.py``.
    2. Setting ``composite_method: name`` in the YAML config.
    """

    target: Optional[str] = Field(
        default="sdi_AGE",
        description="Single metadata column to use as the prediction target.",
    )
    composite_columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Ordered list of metadata columns that form the composite. "
            "Column order matters — each composite_method expects a "
            "specific ordering (e.g. IES expects [RT, accuracy])."
        ),
    )
    composite_method: Optional[str] = Field(
        default=None,
        description=(
            "Registered composite method name (e.g. 'weighted', 'ies', "
            "'pca').  Must match a key in the composite registry."
        ),
    )
    composite_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional keyword arguments forwarded to the composite "
            "constructor.  E.g. ``{weights: [1.0, -1.0]}`` for weighted, "
            "``{n_components: 1}`` for PCA."
        ),
    )
    metadata_file: str = Field(
        default="Tabular_Data/ALL tabular data.csv",
        description="Relative path (within each timepoint dir) to the metadata CSV.",
    )
    id_column: str = Field(
        default="ID",
        description="Name of the subject-ID column in the metadata CSV.",
    )
    glm_id_column: Optional[str] = Field(
        default=None,
        description=(
            "Name of the metadata CSV column containing subject IDs "
            "used for GLM map filenames.  When set, GLM files are loaded "
            "using this column instead of the filename-derived ID.  "
            "E.g. 'SUBJID_MATCHED' in PNC."
        ),
    )
    normalize_components: bool = Field(
        default=False,
        description=(
            "When True and composite is set, each component column is z-scored "
            "using train-fold statistics before compositing. "
            "Must be used via CrossValidator (per-fold fitting) to avoid leakage."
        ),
    )

    @model_validator(mode="after")
    def _check_composite_consistency(self) -> "LabelConfig":
        """Ensure composite_columns and composite_method are set together."""
        has_cols = self.composite_columns is not None
        has_method = self.composite_method is not None

        if has_cols and not has_method:
            raise ValueError(
                "'composite_columns' is set but 'composite_method' is missing. "
                "Specify a registered composite method (e.g. 'ies', 'weighted')."
            )
        if has_method and not has_cols:
            raise ValueError(
                "'composite_method' is set but 'composite_columns' is missing. "
                "Provide the list of metadata columns for the composite."
            )
        if has_cols and self.target is not None:
            raise ValueError(
                "Cannot set both 'target' and 'composite_columns'. "
                "Use 'target' for single-column labels or "
                "'composite_columns' + 'composite_method' for composites."
            )
        if not has_cols and self.target is None:
            raise ValueError(
                "Must specify either 'target' (single column) or "
                "'composite_columns' + 'composite_method' (composite)."
            )
        return self

    @property
    def is_composite(self) -> bool:
        """Return True when a composite label is configured."""
        return self.composite_columns is not None
