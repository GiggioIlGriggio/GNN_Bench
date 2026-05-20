"""Pydantic v2 schema for the ``explainer`` Hydra config group."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class ExplainerConfig(BaseModel):
    """Configuration for post-training GNNExplainer analysis.

    GNNExplainer is run after cross-validation (or sweep / finetuning)
    completes.  For each fold it uses that fold's best-checkpoint model
    and explains only the held-out **test** graphs.  Edge importance
    matrices are averaged across folds and saved to *output_dir*.

    edge_size note
    --------------
    At every GNNExplainer optimisation step the regularisation loss
    term added is::

        edge_size_coeff × num_nodes_in_graph × sum(edge_mask)

    For dense brain-connectivity graphs (high average degree) this
    penalty grows large quickly, producing over-sparse explanations.
    Adjust ``edge_size`` *downward* for dense graphs
    (rule of thumb: ``edge_size ≈ 1 / avg_degree``).
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch.  Set to 'true' via CLI or YAML to activate "
            "post-training GNNExplainer."
        ),
    )

    # --- Optimisation --------------------------------------------------------
    epochs: int = Field(
        default=200,
        description="Number of GNNExplainer optimisation steps per graph.",
    )
    lr: float = Field(
        default=0.01,
        description="Learning rate for the GNNExplainer mask optimiser.",
    )

    # --- Regularisation coefficients -----------------------------------------
    edge_size: Union[float, Literal["auto"]] = Field(
        default="auto",
        description=(
            "Edge-mask size regularisation coefficient (multiplied by "
            "num_nodes at every step).  Lower values for denser graphs. "
            "Set to 'auto' to compute 1 / avg_degree from the dataset "
            "at run time.  Override with a fixed float when you need "
            "reproducible, hand-tuned sparsity (e.g. edge_size=0.02)."
        ),
    )
    edge_ent: float = Field(
        default=1.0,
        description="Edge-mask entropy regularisation coefficient.",
    )
    node_feat_size: float = Field(
        default=1.0,
        description="Node-feature mask size regularisation coefficient.",
    )
    node_feat_ent: float = Field(
        default=0.1,
        description="Node-feature mask entropy regularisation coefficient.",
    )

    # --- Mask types ----------------------------------------------------------
    node_mask_type: Optional[Literal["object", "common_attributes", "attributes"]] = Field(
        default=None,
        description=(
            "Whether to also learn a node-feature mask.  "
            "None = edge masks only (recommended for connectivity data).  "
            "Set to 'attributes' to get per-node-feature importances as well."
        ),
    )

    # --- Output --------------------------------------------------------------
    output_dir: str = Field(
        default="",
        description=(
            "Directory to write explanation outputs.  "
            "Empty string (default) means '<checkpoint_dir>/explanations/'."
        ),
    )
    save_subject_masks: bool = Field(
        default=True,
        description=(
            "If True, save per-subject edge-mask tensors as .pt files "
            "under <output_dir>/fold_<N>/subjects/."
        ),
    )
