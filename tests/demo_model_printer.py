#!/usr/bin/env python
"""Demo script to show model summary printing in action."""

from pathlib import Path

# Add workspace to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

# Load sample config
cfg_path = Path(__file__).parent.parent / "configs" / "experiment.yaml"
if cfg_path.exists():
    cfg = OmegaConf.load(cfg_path)
    
    # Create a sample model instance
    from src.models.registry import get_model
    from src.configs.model_config import ModelConfig
    from src.configs.feature_config import FeatureConfig
    
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    feature_cfg = FeatureConfig(**OmegaConf.to_container(cfg.features, resolve=True))
    
    model = get_model(
        name=model_cfg.name,
        cfg=model_cfg,
        node_feat_dim=feature_cfg.node_feat_dim,
        edge_feat_dim=feature_cfg.edge_feat_dim,
        num_nodes=200,  # Example node count
    )
    
    # Print the professional summary
    from src.utils.model_printer import print_model_summary
    print_model_summary(model, model_cfg)
else:
    print(f"Config file not found at {cfg_path}")
    print("Please run this from the workspace root: PYTHONPATH=$(pwd) python tests/demo_model_printer.py")
