"""Professional model architecture printing utilities.

Provides compact, visually appealing summaries of model architectures
and parameter statistics.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from src.configs.model_config import ModelConfig
from src.models.base_model import BrainGNN


def print_all_configs(
    dataset_cfg,
    feature_cfg,
    label_cfg,
    model_cfg,
    trainer_cfg,
    logging_cfg,
    ft_cfg,
    explainer_cfg=None,
    console: Optional[Console] = None,
) -> None:
    """Print all experiment configuration groups in organized tables.

    Displays each configuration group (dataset, features, labels, model, trainer,
    logging, finetuning, explainer) in separate professional tables for easy reference.

    Parameters
    ----------
    dataset_cfg : DatasetConfig
        Dataset configuration.
    feature_cfg : FeatureConfig
        Feature extraction configuration.
    label_cfg : LabelConfig
        Label configuration.
    model_cfg : ModelConfig
        Model configuration.
    trainer_cfg : TrainerConfig
        Trainer configuration.
    logging_cfg : LoggingConfig
        Logging configuration.
    ft_cfg : FinetuningConfig
        Fine-tuning configuration.
    explainer_cfg : ExplainerConfig, optional
        GNNExplainer configuration.  Displayed only when supplied.
    console : Console, optional
        Rich console for output. If None, creates a new console.
    """
    if console is None:
        console = Console()

    # Collect all configs
    configs = {
        "Dataset": dataset_cfg,
        "Features": feature_cfg,
        "Labels": label_cfg,
        "Model": model_cfg,
        "Trainer": trainer_cfg,
        "Logging": logging_cfg,
        "Fine-tuning": ft_cfg,
    }
    if explainer_cfg is not None:
        configs["GNNExplainer"] = explainer_cfg

    console.print("[bold magenta]═" * 60)
    console.print("[bold magenta]Experiment Configurations[/bold magenta]")
    console.print("[bold magenta]═" * 60)
    console.print()

    for cfg_name, cfg_obj in configs.items():
        _print_config_table(cfg_name, cfg_obj, console)

    console.print()


def _print_config_table(
    title: str, cfg_obj, console: Console
) -> None:
    """Print a single configuration group as a formatted table.

    Parameters
    ----------
    title : str
        Name of the configuration group.
    cfg_obj
        Configuration object (any pydantic model).
    console : Console
        Rich console for output.
    """
    if hasattr(cfg_obj, "model_dump"):
        # Pydantic v2
        cfg_dict = cfg_obj.model_dump(exclude_none=True)
    else:
        # Fallback: convert to dict
        cfg_dict = vars(cfg_obj)

    if not cfg_dict:
        return

    config_table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        box=ROUNDED,
    )
    config_table.add_column("Parameter", style="magenta", width=30)
    config_table.add_column("Value", style="green", width=60)

    for key, value in sorted(cfg_dict.items()):
        config_table.add_row(key, str(value))

    console.print(config_table)
    console.print()


def print_model_summary(
    model: BrainGNN,
    cfg: ModelConfig,
    console: Optional[Console] = None,
) -> None:
    """Print a professional model architecture summary.

    Displays the model name, backbone, configuration parameters, and
    comprehensive parameter statistics in a compact, visually organized format.

    Parameters
    ----------
    model : BrainGNN
        The model instance to summarize.
    cfg : ModelConfig
        Model configuration object.
    console : Console, optional
        Rich console for output. If None, creates a new console.
    """
    if console is None:
        console = Console()

    # Create parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Create header info table
    header_table = Table(title="Model Architecture", show_header=False, box=None)
    header_table.add_column(width=20)
    header_table.add_column(width=50)

    header_table.add_row("Model Name", f"[bold cyan]{type(model).__name__}[/bold cyan]")
    header_table.add_row("Backbone", f"[bold cyan]{cfg.backbone}[/bold cyan]")
    header_table.add_row("Model Type", f"[bold yellow]{cfg.name}[/bold yellow]")

    console.print(header_table)
    console.print()

    # Create configuration table
    if hasattr(cfg, "model_dump"):
        # Pydantic v2
        cfg_dict = cfg.model_dump(exclude_none=True)
    else:
        # Fallback: convert to dict
        cfg_dict = vars(cfg)

    config_table = Table(
        title="Configuration", show_header=True, header_style="bold magenta", box=ROUNDED
    )
    config_table.add_column("Parameter", style="cyan", width=25)
    config_table.add_column("Value", style="green", width=50)

    for key, value in sorted(cfg_dict.items()):
        if key not in ("name", "backbone"):  # Already shown in header
            config_table.add_row(key, str(value))

    if len(cfg_dict) > 2:  # Only print if there are config items
        console.print(config_table)
        console.print()

    # Create parameter statistics table
    stats_table = Table(
        title="Parameter Statistics",
        show_header=True,
        header_style="bold magenta",
        box=ROUNDED,
    )
    stats_table.add_column("Metric", style="cyan", width=25)
    stats_table.add_column("Count", justify="right", style="yellow", width=20)
    stats_table.add_column("Percentage", justify="right", style="green", width=15)

    stats_table.add_row(
        "Total Parameters",
        f"{total_params:,}",
        "100.0%",
    )
    stats_table.add_row(
        "Trainable Parameters",
        f"{trainable_params:,}",
        f"{(trainable_params / total_params * 100):.1f}%"
        if total_params > 0
        else "0.0%",
    )

    if frozen_params > 0:
        stats_table.add_row(
            "Frozen Parameters",
            f"{frozen_params:,}",
            f"{(frozen_params / total_params * 100):.1f}%"
            if total_params > 0
            else "0.0%",
        )

    console.print(stats_table)
    console.print()

    # Print layer breakdown
    _print_layer_breakdown(model, console)


def _print_layer_breakdown(model: nn.Module, console: Console) -> None:
    """Print detailed layer-by-layer breakdown of model parameters.

    Parameters
    ----------
    model : nn.Module
        The model to analyze.
    console : Console
        Rich console for output.
    """
    layer_info = []
    for name, module in model.named_modules():
        if not name:  # Skip top-level model
            continue

        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            layer_info.append((name, type(module).__name__, params))

    if not layer_info:
        return

    layer_table = Table(
        title="Layer Breakdown", show_header=True, header_style="bold magenta", box=ROUNDED
    )
    layer_table.add_column("Layer Name", style="cyan", width=35)
    layer_table.add_column("Type", style="blue", width=20)
    layer_table.add_column("Parameters", justify="right", style="yellow", width=18)

    total_params = sum(p.numel() for p in model.parameters())
    for name, layer_type, params in layer_info:
        percentage = (params / total_params * 100) if total_params > 0 else 0.0
        param_str = f"{params:,} ({percentage:.1f}%)"
        layer_table.add_row(name, layer_type, param_str)

    console.print(layer_table)
    console.print()
