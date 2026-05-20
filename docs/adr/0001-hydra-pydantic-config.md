# ADR-0001: Hydra + Pydantic v2 for configuration management

**Status**: Accepted  
**Date**: 2025-03

## Context

The platform needs to run many different experiment configurations (different datasets, models, feature sets, label targets, sweep spaces, finetuning scenarios) without modifying code. CLI overrides must be possible for quick iteration in an HPC/Slurm environment. Configuration also needs to be type-safe so that typos and wrong types are caught before a long training run fails.

## Decision

- **Hydra** (`hydra-core`) manages config composition: each module has its own YAML group under `configs/`. The entry point (`scripts/run_experiment.py`) is decorated with `@hydra.main`.
- **Pydantic v2** (`pydantic`) validates every config group at startup: the raw `DictConfig` is passed to the matching pydantic schema (e.g. `DatasetConfig`, `TrainerConfig`) via `OmegaConf.to_container(..., resolve=True)`.

The two systems are deliberately kept separate: Hydra handles composition and CLI overrides; pydantic handles validation and type annotation. No pydantic model is a Hydra structured config.

## Consequences

- All config fields are type-annotated with validation — invalid values raise a `ValidationError` immediately at startup, not deep in training.
- CLI overrides use Hydra dot notation: `python scripts/run_experiment.py trainer.lr=1e-4 model=gat`.
- Sweep search spaces are defined in YAML (e.g. `configs/sweeper/bayesian.yaml`) and passed to Hydra's Optuna sweeper plugin.
- Adding a new config field requires both: (1) adding the field to the pydantic schema, and (2) adding it to the relevant YAML file(s).
- Hydra writes output and hydra run logs to `outputs/` and `multirun/` — do not delete these during active experiments as they contain sweep state.
