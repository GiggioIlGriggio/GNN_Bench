# ADR-0005: Decorator-based dataset and model registries

**Status**: Accepted  
**Date**: 2025-03

## Context

Adding a new dataset or GNN backbone should not require modifying the entry point or any existing file beyond the new implementation. The entry point needs to instantiate implementations by string name (from YAML config).

## Decision

Two decorator-based registries:

- `@register_dataset("name")` in `src/datasets/registry.py` — maps a string key to a `BrainGraphDataset` subclass.
- `@register_model("name")` in `src/models/registry.py` — maps a string key to a `BrainGNN` subclass.

The entry point calls `get_dataset(name=cfg.name, ...)` and `get_model(name=cfg.name, ...)`. The registry raises `KeyError` for unknown names.

Import side effects register the implementations: Python must import the concrete dataset/model module before the registry lookup. The entry point handles this implicitly via the dataset and model `__init__.py` files.

## Consequences

- Adding a new dataset or model requires: (1) create the implementation file, (2) apply the decorator, (3) ensure the module is imported at startup (add to `__init__.py` if needed), (4) add a YAML config file.
- The registry is a plain dict — no DI framework, no metaclass magic.
- Names in the registry must match `DatasetConfig.name` / `ModelConfig.name` exactly. Typos raise `KeyError` at startup with a clear message.
