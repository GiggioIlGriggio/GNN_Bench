# ADR-0007: Finetuning auto-adopts pretrained feature config

**Status**: Accepted  
**Date**: 2025-04

## Context

When fine-tuning a pretrained model on a new dataset or label target, the feature configuration (node feature set, GLM contrasts) must match what the pretrained model was trained with — the model's first layer input dimension is fixed. Users may forget to pass the correct `features=...` override, producing a silent dimension mismatch that corrupts the pretrained weights.

## Decision

At the start of a finetuning run, the entry point reads `feature_config.json` from the checkpoint folder. Two cases:

1. **User did not pass `features=...` on the CLI**: silently adopt the pretrained `FeatureConfig`. Log a warning with the old and new `node_feat_dim`.
2. **User explicitly passed `features=...`**: validate that `node_feat_dim` matches. If it does not, raise `ValueError` with a clear message explaining the mismatch and suggesting to omit the override.

Detection of explicit override uses `HydraConfig.get().overrides.task`.

## Consequences

- The common case (fine-tuning on a new label without changing features) just works.
- Feature mismatches are caught at startup, not as a cryptic tensor shape error during the first forward pass.
- If `feature_config.json` is absent (e.g. old checkpoint), a warning is logged and the current config is used — the user accepts responsibility.
- This logic lives in the entry point (`scripts/run_experiment.py`), not in `Finetuner`, to keep the finetuner class independent of Hydra.
