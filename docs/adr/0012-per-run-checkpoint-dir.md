# ADR-0012: Per-run checkpoint directory keyed on a unique run id

## Status
Accepted — 2026-06-01

## Context
Nested CV writes every artifact — per-fold `test_predictions.npz`,
`best_hparams.json`, fold checkpoints, and the run-level
`nested_cv_result.json` — under `cfg.checkpoint_dir` (default `checkpoints/`,
resolved to a single shared SSD path on the cluster). The root was
`Path(cfg.checkpoint_dir)` verbatim, identical for every run, so **each run
overwrote the previous run's artifacts**. After the 2026-05-29 VWM GLM batch
(7 runs) only the last-finished run's predictions survived; the other six were
clobbered.

That loss is not cosmetic. Per-fold predictions are the input to two things
that cannot be reconstructed from the logged scalar metrics:
- **pooled r²** — a single r² over all out-of-fold predictions (vs the
  mean-of-per-fold r² already logged); and
- the **ADR-0008 corrected resampled paired t-test** (`scripts/compare_models.py`),
  which consumes each run's `nested_cv_result.json` to decide model-vs-model
  significance. With six JSONs destroyed, the batch's significance claims could
  not be computed at all.

wandb is not a fallback: `WandbLogger.log_prediction_scatter` (the only code
that logs a `y_true/y_pred` table) is never called from the nested-CV loop.

The obvious fix — root the dir at `cfg.checkpoint_dir / experiment_name` — is
insufficient: `experiment_name` is a **reusable recipe label**, so re-running
the same experiment would still overwrite its own prior artifacts. The dir must
be keyed on something unique to a single *execution*.

## Decision
1. **Unique run name.** Introduce `src.training.run_identity`:
   `run_name = f"{experiment_name}-{run_uid}"`, where `run_uid` is
   `$SLURM_JOB_ID` on the cluster, falling back to `<utc-timestamp>-<pid>` for
   local runs. The Slurm job id ties the artifact directory back to the Slurm
   log, `cluster-history`, and the `Job ID` recorded in `EXPERIMENTS.md`.
2. **Root artifacts per run.** `NestedCrossValidator.run` roots
   `ckpt_root = Path(cfg.checkpoint_dir) / run_name`. No execution can clobber
   another; the directory self-identifies.
3. **`run_name` flows from the entrypoint.** `scripts/run_experiment.py` passes
   `run_name=build_run_name(cfg.experiment_name)`. This is the value persisted to
   `NestedCVResult.run_name` and used as the display label in
   `compare_models.py` — where a unique id is an improvement (two re-runs of one
   experiment are now distinguishable).

The wandb run *name* is unchanged: it comes from `LoggingConfig.run_name`
(`WandbLogger`), which is decoupled from `NestedCVResult.run_name`, so the wandb
dashboard label stays the clean `experiment_name`.

## Consequences
- Predictions and `nested_cv_result.json` are now always recoverable, so pooled
  r² and the ADR-0008 corrected comparison can be run for any batch after this
  change.
- **Downstream consumers that read nested-CV outputs by `checkpoint_dir` must now
  point at the per-run subdir** (`<checkpoint_dir>/<run_name>/`). The GNN
  explainer (`trainer_cfg.checkpoint_dir / fold_*`) is the main case; finetuning
  uses its own `epoch_checkpoint_dir` and is unaffected. Hydra sweeps
  (`hydra_sweep.py`) build their own trial dirs and are unaffected.
- `checkpoints/` accumulates one subdir per run instead of being self-cleaning;
  disk grows monotonically until pruned. Acceptable — predictions are small
  relative to the dataset, and retention is the whole point.
