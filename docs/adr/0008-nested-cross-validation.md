# ADR-0008: Repeated nested cross-validation with per-outer-fold Optuna HPO

**Status**: Accepted
**Date**: 2026-05
**Supplements**: [ADR-0003](./0003-stratified-kfold-continuous.md), [ADR-0004](./0004-no-data-leakage.md)

## Context

The original evaluation paradigm (one stratified 5-fold CV, one HP set chosen across all folds via `HydraSweep`) lets hyperparameter selection see the test folds. For benchmark publication that is a known source of optimistic bias. Mhanna et al. (MIDL 2026, openreview `0p3r8I3jps`) propose a repeated stratified nested CV — 5 outer folds × 10 repetitions = 50 outer folds, with a stratified 3:1 inner train/val split per outer fold, HPs tuned on the inner split, refit on full outer TrainVal, evaluated on outer Test. They pair it with the Bouckaert–Frank corrected resampled t-test + Benjamini–Hochberg FDR for cross-model comparison. We adopt that protocol as the canonical benchmark evaluation.

## Decision

A single `NestedCrossValidator` in `src/training/nested_cross_validation.py` is the only evaluation entry point. It takes three knobs that subsume both the legacy fast CV and the paper-grade nested CV:

| Knob | Fast mode (legacy) | Complete mode (paper) |
|---|---|---|
| `n_repetitions` | `1` | `10` |
| `n_outer_folds` | `5` | `5` |
| `inner_hpo_trials` | `0` (skip inner HPO, fixed HPs) | `20` (Optuna TPE per outer fold) |

The previous `CrossValidator` is replaced by `NestedCrossValidator` with the legacy knob preset. `HydraSweep` is left untouched in this PR.

**Outer loop.** `n_repetitions × n_outer_folds` outer folds, each producing one `(train_val_idx, test_idx)` split. Per-rep `StratifiedKFold(shuffle=True, random_state=outer_seed[r])`. Bin continuous regression labels with `pd.qcut(..., q=stratify_bins, duplicates="drop")` exactly as ADR-0003 prescribes. Default seed derivation: `outer_seed[r] = trainer.seed + r`; override via optional `outer_seeds: list[int]` in `TrainerConfig`.

**Inner split.** Per outer fold, one stratified 4-way split of the outer TrainVal pool — 3 train folds, 1 val fold (3:1 ratio matching the paper's Figure 1). Seed: `inner_seed[r, k] = outer_seed[r] * 1000 + k`. The inner Val drives Optuna HP selection and Trainer early stopping.

**Inner HPO.** When `inner_hpo_trials > 0`, run that many Optuna trials in-process via `optuna.create_study(sampler=TPESampler(seed=inner_seed))` directly — not via the Hydra-Optuna multirun plugin. The HP search space is read from `configs/sweeper/<model>.yaml` and parsed by a small DSL parser (`choice(...)`, `range(...)`, `interval(...)`, `tag(log, interval(...))`) that maps to `trial.suggest_*` calls. Same Optuna engine, same search spaces as the legacy sweep — different orchestration. Optimisation direction is `minimize` for `val_mae` (default) and `maximize` for `val_r2`; configured via `TrainerConfig.hpo_metric`.

**Refit-on-TrainVal.** After Optuna picks the winning HPs for an outer fold, retrain one model on the full outer TrainVal (inner Train ∪ inner Val) with those HPs, for a fixed epoch count equal to the winning trial's `best_epoch`. No held-out val during refit — using one would amount to not refitting at all. The refit checkpoint is the only model evaluated on outer Test.

> **Correction (2026-05-29, [PR #22](https://github.com/GiggioIlGriggio/GNN_Bench/pull/22), commit `f374aea`).** The two paragraphs above describe the *intended* behaviour, but `Trainer.fit` originally hardcoded best-epoch selection for **minimisation** (`best_val_metric = +inf`; `is_best = monitored < best`). With `hpo_metric: val_r2` (higher-is-better) this never took effect: epoch 0 has the lowest R², so `best_epoch` froze at 0, `refit_epochs = best_epoch + 1` collapsed to **1**, and the refit model predicted the mean → outer-test R² ≈ 0. The Optuna objective (`best_val_metrics["r2"]` read at the frozen epoch 0) was likewise noise, so HP selection was effectively random. `val_mae` runs were unaffected (minimisation matched the hardcoded direction). The fix makes best-epoch selection, early stopping, and the `ReduceLROnPlateau` mode direction-aware (derived from `hpo_metric`), so the code now matches the spec above. **Any `val_r2` nested-CV results produced before `f374aea` are invalid** and must be re-run — see the regression test `tests/test_training.py::TestTrainerBestEpochDirection`.

**Aggregation.** Mean ± std over the `n_repetitions × n_outer_folds` outer-test metrics, reported per regression metric (MAE, RMSE, R², Pearson r). Outer-test predictions (`y_true`, `y_pred`, `outer_fold_id`) are persisted as a CSV so any downstream metric or test can be recomputed.

**Statistical tests.** A new pure-function module `src/training/statistical_tests.py` implements:
- Bouckaert–Frank corrected resampled paired t-test (corrects the naive paired t-test for the dependence between repeated-CV scores).
- Benjamini–Hochberg FDR across all pairwise comparisons.

A `scripts/compare_models.py` CLI consumes ≥2 saved `NestedCVResult` JSONs (one per model run) and emits a pairwise p-value matrix + BH-corrected p-values + a markdown summary. Tests live outside the training run because they are inherently cross-run.

**Wandb layout.** Per-trial training curves under `rep_<r>/fold_<k>/trial_<t>/...`. Outer-test metrics under `rep_<r>/fold_<k>/test/...`. The 50-fold aggregate under `final/test/mean/...` and `final/test/std/...`. New keys are declared in `src/logging/log_schema.py`.

**Checkpoint policy.** One refit checkpoint per outer fold at `checkpoints/rep_<R>/fold_<K>/`. Each fold's directory also contains `best_hparams.json`, `trials.csv` (per-trial HPs + inner-val metric), and `test_predictions.npz`. No per-trial model weights are saved. Roughly 50 checkpoints per (dataset, model) — same order of magnitude as today.

## Considered alternatives

- **Keep one HP set for all folds (today's `HydraSweep`).** Cheaper but leaks the test fold into HP selection; rejected for the canonical benchmark. The workflow stays available as a legacy entry point.
- **Reuse winning trial's model (no refit).** Saves ~50 trainings but inner Val drove both HP selection and early stopping — biased. Rejected.
- **Drive the inner Optuna loop through Hydra-Optuna multirun (Shape α).** Would require launching 50 `--multirun` jobs as a shell-driven outer loop; ugly wandb integration, ~5–10s process-startup overhead × 1000 trials. Rejected in favour of in-process Optuna (Shape β).
- **New `configs/nested_hpo/<model>.yaml` namespace.** Cleaner schema but duplicates search-space definitions with `configs/sweeper/*.yaml`. Rejected to keep one source of truth.

## Consequences

- All benchmark numbers reported in the paper-grade mode are produced under nested CV — no HP-selection leakage.
- A complete-mode run costs ~50× a fast-mode run (50 outer folds × 1 refit) plus the inner-HPO budget (50 × `inner_hpo_trials` trainings). Designed to be resumable per outer fold so SLURM preemption is recoverable, but no SLURM scaffolding ships in this PR.
- `trainer.seed` becomes a *base* seed; the effective outer seeds are derived and persisted to the run output for reproducibility.
- ADR-0003's "random shuffle of train+val for inner val" is superseded — inner splits are now stratified. ADR-0003's outer K-fold logic is unchanged for the `n_repetitions=1` case.
- The fast mode survives as a config preset; nobody loses the ability to run a quick CV.
- `src/sweeps/hydra_sweep.py` is unchanged and continues to work via `--multirun`. Any retirement is a follow-up decision.
