# Design: PNC identity → age → PCPT transfer

**Date:** 2026-06-15
**Status:** approved (brainstorm) — pending implementation plan
**Cloned from:** `docs/superpowers/specs/2026-06-13-pnc-age-vwm-transfer-design.md` (the age→VWM transfer batch, EXPERIMENTS.md Batch 2026-06-13). Only the downstream target (VWM → PCPT) and the carrier (identity-only) change; the infrastructure is reused.

---

## 1. Objective & scientific framing

Re-run the age→transfer paradigm with **identity-only** node features: pretrain a GNN backbone on **age** regression, then transfer to **PCPT accuracy** (`PCPT_precision`, the Penn Continuous Performance Test sustained-attention measure).

The reason this is non-obvious — and worth running — is an age-coupling gap measured on the cohort (n≈937):

| Downstream target | corr(age, ·) | A5 floor = corr² |
|---|---|---|
| VWM d-prime (the *prior* batch's target) | +0.239 | **0.057** |
| **PCPT_precision (this batch)** | **+0.533** | **0.284** |

In the VWM batch the identity carrier was the **loser** (full-FT ≈ 0, frozen ≈ the negligible 0.057 floor); the GLM carrier carried the signal. Here we deliberately run that *weak* carrier against a target that is **~5× more age-coupled**. The question therefore shifts:

> Does the strong identity→age backbone (SC→age pooled r²≈0.456) transfer to PCPT — where the age-only floor is *already high* (0.284) — when it did **not** transfer to VWM?

Because the floor is high, a bare transfer number is uninterpretable on its own. The controls (A5 floor, A3 from-scratch, frozen-random) are **load-bearing**, not decorative — they are what separate "the backbone carries PCPT-specific structure" from "PCPT just correlates with age."

## 2. Cohort — TRITASK (n ≈ 937)

One shared allowlist `configs/subject_lists/pnc_tritask_cohort.txt` defined as

```
graph ∩ age ∩ VWM ∩ PCPT_precision ∩ GLM-map
```

**Verified:** this is the VWM-940 cohort **minus 3** = **937** subjects (within the 940, all 940 have age; 937 have non-NaN `PCPT_precision`). It is a strict subset of VWM-940, so:

- The GLM-map requirement is automatically satisfied (the 940 already required it).
- The PCPT arms are **same-subject comparable** to the prior VWM arms (B1/B2 identity, B3/B4 GLM) and reuse one identical subject set — comparability, not extra power, is the point.

Generate **on the cluster** (`slurm/make_cohort.sh`) so the list contains cluster-loadable subjects, and confirm the exact count there (expected 937; local-vs-cluster data parity assumed but to be verified at gen time).

> **Verified data facts (correcting the handoff):** `PCPT_precision` **is** present in the main `Tabular_data/PNC_ALL_SCORES.csv` (col 996), and `PNC_ALL_SCORES_PCPT.csv` is **byte-identical** to it. So the accuracy target needs no special data file, and the `pnc_PCPT_IES.yaml` config (which points at the main CSV) is **not** broken. The measure choice is therefore a purely scientific call, not a data-availability one.

## 3. The five arms

| Arm | Config | Target | Stratify | HPO sweeper | Purpose |
|---|---|---|---|---|---|
| **B-fullFT** | `transfer=from_age_ft` + source ckpt | `PCPT_precision` | `PCPT_precision` | `transfer_finetune` (lr-fixed) | headline transfer |
| **B-frozen** | `transfer=from_age_frozen` + source ckpt | `PCPT_precision` | `PCPT_precision` | `transfer_finetune` (lr-fixed) | does the SC→age rep *linearly* carry PCPT |
| **A3 from-scratch** | `transfer=none` | `PCPT_precision` | `PCPT_precision` | pinned-arch, source-style space | transfer-vs-scratch contrast |
| **frozen-random** | `transfer=frozen_random` *(new)* | `PCPT_precision` | `PCPT_precision` | `transfer_finetune` (lr-fixed) | age-specificity: is it the *training* or any random graph projection |
| **A5 floor** | closed-form `corr(age, PCPT_precision)²` | — | — | — | the age-only bar (= 0.284 on the 937) |

Single **identity** carrier ⇒ **one** source run and **two** transfer arms (full-FT + frozen), not four.

### Fold alignment (the one hard invariant)

`SourceBackboneProvider.assert_aligned` fails closed unless the consuming run's outer split is byte-identical to the source run's. So **every fold-based arm shares**: cohort=937, same seed, `stratify on PCPT_precision`, paper preset **10×5×20**. A new source run is mandatory:

- **New source required.** The existing `src_age_id-361349` stratified on VWM over 940 → a different partition. Run a fresh identity→age source on the **937** cohort with `stratify_target=PCPT_precision` (so the transfer arms' default stratification aligns), `source_age_pinned` sweeper. Target=`age_at_cnb`, stratify=`PCPT_precision`.

The VWM identity arms (B1/B2) are **not** re-run on 937 — they sit on 940 (3 subjects off), and pooled-r² makes them a perfectly good cross-reference without byte-identical folds.

## 4. Infrastructure changes (each TDD'd → PR)

All three serve this one experiment; bundle on `feature/age-pcpt-transfer` (the lr fix is logically separable as a repo-wide bugfix — split into its own PR if preferred).

1. **lr=5e-3 fix (MANDATORY — gates the headline arm).** Per `age_vwm_transfer_dynamics`, identity *full*-FT diverged in 7/50 folds at lr=5e-3 (test r²≈−0.51) on the dense-SC/identity backbone; 1e-4/5e-4/1e-3 were stable. Fix = **drop `0.005`** from `trainer.lr` in `configs/sweeper/transfer_finetune.yaml` (grid → `1e-4, 5e-4, 1e-3`). Guard test asserts the grid no longer contains `0.005`. (Footnote in results: the VWM B1 number was itself run with the buggy grid, so the VWM-vs-PCPT *full-FT* comparison carries that caveat — honest and unavoidable.) Rejected alternatives: grad-clipping / discriminative-LR / warmup — more general but break protocol-identity with the VWM batch and add hyperparameters without guaranteeing 5e-3 converges.

2. **frozen-random control (new code path).** The existing infra freezes the backbone *only* inside `transfer_factory`, which requires a real `SourceBackboneProvider` (checkpoint + `fold_indices.json`). There is no path that freezes a backbone without loading source weights. Add one:
   - New `configs/transfer/frozen_random.yaml`.
   - `scripts/run_experiment.py`: wire a mode that passes `frozen_layers=[backbone]` to the nested runner **without** constructing a `SourceBackboneProvider`.
   - `src/training/nested_cross_validation.py::_wrap_factory_for_fold`: add a freeze-only branch — when there is no provider but `frozen_layers` is non-empty, wrap a factory that builds a fresh model → `reinit_head` → `freeze_layers([backbone])`, loads **no** checkpoint.
   - Use the **same pinned architecture** as the source so the only difference vs B-frozen is *trained-vs-random* backbone weights.
   - Test: backbone params are `requires_grad=False`, head is trainable, and no checkpoint file is read.

3. **cohort generator — multi-column required-non-NaN.** Extend `scripts/make_pnc_vwm_cohort.py` to accept additional "required-non-NaN" columns beyond the single target (here add `PCPT_precision`), emitting `pnc_tritask_cohort.txt`. Small unit test. (A3 from-scratch needs no code — `transfer=none` already returns the plain factory.)

## 5. Execution (cluster, after infra merges)

`cluster-helper`, gpunode02/rad2, `--time=2-00:00:00`, paper preset 10×5×20, `-J=<arm>`, `logging.project=age_pcpt_transfer` (set explicitly — the trainer default drifts; trainer.epochs default is 300).

1. **cohort-gen job** → `pnc_tritask_cohort.txt`, confirm count.
2. **source run** — identity→age on 937, stratify=`PCPT_precision`, `source_age_pinned`. Produces per-(rep,fold) backbones + `fold_indices.json`. Checkpoint dir = `<exp_name>-<jobid>`.
3. **transfer + control arms** (parallel): B-fullFT, B-frozen (both `transfer.source_checkpoint_root=checkpoints/<source-run-name>`), A3 (`transfer=none`), frozen-random (`transfer=frozen_random`).
4. **A5 floor** — closed-form `corr(age, PCPT_precision)²` on the exact 937 cohort (≈0.284); record.
5. **backfill** — `backfill-experiment-results` → EXPERIMENTS.md (mean-of-folds + pooled r² + wandb links).

Submission gotchas (from `age_vwm_transfer_infra`): pass the cohort allowlist as `dataset.subject_list_file` to **all** arms; `trainer.search_space=<path>` (not `sweeper=`); the cohort persists on the cluster because `cluster-submit` never `git clean`s.

## 6. Interpretation / success criteria

- **B-frozen vs A5 (0.284):** `> floor` ⇒ the SC→age representation carries PCPT structure **beyond** raw age (the interesting positive). `≈ floor` ⇒ it merely re-encodes age.
- **B-fullFT vs A3:** `> scratch` ⇒ age-pretraining genuinely helps; `≈ scratch` ⇒ the from-scratch plateau (as seen for VWM full-FT).
- **B-frozen vs frozen-random:** `> random` ⇒ the effect is the *age training*, not any random nonlinear graph projection; `≈ random` ⇒ kills the transfer story.
- **Headline contrast:** does the identity backbone — the *loser* for VWM — clear its floor for the more age-coupled PCPT?

Power note: at n≈937 with ~10 reps the VWM batch found ~half the per-pair comparisons significant; expect similar power here. Report pooled r² and mean-of-folds with the same caveats.

## 7. Out of scope (YAGNI)

- GLM / lappe / any non-identity carrier (single identity carrier by design).
- Re-running VWM arms on the 937 cohort for byte-identical folds (940 is a close enough reference).
- Other PCPT measures (RT, IES) — accuracy chosen; RT/IES could be follow-ups but are not in this batch.
- Backbones other than the one used for the VWM identity arms (keep architecture pinned and comparable).
