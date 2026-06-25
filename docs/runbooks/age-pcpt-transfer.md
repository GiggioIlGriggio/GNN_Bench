# Runbook: PNC Age→PCPT Transfer (identity carrier, arms + controls)

Leakage-safe nested-CV transfer: an **age-pretrained identity backbone** is loaded
per outer fold into a **PCPT-accuracy** (`PCPT_precision`) prediction run. Single
**identity** carrier ⇒ one source + two transfer arms (full-FT, frozen) + three
controls (A3 from-scratch, frozen-random, A5 floor).

Design spec: `docs/superpowers/specs/2026-06-15-pnc-age-pcpt-transfer-design.md`.
Plan: `docs/superpowers/plans/2026-06-15-pnc-age-pcpt-transfer-infrastructure.md`.
Cloned from the VWM runbook `docs/runbooks/age-vwm-transfer.md` (read it for the
deep "why" on fold alignment, the `SourceBackboneProvider` guard, and `search_space=` vs `sweeper=`).

> **Status:** infra merged to `feature/age-pcpt-transfer`; **cluster smoke is GREEN**
> (gpunode03, fast preset — jobs 361658 cohort, 361997 source, 361998 B-frozen,
> 361994 frozen-random, 361995 A3). This runbook is the **full 10×5×20 run**.

---

## ⚠️ 0. Two things that bite (READ FIRST)

1. **Fold alignment.** The source run and every transfer arm must use the **identical
   subject set and identical stratification** so their nested-CV outer folds match
   byte-for-byte — `SourceBackboneProvider.assert_aligned` hard-fails otherwise
   (`ValueError: fold-index mismatch … refusing to transfer`). Here: cohort =
   `pnc_tritask_cohort.txt` (937), stratify = `PCPT_precision`, everywhere.

2. **Cluster CSV split (the smoke-discovered gotcha).** On the **cluster**, the main
   `Tabular_data/PNC_ALL_SCORES.csv` does **not** carry `PCPT_precision` — only
   `PNC_ALL_SCORES_PCPT.csv` does (they're identical on the laptop, NOT on the
   cluster). So anything that reads `PCPT_precision` (target or `stratify_target`)
   must read it from `PNC_ALL_SCORES_PCPT.csv`:
   - transfer arms: `labels=pnc_PCPT_accuracy` (target `PCPT_precision` ← that file). ✓
   - the age source: **`labels=pnc_age_pcptcsv`** (target `age_at_cnb` but metadata ←
     that file) so `stratify_target=PCPT_precision` resolves. **Do NOT use `pnc_default`
     for the source** — it reads the main CSV and dies with `KeyError: 'PCPT_precision'`.

---

## 1. Key facts

- **Carrier:** identity (`features=identity`) for **every** arm (source, B-fullFT,
  B-frozen, frozen-random, A3). `features=` must equal the source carrier or the
  backbone-load guard fires.
- **Targets / labels:**
  - age source → `labels=pnc_age_pcptcsv` (target `age_at_cnb`, metadata `PNC_ALL_SCORES_PCPT.csv`).
  - PCPT arms → `labels=pnc_PCPT_accuracy` (target `PCPT_precision`).
- **Stratification:** `stratify_target=PCPT_precision` on the source; the PCPT arms
  stratify on their own target (`PCPT_precision`) by default → folds align.
- **HPO search space (nested inner HPO):** pass the sweeper YAML as a file path via
  `trainer.search_space=…` (NOT `sweeper=…` — that only configures Hydra multirun).
  Source & A3 use `source_age_pinned.yaml` (pins backbone arch); B-arms & frozen-random
  use `transfer_finetune.yaml` (**lr=5e-3 already dropped** → grid 1e-4/5e-4/1e-3).
- **Paper preset:** `trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20`.
- **Logging:** `logging.project=age_pcpt_transfer` (set explicitly — the trainer default drifts).
- **Checkpoint dir:** `checkpoints/<experiment_name>-<SLURM_JOB_ID>/` (the `<jobid>` is the Slurm id printed by `cluster-submit`).

## cluster-helper quick ref (the CLI isn't on the non-interactive PATH)

```bash
export PATH="/home/compa/.claude/skills/cluster-helper/bin:$PATH"
cd /home/compa/Documents/working_dir/GNN_Bench-age-pcpt-transfer   # the feature worktree
```
- Submit: `cluster-submit --node <node> slurm/<job>.sh "--export=ALL,RUN_ARGS=<hydra overrides>" -J <name>` (deploys the **current branch**; do this from the worktree).
- Read a finished log (static): `cluster-fetch slurm/logs/<jobid>.out /tmp/x.out` (paths are **relative to project_root**). `cluster-tail <id>` only *follows* (tail -f).
- Queue / availability: `cluster-status`, `cluster-gpus`, `cluster-cpus`.
- The generated cohort `.txt` is git-ignored and **persists** on the cluster across submits (`cluster-submit` does `git reset --hard`, never `git clean`).

---

## 2. Step 0 — generate the TRITASK cohort (once, on the cluster)

```bash
cluster-submit --node node01 slurm/make_cohort.sh \
  "--export=ALL,RUN_ARGS=configs/subject_lists/pnc_tritask_cohort.txt pnc_VWMdprime glm_diagonal pnc_PCPT_accuracy" \
  -J make-pnc-tritask-cohort
```
The 4th positional `pnc_PCPT_accuracy` is the `ALSO_REQUIRE` label → the generator
loads graph∩VWM∩GLM (base) and graph∩PCPT∩GLM, then intersects = `graph ∩ VWM ∩
PCPT ∩ GLM`. **Expected count: 937** (= VWM-940 − 3). Confirm in the job log
(`[cohort] … has 937 subject ids`). Pass `dataset.subject_list_file=configs/subject_lists/pnc_tritask_cohort.txt`
to every arm below.

## 3. Step A — the age source (identity carrier)

```bash
cluster-submit --node <gpunode> slurm/train.sh -J src-age-id-pcpt "--export=ALL,RUN_ARGS=\
dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_tritask_cohort.txt \
model=gcn features=identity labels=pnc_age_pcptcsv stratify_target=PCPT_precision \
trainer.search_space=configs/sweeper/source_age_pinned.yaml \
experiment_name=src_age_id_pcpt \
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
logging.project=age_pcpt_transfer"
```
Record the Slurm job id → the source dir is `checkpoints/src_age_id_pcpt-<jobid>`.
Wait for it to COMPLETE before the B-arms (or submit them with `--dependency=afterok:<jobid>`).

## 4. Step B — the two transfer arms + frozen-random + A3

Substitute `<SJOB>` = the source's Slurm job id. All share cohort + `features=identity`
+ `labels=pnc_PCPT_accuracy` (target `PCPT_precision`, stratify defaults to it).

| arm | `transfer=` | extra | `experiment_name` | `search_space` |
|-----|-------------|-------|-------------------|----------------|
| **B-fullFT** | `from_age_ft` | `transfer.source_checkpoint_root=checkpoints/src_age_id_pcpt-<SJOB>` | `b_fullft_pcpt` | `transfer_finetune.yaml` |
| **B-frozen** | `from_age_frozen` | `transfer.source_checkpoint_root=checkpoints/src_age_id_pcpt-<SJOB>` | `b_frozen_pcpt` | `transfer_finetune.yaml` |
| **frozen-random** | `frozen_random` | *(no source ckpt)* | `frozen_random_pcpt` | `transfer_finetune.yaml` |
| **A3 from-scratch** | `none` | *(no source ckpt)* | `a3_scratch_pcpt` | `source_age_pinned.yaml` |

Template (fill in the row):
```bash
cluster-submit --node <gpunode> slurm/train.sh -J <name> "--export=ALL,RUN_ARGS=\
dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_tritask_cohort.txt \
model=gcn features=identity labels=pnc_PCPT_accuracy \
transfer=<from_age_ft|from_age_frozen|frozen_random|none> \
[transfer.source_checkpoint_root=checkpoints/src_age_id_pcpt-<SJOB>] \
trainer.search_space=configs/sweeper/<transfer_finetune|source_age_pinned>.yaml \
experiment_name=<name> \
trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
logging.project=age_pcpt_transfer"
```
`from_age_ft` = full fine-tune (`frozen_layers=[]`); `from_age_frozen` = head-only
(`frozen_layers=['backbone']`); `frozen_random` = random frozen backbone, no checkpoint
(enabled=false + frozen_layers=['backbone']); `none` = from scratch.

B-fullFT is the arm that diverged at lr=5e-3 in the VWM batch — it's now safe (the
value is dropped from `transfer_finetune.yaml`).

## 5. A5 — the age-only floor (closed-form, no run)

Not a training run. `r² = corr(age, PCPT_precision)²` on the 937 cohort. Computed
locally = **0.284** (`corr = +0.533`). Recompute on the exact cluster cohort if you
want byte-exactness, and record in `EXPERIMENTS.md`. **Every transfer arm must beat
0.284 to show the graph/backbone adds anything over chronological age.**

## 6. Backfill

When jobs finish: `backfill-experiment-results` → `EXPERIMENTS.md` (mean-of-folds +
pooled r² + wandb links). Interpretation (spec §6): B-frozen vs 0.284 (does the SC→age
rep carry PCPT beyond age?); B-fullFT vs A3 (does pretraining beat scratch?); B-frozen
vs frozen-random (is it the age training or any random projection?).

## 7. Troubleshooting

- `KeyError: 'PCPT_precision'` in the source → you used `labels=pnc_default`; use
  `labels=pnc_age_pcptcsv` (§0.2).
- `fold-index mismatch` in a B-arm → source/arm cohort or stratification differ;
  confirm both use `pnc_tritask_cohort.txt` and the source used `stratify_target=PCPT_precision`.
- `Source backbone did not fully load` → the arm's `features=` ≠ source carrier (must be `identity`).
- A B-arm stuck `PENDING (DependencyNeverSatisfied)` → its `afterok` source job failed; fix the source and resubmit.
