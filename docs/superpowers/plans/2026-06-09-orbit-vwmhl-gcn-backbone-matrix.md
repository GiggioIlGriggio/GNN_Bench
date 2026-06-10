# ORBIT VWM-HL × Backbone Matrix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Cluster steps (Tasks 2, 3, 5) must be run through the `cluster-helper` skill** — it puts `cluster-*` on PATH, picks GPU nodes, and tails/fetches logs.

**Goal:** Run the 28-cell (4 backbones × 7 GLM node-feature presets) VWM-HL experiment matrix on ORBIT — reproducing the 2026-06-04 PNC backbone-generalization batch with BrainGNN→GCN and PNC→ORBIT — then analyse and write it up.

**Architecture:** Pure config-selection + a local batch-submit script + analysis. No training-code change (ADR-0013). Every Hydra config the batch references already exists on `main` (and therefore on the cluster deploy), so nothing new ships to the cluster; the script issues 28 `cluster-submit` calls, one Slurm job per cell → ADR-0012 per-run checkpoint dirs → `backfill-experiment-results`.

**Tech Stack:** Bash + `cluster-helper` (Slurm submission), Hydra+Pydantic config, the repo's nested-CV runner, wandb (`orbitglm`/`teampolpetta`), `scripts/compare_models.py` (ADR-0008 corrected paired t-test).

**Spec:** `docs/superpowers/specs/2026-06-09-orbit-vwmhl-gcn-backbone-matrix-design.md`

---

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `slurm/submit_orbit_vwmhl_matrix.sh` | Encodes the backbone→sweeper map, suffix→preset map, per-cell naming, and the byte-identical protocol; loops `cluster-submit` (one job/cell). `DRY_RUN`/`SMOKE` modes. | Create (Task 1) |
| `EXPERIMENTS.md` | New `## Batch 2026-06-09` section: submission manifest (Task 4) → results tables (Task 5) → headline (Task 6). | Modify |
| `reports/2026-06-09-orbit-vwmhl-gcn-backbone-generalization.md` | Full write-up mirroring the 2026-06-04 report. | Create (Task 6) |

The 28 cells: backbones `{gcn, gat, gin, transformer}` × suffixes `{glmdiag, id-glmscalar, id-glmdiag, scprof-glmscalar, scprof-glmdiag, lappe-glmscalar, lappe-glmdiag}`.

---

## Task 1: Batch-submit script + local dry-run verification

**Files:**
- Create: `slurm/submit_orbit_vwmhl_matrix.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# =============================================================================
# submit_orbit_vwmhl_matrix.sh — 2026-06-09 ORBIT VWM-HL node-feature × backbone
# matrix. 4 backbones (gcn/gat/gin/transformer) × 7 GLM presets = 28 jobs.
# Reproduces the 2026-06-04 PNC batch with BrainGNN->GCN and PNC->ORBIT.
# Byte-identical 10x5 nested-CV protocol (ADR-0013). One Slurm job per cell.
#
#   bash slurm/submit_orbit_vwmhl_matrix.sh          # submit the 28 real jobs
#   DRY_RUN=1 bash slurm/submit_orbit_vwmhl_matrix.sh  # print, don't submit
#   SMOKE=1   bash slurm/submit_orbit_vwmhl_matrix.sh  # 28 tiny smoke jobs
#
# Env knobs: NODE (default gpunode02), TIME (2-00:00:00), PROJECT (orbitglm),
#            CLUSTER_SUBMIT (default "cluster-submit"; set to a full path when
#            cluster-* is not on PATH).
# =============================================================================
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
NODE="${NODE:-gpunode02}"
TIME="${TIME:-2-00:00:00}"
PROJECT="${PROJECT:-orbitglm}"
CLUSTER_SUBMIT="${CLUSTER_SUBMIT:-cluster-submit}"
SCRIPT="slurm/train.sh"

BACKBONES=(gcn gat gin transformer)
declare -A SWEEPER=(
  [gcn]=gcn_embedding_dim
  [gin]=gcn_embedding_dim
  [gat]=gat_embedding_dim
  [transformer]=transformer_embedding_dim
)

SUFFIXES=(glmdiag id-glmscalar id-glmdiag scprof-glmscalar scprof-glmdiag lappe-glmscalar lappe-glmdiag)
declare -A PRESET=(
  [glmdiag]=glm_diagonal
  [id-glmscalar]=identity_glm_scalar
  [id-glmdiag]=identity_glm_diagonal
  [scprof-glmscalar]=scprofile_glm_scalar
  [scprof-glmdiag]=scprofile_glm_diagonal
  [lappe-glmscalar]=laplacian_pe_glm_scalar
  [lappe-glmdiag]=laplacian_pe_glm_diagonal
)

if [[ "$SMOKE" == "1" ]]; then
  PROTO="trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=2 trainer.epochs=2"
  EXTRA="logging.enabled=false"
  NAME_SUFFIX="-smoke"
else
  PROTO="trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.epochs=300"
  EXTRA=""
  NAME_SUFFIX=""
fi

COUNT=0
for b in "${BACKBONES[@]}"; do
  for s in "${SUFFIXES[@]}"; do
    name="${b}-orbit-sc-vwmhl-${s}${NAME_SUFFIX}"
    overrides="experiment_name=${name} features=${PRESET[$s]} dataset=orbit model=${b} labels=orbit_mri_VWM_HL_p features.glm_normalize=true ${PROTO} trainer.search_space=configs/sweeper/${SWEEPER[$b]}.yaml trainer.hpo_metric=val_r2 logging.project=${PROJECT}${EXTRA:+ $EXTRA}"
    export_arg="--export=ALL,RUN_ARGS=${overrides}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'cluster-submit --node %s %s -J %s --time=%s %q\n' "$NODE" "$SCRIPT" "$name" "$TIME" "$export_arg"
    else
      echo "[submit] $name"
      "$CLUSTER_SUBMIT" --node "$NODE" "$SCRIPT" -J "$name" --time="$TIME" "$export_arg"
      sleep 1
    fi
    COUNT=$((COUNT + 1))
  done
done

echo "-----------------------------------------------------------"
echo "${DRY_RUN:+[DRY-RUN] }${SMOKE:+[SMOKE] }cells: ${COUNT}"
```

- [ ] **Step 2: Verify the dry-run emits exactly 28 well-formed cells**

Run:
```bash
cd /home/compa/Documents/working_dir/GNN_Bench-orbit-vwmhl-backbone-matrix
chmod +x slurm/submit_orbit_vwmhl_matrix.sh
DRY_RUN=1 bash slurm/submit_orbit_vwmhl_matrix.sh | grep -c '^cluster-submit'
```
Expected: `28`

- [ ] **Step 3: Verify the per-backbone sweeper map and shared overrides**

Run:
```bash
DRY_RUN=1 bash slurm/submit_orbit_vwmhl_matrix.sh > /tmp/orbit_dry.txt
echo "gcn->base:";          grep -- '-J gcn-orbit-sc-vwmhl-glmdiag '         /tmp/orbit_dry.txt | grep -o 'gcn_embedding_dim'
echo "gin->base:";          grep -- '-J gin-orbit-sc-vwmhl-glmdiag '         /tmp/orbit_dry.txt | grep -o 'gcn_embedding_dim'
echo "gat->gat:";           grep -- '-J gat-orbit-sc-vwmhl-glmdiag '         /tmp/orbit_dry.txt | grep -o 'gat_embedding_dim'
echo "transformer->tx:";    grep -- '-J transformer-orbit-sc-vwmhl-glmdiag ' /tmp/orbit_dry.txt | grep -o 'transformer_embedding_dim'
echo "all cells dataset/label/project/normalize:"
grep -c 'dataset=orbit' /tmp/orbit_dry.txt          # expect 28
grep -c 'labels=orbit_mri_VWM_HL_p' /tmp/orbit_dry.txt   # expect 28
grep -c 'logging.project=orbitglm' /tmp/orbit_dry.txt    # expect 28
grep -c 'features.glm_normalize=true' /tmp/orbit_dry.txt # expect 28
echo "preset wiring (one example each):"
grep -- '-J gcn-orbit-sc-vwmhl-id-glmdiag '   /tmp/orbit_dry.txt | grep -o 'features=identity_glm_diagonal'
grep -- '-J gcn-orbit-sc-vwmhl-lappe-glmscalar ' /tmp/orbit_dry.txt | grep -o 'features=laplacian_pe_glm_scalar'
```
Expected: `gcn_embedding_dim`, `gcn_embedding_dim`, `gat_embedding_dim`, `transformer_embedding_dim`; four `28`s; `features=identity_glm_diagonal`; `features=laplacian_pe_glm_scalar`.

- [ ] **Step 4: Verify SMOKE mode is also 28 cells, with the protocol shrunk and wandb off**

Run:
```bash
DRY_RUN=1 SMOKE=1 bash slurm/submit_orbit_vwmhl_matrix.sh > /tmp/orbit_smoke.txt
grep -c '^cluster-submit' /tmp/orbit_smoke.txt        # expect 28
grep -c 'epochs=2' /tmp/orbit_smoke.txt                # expect 28 (real batch is epochs=300 -> 0)
grep -c 'logging.enabled=false' /tmp/orbit_smoke.txt   # expect 28
grep -c -- '-smoke ' /tmp/orbit_smoke.txt              # expect 28 (job names suffixed)
```
Expected: four `28`s. (Cross-check the real dry-run has none of the shrunk protocol: `grep -c 'epochs=2' /tmp/orbit_dry.txt` → `0`.)

- [ ] **Step 5: Commit**

```bash
cd /home/compa/Documents/working_dir/GNN_Bench-orbit-vwmhl-backbone-matrix
git add slurm/submit_orbit_vwmhl_matrix.sh
git commit -m "add ORBIT VWM-HL backbone-matrix batch-submit script

28-cell (gcn/gat/gin/transformer x 7 GLM presets) launcher reproducing
the 2026-06-04 PNC batch on ORBIT. DRY_RUN/SMOKE modes; matched-HPO
sweeper map (ADR-0013) with GCN in BrainGNN's slot.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Cluster smoke gate (validates every preset/backbone path + deploy freshness)

> Run through the **cluster-helper** skill. The 28 smoke jobs are tiny (1 rep / 2 folds / 2 epochs / N≈95, wandb off) — seconds-to-minutes each. They confirm (a) the cluster deploy is fresh enough to have `transformer.yaml`, the orbit label, and all 7 presets; (b) every untested ORBIT feature-builder path (identity-concat, scprofile, laplacian_pe, glm_scalar) runs; (c) all four backbones instantiate. **Do not launch Task 3 until all 28 smokes are `COMPLETED`.**

- [ ] **Step 1: Check GPU availability and pick a node**

Use the cluster-helper skill: `cluster-gpus`. Pick a free node (the original batch used `gpunode02`). Note it as `<NODE>`.

- [ ] **Step 2: Submit the 28 smoke jobs**

```bash
cd /home/compa/Documents/working_dir/GNN_Bench-orbit-vwmhl-backbone-matrix
SMOKE=1 NODE=<NODE> bash slurm/submit_orbit_vwmhl_matrix.sh 2>&1 | tee /tmp/orbit_smoke_submit.log
```
Expected: 28 `[submit] <name>-smoke` lines, each followed by a `cluster-submit` job-id confirmation, ending `[SMOKE] cells: 28`.

- [ ] **Step 3: Wait for completion and verify all 28 COMPLETED**

Via cluster-helper: `cluster-status` (or `cluster-history`), filter for `*-smoke`. Expected: 28 jobs `COMPLETED 0:0`, zero `FAILED`/`TIMEOUT`/`CANCELLED`.

- [ ] **Step 4: If any smoke FAILED, diagnose before proceeding**

Fetch the failing log: `cluster-fetch slurm/logs/<jobid>.out`, then read it locally. Common causes and fixes:
- `Could not find 'sweeper/transformer_embedding_dim'` or missing orbit label → **cluster deploy is stale**; sync the deploy to `main` HEAD (≥ `9810d73`) and re-smoke.
- A node-dim / GLM-contrast-path error in one preset → a genuine ORBIT preset bug; STOP and raise it as a separate minimal fix (out of this batch's scope per spec §10). Do not launch the real 28.

No commit (operational gate).

---

## Task 3: Launch the real 28-cell batch

> cluster-helper skill. Only after Task 2 is fully green.

- [ ] **Step 1: Submit**

```bash
cd /home/compa/Documents/working_dir/GNN_Bench-orbit-vwmhl-backbone-matrix
NODE=<NODE> bash slurm/submit_orbit_vwmhl_matrix.sh 2>&1 | tee /tmp/orbit_vwmhl_submit.log
```
Expected: 28 `[submit] <name>` lines + job-id confirmations, ending `cells: 28`.

- [ ] **Step 2: Capture the name→jobid manifest**

```bash
grep -E '^\[submit\]|Submitted batch job' /tmp/orbit_vwmhl_submit.log
```
(Or reconcile via cluster-helper `cluster-history` filtered to the 28 `*-orbit-sc-vwmhl-*` names.) Record the 28 job IDs for Task 4.

- [ ] **Step 3: Confirm all 28 are queued/running**

Via cluster-helper `cluster-status`: expect 28 `*-orbit-sc-vwmhl-*` jobs `PENDING`/`RUNNING`. No commit yet.

---

## Task 4: Record the submission manifest in EXPERIMENTS.md

**Files:**
- Modify: `EXPERIMENTS.md` (add `## Batch 2026-06-09` near the top of the batch list)

- [ ] **Step 1: Add the batch section with a 28-row submission manifest**

Mirror the 2026-06-04 manifest structure. Include: a **What** paragraph (reproduces 2026-06-04 with BrainGNN→GCN, PNC→ORBIT; cite the spec + this plan), the **Shared recipe** block (the exact overrides from the script), the **Matched-HPO** note (GCN/GIN→`gcn_embedding_dim`, GAT/Transformer→`*_embedding_dim` + heads), the **N≈95 caveat** (spec §9), and the table:

```markdown
| Job | experiment_name | backbone | sweeper | R² (mean-of-folds) | R² (pooled) |
|---|---|---|---|---|---|
| <id> | `gcn-orbit-sc-vwmhl-glmdiag` | gcn | gcn_embedding_dim | TBD | TBD |
| ... (28 rows) ... |
```

Fill the `<id>` column from Task 3's manifest; leave metrics `TBD`.

- [ ] **Step 2: Commit**

```bash
cd /home/compa/Documents/working_dir/GNN_Bench-orbit-vwmhl-backbone-matrix
git add EXPERIMENTS.md
git commit -m "log ORBIT VWM-HL backbone-matrix submission manifest (Batch 2026-06-09)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Backfill results (after all 28 are COMPLETED — may be a later session)

> Often a separate session days later. Use the **backfill-experiment-results** skill, which pulls both mean-of-folds and pooled r² (+ Pearson/MAE/RMSE) from wandb (`orbitglm`/`teampolpetta`) and per-run `checkpoints/<name>-<jobid>/`.

- [ ] **Step 1: Confirm all 28 finished**

cluster-helper `cluster-history` → 28 `*-orbit-sc-vwmhl-*` jobs `COMPLETED 0:0`.

- [ ] **Step 2: Run the backfill skill** for the 28 experiment names; it fetches metrics + wandb run links and writes them into the `## Batch 2026-06-09` tables (per-backbone, sorted by mean-of-folds r², matching the 2026-06-04 layout).

- [ ] **Step 3: Commit** the filled metrics.

```bash
git add EXPERIMENTS.md
git commit -m "backfill ORBIT VWM-HL backbone-matrix results (Batch 2026-06-09)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Analysis + report

**Files:**
- Create: `reports/2026-06-09-orbit-vwmhl-gcn-backbone-generalization.md`
- Modify: `EXPERIMENTS.md` (headline + report link)

- [ ] **Step 1: Within-backbone corrected significance**

For each backbone, run the ADR-0008 corrected resampled paired t-test over per-outer-fold r² across its 7 cells, BH-adjusting across the within-backbone pairs:
```bash
cd /home/compa/Documents/working_dir/GNN_Bench-orbit-vwmhl-backbone-matrix
python scripts/compare_models.py --help   # confirm the per-fold-r² input form used by the 2026-06-04 batch
```
Produce the diagonal-vs-scalar, carrier-matched table (id / scprof / lappe) per backbone, exactly like the 2026-06-04 `Corrected significance` table.

- [ ] **Step 2: Write the report** mirroring `reports/2026-06-04-vwm-glm-node-features-cross-backbone-generalization.md`: per-backbone tables, the corrected-significance table, and a **headline** answering the within-batch questions (does diagonal ≳ scalar replicate on ORBIT VWM-HL? does the diagonal plateau appear?). Lead with **pooled r² + Pearson r** (spec §9); report mean-of-folds but flag it noisy at N≈95.

- [ ] **Step 3: Finalize the EXPERIMENTS.md batch section** — add the headline bullets + `**Full report:**` link, matching the 2026-06-04 entry's tail.

- [ ] **Step 4: Commit**

```bash
git add reports/2026-06-09-orbit-vwmhl-gcn-backbone-generalization.md EXPERIMENTS.md
git commit -m "report: ORBIT VWM-HL node-features x backbone generalization (Batch 2026-06-09)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Finish the branch

- [ ] Use the **superpowers:finishing-a-development-branch** skill. Default: open a PR `feature/orbit-vwmhl-backbone-matrix` → `main` via `gh`, summarising the batch + findings; squash-merge after review; then `git worktree remove` and branch cleanup per `docs/agents/git-workflow.md`.

---

## Notes for the executor

- **N is small (~95, GLM-bounded).** Expect volatile per-fold r² and pathological single-fold blow-ups in some scalar cells (the 2026-06-04 PNC batch saw ±0.476 / ±0.755 cells even at N=940). That is expected, not a bug — report pooled r² + Pearson as the headline.
- **No code or config changes are required for the runs.** If a smoke surfaces a real bug in an untested ORBIT preset path, that fix is out of scope (spec §10) — stop and flag it.
- **cluster-* on PATH:** in non-interactive shells, `cluster-*` may not be on PATH (see the cluster-setup memory). Run cluster steps through the cluster-helper skill, or set `CLUSTER_SUBMIT=/home/compa/.claude/skills/cluster-helper/bin/cluster-submit`.
