# Cluster-helper onboarding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Onboard the `LLM_codebase` repo onto the `/cluster-helper` skill so a single `cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=..."` command on the laptop submits a real Slurm job on `al5165@155.105.223.17`.

**Architecture:** Replace the bespoke `slurm/` launcher (`sed`-substituted templates) with three flat committed bash scripts (`train.sh`, `sweep.sh`, `finetune.sh`) that read Hydra overrides from a `$RUN_ARGS` environment variable. The variable is delivered through Slurm's `sbatch --export=ALL,RUN_ARGS=<args>` mechanism, because `cluster-submit` forwards extra args to sbatch (not to the script). Keep `run_experiments.sh` as the parameter-array batch driver, but rewire it to call `cluster-submit` per combo. Container build and code transport follow cluster-helper's standard flow (`cluster-push-container` + git push + ssh-driven `git reset --hard`).

**Tech stack:** Bash, Slurm, Singularity, Hydra (Python config), git, gh CLI, yq (YAML parser on laptop), ssh.

**Spec:** `docs/superpowers/specs/2026-05-21-cluster-helper-onboarding-design.md`

**Worktree:** `/home/compa/Documents/working_dir/GNN_Bench-cluster-helper/` on branch `feature/cluster-helper-onboarding`. All in-repo edits below happen there. The main checkout (`LLM_codebase/`) is **not** touched.

---

## File structure (locked in here so later tasks reference exact paths)

### Files to CREATE (in worktree)

| Path | Responsibility |
|---|---|
| `.cluster-helper.yaml` | Project Manifest. Tells every `cluster-*` script where this project lives on the cluster, the container name, and Slurm defaults. |
| `slurm/train.sh` | Slurm job script for standard CV training. Reads `$RUN_ARGS`, runs `singularity exec ... python scripts/run_experiment.py ...`. |
| `slurm/sweep.sh` | Same shape as `train.sh`, but adds `--multirun` to the python invocation. 48-hour wall-time. |
| `slurm/finetune.sh` | Same shape as `train.sh`. 12-hour wall-time. Hydra overrides for finetuning arrive via `$RUN_ARGS` like all others. |
| `slurm/run_experiments.sh` | Batch launcher (rewired). Loops over parameter arrays and calls `cluster-submit slurm/${MODE}.sh "--export=..."` per combination. Replaces the old `run_experiments.sh`. |
| `slurm/logs/.gitkeep` | Keeps `slurm/logs/` in git so it exists when Slurm tries to write `.out`/`.err`. |

### Files to MODIFY

| Path | Change |
|---|---|
| `slurm/README.md` | Rewrite for the new flow. Remove all references to `config.sh`, `submit_job.sh`, sed substitution. |
| `.gitignore` | Add `cluster_outputs/` (where `cluster-fetch` lands files) and `slurm/logs/*` with `!slurm/logs/.gitkeep` exception. |

### Files to DELETE

- `slurm/config.sh`
- `slurm/submit_job.sh`
- `slurm/templates/train.slurm`
- `slurm/templates/sweep.slurm`
- `slurm/templates/finetune.slurm`
- `slurm/templates/` (directory becomes empty)
- The old `slurm/run_experiments.sh` (replaced — not deleted-then-recreated; the new content overwrites the old in one commit)

### Files NOT touched

- `Dockerfile` — already correct (produces the gnn_bench:latest image used to build `gnn_bench.sif`).
- `requirements.txt` — unchanged.
- `src/`, `scripts/run_experiment.py`, `configs/`, `tests/`, `CONTEXT.md`, `docs/adr/` — all unchanged.

### Out-of-repo changes (one-time, by humans / via ssh)

- Laptop `~/.bashrc`: append PATH line. (Task 1.)
- Cluster `al5165@155.105.223.17:~/.bashrc`: append `WANDB_API_KEY` line. (Task 11.)

---

## Tasks

### Task 1: Install cluster-helper on laptop PATH

**Files:** `~/.bashrc` (out-of-repo, on laptop)

- [ ] **Step 1: Check whether the PATH entry is already present**

```bash
grep -F 'cluster-helper/bin' ~/.bashrc || echo "MISSING"
```

Expected: prints `MISSING` (the line is not there yet). If it already prints a matching line, skip to Step 3.

- [ ] **Step 2: Append the export line**

```bash
echo 'export PATH="$HOME/.claude/skills/cluster-helper/bin:$PATH"' >> ~/.bashrc
```

- [ ] **Step 3: Source it in this session and verify the binaries resolve**

```bash
source ~/.bashrc
which cluster-init cluster-submit cluster-tail cluster-status cluster-fetch cluster-push-container cluster-upload-dataset
```

Expected: seven lines, all starting with `/home/compa/.claude/skills/cluster-helper/bin/`.

- [ ] **Step 4: Verify `--help` works on at least one binary**

```bash
cluster-submit --help 2>&1 | head -5
```

Expected: prints `Usage: cluster-submit <job.sh> [extra sbatch args...]` and a few lines of help. (`--help` exits non-zero by design — that's fine, just check the output text.)

- [ ] **Step 5: No commit** (this task only changes `~/.bashrc`, outside the repo).

---

### Task 2: Write `.cluster-helper.yaml`

**Files:**
- Create: `/home/compa/Documents/working_dir/GNN_Bench-cluster-helper/.cluster-helper.yaml`

- [ ] **Step 1: Verify cwd is the worktree**

```bash
cd /home/compa/Documents/working_dir/GNN_Bench-cluster-helper
git branch --show-current
```

Expected: `feature/cluster-helper-onboarding`. (If wrong, stop and re-check — the rest of the plan edits this worktree only.)

- [ ] **Step 2: Write the manifest**

Create `.cluster-helper.yaml` with exactly this content:

```yaml
# Cluster_Helper Project Manifest.
# Single source of truth for where this project lives on the cluster.
# See ~/.claude/skills/cluster-helper/CONTEXT.md for terminology.

project_name: gnn_bench

cluster:
  host: al5165@155.105.223.17
  project_root: /data/bdip_ssd/al5165/gnn_bench
  container: gnn_bench.sif

git:
  remote: origin
  branch: main

slurm:
  partition: rad2
  qos: 16cpu
  account: rad
  log_dir: slurm/logs
  # Defaults documented here for reference; each slurm/*.sh hardcodes its own
  # SBATCH headers, so changing values here does NOT change job behaviour.
  gpus: 1
  cpus: 4
  mem: 32G
  time: "23:59:00"

wandb:
  project: orbitglm
  entity: teampolpetta
```

- [ ] **Step 3: Verify the YAML is well-formed and contains the expected keys**

```bash
yq '.project_name, .cluster.host, .cluster.project_root, .cluster.container' .cluster-helper.yaml
```

Expected, exact output:
```
gnn_bench
al5165@155.105.223.17
/data/bdip_ssd/al5165/gnn_bench
gnn_bench.sif
```

- [ ] **Step 4: Verify cluster-helper can auto-discover this manifest**

```bash
cluster-status 2>&1 | head -5
```

Expected: either prints a `squeue` table (if you have jobs running — unlikely on first run) or `No jobs found` / similar. The point is that it must NOT error with "manifest not found" — `cluster-status` walks up from cwd to find `.cluster-helper.yaml`. If it errors with "manifest not found," the file was written to the wrong place.

- [ ] **Step 5: Commit**

```bash
git add .cluster-helper.yaml
git commit -m "$(cat <<'EOF'
chore(cluster): add cluster-helper Project Manifest

Declares project_name=gnn_bench, deploy target at /data/bdip_ssd/al5165/gnn_bench/,
container=gnn_bench.sif. Brand-new sibling dir so the existing GNNBenchV2/
checkout on the cluster stays untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Write `slurm/train.sh`

**Files:**
- Create: `slurm/train.sh`

- [ ] **Step 1: Verify SBATCH header values against the manifest**

```bash
yq '.slurm | "partition=" + .partition + " qos=" + .qos + " account=" + .account + " mem=" + .mem + " time=" + .time' .cluster-helper.yaml
```

Expected: `partition=rad2 qos=16cpu account=rad mem=32G time=23:59:00`. These values must match the SBATCH lines in Step 2.

- [ ] **Step 2: Write the file**

Create `slurm/train.sh` with exactly this content:

```bash
#!/bin/bash
#SBATCH --job-name=gnn_bench_train
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

echo "[run] SHA=$(git rev-parse HEAD)"
echo "[run] host=$(hostname)"
echo "[run] container=$SIF"
echo "[run] RUN_ARGS=${RUN_ARGS:-<none>}"
nvidia-smi || true

PYTHONPATH="$(pwd)" singularity exec --nv \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    "$SIF" \
    python scripts/run_experiment.py \
        logging.project=orbitglm \
        logging.entity=teampolpetta \
        ${RUN_ARGS:-}
```

- [ ] **Step 3: Make it executable**

```bash
chmod +x slurm/train.sh
```

- [ ] **Step 4: Syntax-check with `bash -n`**

```bash
bash -n slurm/train.sh && echo OK
```

Expected: `OK`. If `bash -n` reports a syntax error, fix it before continuing.

- [ ] **Step 5: Sanity-check that `${RUN_ARGS:-}` is referenced in the python invocation**

```bash
grep -F '${RUN_ARGS:-}' slurm/train.sh
```

Expected: exactly one match — the line near `python scripts/run_experiment.py`. Zero or two matches is a bug.

- [ ] **Step 6: Commit**

```bash
git add slurm/train.sh
git commit -m "$(cat <<'EOF'
feat(slurm): add cluster-helper-native train.sh

Self-contained committed bash script (no sed substitution). Reads Hydra
overrides from \$RUN_ARGS, delivered via 'sbatch --export=ALL,RUN_ARGS=...'.
SBATCH resources match the manifest's slurm: defaults (rad2/16cpu/rad,
1 GPU, 4 CPUs, 32G, 24h).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Write `slurm/sweep.sh`

**Files:**
- Create: `slurm/sweep.sh`

- [ ] **Step 1: Write the file**

Create `slurm/sweep.sh` with exactly this content (differs from `train.sh` only in `--job-name`, `--time=47:59:00`, and the `--multirun` flag on the python invocation):

```bash
#!/bin/bash
#SBATCH --job-name=gnn_bench_sweep
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=47:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

echo "[run] SHA=$(git rev-parse HEAD)"
echo "[run] host=$(hostname)"
echo "[run] container=$SIF"
echo "[run] RUN_ARGS=${RUN_ARGS:-<none>}"
nvidia-smi || true

PYTHONPATH="$(pwd)" singularity exec --nv \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    "$SIF" \
    python scripts/run_experiment.py \
        --multirun \
        logging.project=orbitglm \
        logging.entity=teampolpetta \
        ${RUN_ARGS:-}
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x slurm/sweep.sh
```

- [ ] **Step 3: Syntax-check and confirm the `--multirun` flag is present**

```bash
bash -n slurm/sweep.sh && grep -F -- '--multirun' slurm/sweep.sh && echo OK
```

Expected: `OK` on the last line. The `grep` must find exactly the `--multirun` line.

- [ ] **Step 4: Confirm wall-time is 47:59:00 (not 23:59:00)**

```bash
grep '^#SBATCH --time=' slurm/sweep.sh
```

Expected: `#SBATCH --time=47:59:00`.

- [ ] **Step 5: Commit**

```bash
git add slurm/sweep.sh
git commit -m "$(cat <<'EOF'
feat(slurm): add cluster-helper-native sweep.sh

Same shape as train.sh; adds '--multirun' to the python invocation and
bumps wall-time to 47:59:00 for Optuna sweeps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Write `slurm/finetune.sh`

**Files:**
- Create: `slurm/finetune.sh`

- [ ] **Step 1: Write the file**

Create `slurm/finetune.sh` with exactly this content (differs from `train.sh` only in `--job-name` and `--time=11:59:00`):

```bash
#!/bin/bash
#SBATCH --job-name=gnn_bench_finetune
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

SIF="$(pwd)/gnn_bench.sif"

echo "[run] SHA=$(git rev-parse HEAD)"
echo "[run] host=$(hostname)"
echo "[run] container=$SIF"
echo "[run] RUN_ARGS=${RUN_ARGS:-<none>}"
nvidia-smi || true

PYTHONPATH="$(pwd)" singularity exec --nv \
    --bind "$(pwd):$(pwd)" --pwd "$(pwd)" \
    "$SIF" \
    python scripts/run_experiment.py \
        logging.project=orbitglm \
        logging.entity=teampolpetta \
        ${RUN_ARGS:-}
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x slurm/finetune.sh
```

- [ ] **Step 3: Syntax-check + confirm wall-time**

```bash
bash -n slurm/finetune.sh && grep '^#SBATCH --time=' slurm/finetune.sh
```

Expected: `#SBATCH --time=11:59:00`.

- [ ] **Step 4: Commit**

```bash
git add slurm/finetune.sh
git commit -m "$(cat <<'EOF'
feat(slurm): add cluster-helper-native finetune.sh

Same shape as train.sh. 12-hour wall-time. Finetuning-specific Hydra
overrides (finetuning=..., finetuning.checkpoint_path=...) arrive via
\$RUN_ARGS like every other override.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Write the rewired `slurm/run_experiments.sh`

**Files:**
- Modify: `slurm/run_experiments.sh` (overwrites the existing one)

- [ ] **Step 1: Confirm the old file is what we expect**

```bash
head -3 slurm/run_experiments.sh
```

Expected: starts with a shebang and a comment header (the bespoke launcher). If the file is missing or radically different, stop and investigate.

- [ ] **Step 2: Overwrite with the new content**

Replace `slurm/run_experiments.sh` with exactly this content (no template substitution; uses `cluster-submit` directly):

```bash
#!/bin/bash
# =============================================================================
# run_experiments.sh — Batch launcher for cluster-helper-driven experiments.
#
# Edit the parameter arrays below, then run:
#   bash slurm/run_experiments.sh
# Or preview without submitting:
#   DRY_RUN=1 bash slurm/run_experiments.sh
#
# Internally, each combination becomes:
#   cluster-submit slurm/<MODE>.sh "--export=ALL,RUN_ARGS=dataset=... model=... ..."
# Hydra overrides arrive in the job script via the $RUN_ARGS env var.
# =============================================================================
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

# ===========================================================================
# ▶ CONFIGURE YOUR EXPERIMENT GRID HERE
# ===========================================================================

MODE="train"                              # train | sweep | finetune

DATASETS=("orbit")                        # orbit | pnc
MODELS=("gcn" "gat")                      # gcn | gat | gin | mlp
FEATURES=("default")                      # default | glm_scalar | glm_diagonal | identity
LABELS=("default")                        # default | pnc_default | ies_immar

# Mode-specific extras
FINETUNING_CFG="default"                  # for MODE=train (usually "default" = disabled) or MODE=finetune
SWEEPER_CFG="bayesian"                    # for MODE=sweep
OBJECTIVE_METRIC="r2"                     # for MODE=sweep
CHECKPOINT_PATH="checkpoints"             # for MODE=finetune

EXTRA_OVERRIDES=""                        # space-separated, e.g. "trainer.lr=0.001 trainer.epochs=50"

# ===========================================================================

SCRIPT="slurm/${MODE}.sh"
[[ -f "$SCRIPT" ]] || { echo "ERROR: $SCRIPT does not exist (MODE=$MODE)" >&2; exit 1; }

COUNT=0
echo "Mode: ${MODE}    DRY_RUN=${DRY_RUN}"
echo "-----------------------------------------------------------"

for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for f in "${FEATURES[@]}"; do
      for l in "${LABELS[@]}"; do

        # Compose Hydra overrides (space-separated; one --export arg).
        OVERRIDES="dataset=${d} model=${m} features=${f} labels=${l}"
        case "${MODE}" in
          train)
            OVERRIDES="${OVERRIDES} finetuning=${FINETUNING_CFG}"
            ;;
          sweep)
            OVERRIDES="${OVERRIDES} sweeper=${SWEEPER_CFG} objective_metric=${OBJECTIVE_METRIC}"
            ;;
          finetune)
            OVERRIDES="${OVERRIDES} finetuning=${FINETUNING_CFG} finetuning.checkpoint_path=${CHECKPOINT_PATH}"
            ;;
          *)
            echo "ERROR: Unknown MODE '${MODE}' (expected train|sweep|finetune)" >&2
            exit 1
            ;;
        esac
        if [[ -n "${EXTRA_OVERRIDES}" ]]; then
          OVERRIDES="${OVERRIDES} ${EXTRA_OVERRIDES}"
        fi

        EXPORT_ARG="--export=ALL,RUN_ARGS=${OVERRIDES}"

        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '[DRY-RUN] cluster-submit %s %q\n' "$SCRIPT" "$EXPORT_ARG"
        else
          echo "[submit] $SCRIPT  $EXPORT_ARG"
          cluster-submit "$SCRIPT" "$EXPORT_ARG"
          sleep 1
        fi

        COUNT=$((COUNT + 1))
      done
    done
  done
done

echo "-----------------------------------------------------------"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete — ${COUNT} job(s) would be submitted."
else
  echo "Done — ${COUNT} job(s) submitted."
fi
```

- [ ] **Step 3: Make it executable**

```bash
chmod +x slurm/run_experiments.sh
```

- [ ] **Step 4: Syntax-check**

```bash
bash -n slurm/run_experiments.sh && echo OK
```

Expected: `OK`.

- [ ] **Step 5: DRY_RUN smoke test (no submission, just print what would be submitted)**

```bash
DRY_RUN=1 bash slurm/run_experiments.sh
```

Expected output (the combinations may differ if the defaults in the file are changed):
```
Mode: train    DRY_RUN=1
-----------------------------------------------------------
[DRY-RUN] cluster-submit slurm/train.sh '--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default finetuning=default'
[DRY-RUN] cluster-submit slurm/train.sh '--export=ALL,RUN_ARGS=dataset=orbit model=gat features=default labels=default finetuning=default'
-----------------------------------------------------------
Dry run complete — 2 job(s) would be submitted.
```

If the count or strings differ, stop and inspect the loop body.

- [ ] **Step 6: Verify error path: unknown MODE fails fast**

```bash
sed 's/^MODE="train"/MODE="bogus"/' slurm/run_experiments.sh > /tmp/_bogus.sh
DRY_RUN=1 bash /tmp/_bogus.sh; echo "exit=$?"
rm /tmp/_bogus.sh
```

Expected: prints `ERROR: $SCRIPT does not exist (MODE=bogus)` or `ERROR: Unknown MODE 'bogus'`, then `exit=1`. (The exact wording depends on which check fires first — both are acceptable.)

- [ ] **Step 7: Commit**

```bash
git add slurm/run_experiments.sh
git commit -m "$(cat <<'EOF'
refactor(slurm): rewire run_experiments.sh around cluster-submit

Same parameter-array UX (DATASETS/MODELS/FEATURES/LABELS/MODE, DRY_RUN=1
preview). Each combo becomes a single cluster-submit call with the Hydra
overrides packed into '--export=ALL,RUN_ARGS=...'. The old sed-substitution
templates and submit_job.sh path is removed in a follow-up commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Create `slurm/logs/.gitkeep` and update `.gitignore`

**Files:**
- Create: `slurm/logs/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create `slurm/logs/` and `.gitkeep`**

```bash
mkdir -p slurm/logs
touch slurm/logs/.gitkeep
```

- [ ] **Step 2: Inspect the current `.gitignore` to confirm what's already there**

```bash
grep -nE '^(slurm/logs|cluster_outputs|\*\.sif)' .gitignore
```

Expected: shows `*.sif` exists (line ~3). Likely shows no entries for `slurm/logs` or `cluster_outputs`. If `slurm/logs` is already covered (`slurm/logs/` line), skip the add for it in Step 3.

- [ ] **Step 3: Append the cluster-helper entries**

Add to `.gitignore` (append at the end, under a fresh section header):

```
# --- cluster-helper artifacts ---
cluster_outputs/
slurm/logs/*
!slurm/logs/.gitkeep
```

- [ ] **Step 4: Verify the `.gitkeep` is tracked but a log file would not be**

```bash
touch slurm/logs/9999.out slurm/logs/9999.err
git check-ignore -v slurm/logs/9999.out slurm/logs/.gitkeep cluster_outputs/foo 2>&1
```

Expected:
- `slurm/logs/9999.out` → matches `.gitignore` line for `slurm/logs/*`.
- `slurm/logs/.gitkeep` → either prints nothing (not ignored) or returns exit code 1.
- `cluster_outputs/foo` → matches `.gitignore` line for `cluster_outputs/`.

Cleanup the dummy files:

```bash
rm slurm/logs/9999.out slurm/logs/9999.err
```

- [ ] **Step 5: Commit**

```bash
git add slurm/logs/.gitkeep .gitignore
git commit -m "$(cat <<'EOF'
chore(gitignore): track slurm/logs/.gitkeep; ignore cluster-helper artifacts

Ensures slurm/logs/ exists in the cluster checkout (Slurm needs the dir to
write %j.out/%j.err). Ignores cluster_outputs/ where cluster-fetch drops
files locally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Delete the bespoke launcher files

**Files:**
- Delete: `slurm/config.sh`, `slurm/submit_job.sh`, `slurm/templates/train.slurm`, `slurm/templates/sweep.slurm`, `slurm/templates/finetune.slurm`, `slurm/templates/`

- [ ] **Step 1: Confirm what's there before deleting**

```bash
ls slurm/
ls slurm/templates/
```

Expected:
- `slurm/` contains `config.sh`, `submit_job.sh`, `run_experiments.sh` (already replaced), `train.sh`, `sweep.sh`, `finetune.sh`, `README.md`, `logs/`, `templates/`.
- `slurm/templates/` contains `train.slurm`, `sweep.slurm`, `finetune.slurm`.

If anything unexpected is present, stop and surface it — this is the destructive step.

- [ ] **Step 2: Verify no other file in the repo references these soon-to-be-deleted files**

```bash
grep -rn -E 'config\.sh|submit_job\.sh|templates/(train|sweep|finetune)\.slurm' \
    --include='*.sh' --include='*.md' --include='*.py' \
    --exclude-dir=.git --exclude-dir=.venv --exclude-dir=docs/superpowers .
```

Expected: matches inside `slurm/README.md` ONLY (which Task 9 will rewrite). Any other match must be investigated before deleting.

- [ ] **Step 3: Delete via `git rm`**

```bash
git rm slurm/config.sh slurm/submit_job.sh \
       slurm/templates/train.slurm slurm/templates/sweep.slurm slurm/templates/finetune.slurm
rmdir slurm/templates
```

- [ ] **Step 4: Verify the directory is gone and nothing unexpected lingers**

```bash
ls slurm/
```

Expected: `train.sh sweep.sh finetune.sh run_experiments.sh README.md logs/` (no `config.sh`, no `submit_job.sh`, no `templates/`).

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(slurm): remove bespoke sed-substitution launcher

Drops config.sh, submit_job.sh, and templates/{train,sweep,finetune}.slurm.
Replaced by cluster-helper-native train.sh / sweep.sh / finetune.sh +
rewired run_experiments.sh in the previous commits. README rewrite follows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Rewrite `slurm/README.md` for the new flow

**Files:**
- Modify: `slurm/README.md`

- [ ] **Step 1: Replace the file contents**

Overwrite `slurm/README.md` with exactly:

````markdown
# SLURM Launcher — `gnn_bench`

This folder contains the Slurm job scripts and batch launcher that the
`/cluster-helper` skill drives. The cluster is `al5165@155.105.223.17`; the
deploy target is `/data/bdip_ssd/al5165/gnn_bench/`.

For the skill's own design and verbs, see
`~/.claude/skills/cluster-helper/README.md` (or the `CONTEXT.md` next to it).
For project-side configuration, see `../.cluster-helper.yaml`.

---

## Folder structure

```
slurm/
├── train.sh             ← standard 5-fold cross-validation job
├── sweep.sh             ← Optuna hyperparameter sweep (--multirun)
├── finetune.sh          ← fine-tuning from checkpoint
├── run_experiments.sh   ← batch launcher (loops over parameter grids)
└── logs/                ← stdout/stderr from Slurm (auto-populated)
```

Each `.sh` is a flat committed bash script. SBATCH headers are hardcoded;
the only runtime input is the `$RUN_ARGS` environment variable, which carries
Hydra overrides.

---

## How Hydra overrides reach the script: `$RUN_ARGS`

`cluster-submit` forwards extra positional args to **sbatch**, not to the
job script. So we cannot pass `dataset=orbit` as a positional arg — sbatch
would reject it. Instead, the slurm script reads `$RUN_ARGS`, and the caller
sets it via Slurm's `--export` mechanism:

```bash
cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default"
```

Inside `slurm/train.sh`, this expands to:

```bash
python scripts/run_experiment.py logging.project=orbitglm logging.entity=teampolpetta \
    dataset=orbit model=gcn features=default labels=default
```

The `ALL,` prefix tells Slurm to also forward the existing environment
(including `WANDB_API_KEY` from `~/.bashrc` on the cluster).

---

## Single submit

```bash
# Standard training
cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default"

# Optuna sweep
cluster-submit slurm/sweep.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default sweeper=bayesian objective_metric=r2"

# Fine-tuning from checkpoint
cluster-submit slurm/finetune.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=identity labels=default finetuning=from_age finetuning.checkpoint_path=checkpoints"
```

After each submit, `cluster-submit` prints `JOB_ID=...` on stdout. Tail the
log with:

```bash
cluster-tail <jobid>
```

---

## Batch experiments

Open `slurm/run_experiments.sh` and edit the arrays at the top:

```bash
MODE="train"                              # train | sweep | finetune
DATASETS=("orbit" "pnc")
MODELS=("gcn" "gat" "gin")
FEATURES=("default" "glm_scalar")
LABELS=("default")
```

Every combination is submitted as an independent Slurm job.

Always dry-run first:

```bash
DRY_RUN=1 bash slurm/run_experiments.sh
```

Then submit for real:

```bash
bash slurm/run_experiments.sh
```

---

## Logs

Slurm writes per-job logs to `slurm/logs/<jobid>.out` and `<jobid>.err` (the
two are gitignored — only `slurm/logs/.gitkeep` is tracked).

```bash
cluster-tail <jobid>           # follow stdout
cluster-tail <jobid> --err     # follow stderr
cluster-status                 # squeue for this user
```

Pull a file back from the cluster to `./cluster_outputs/`:

```bash
cluster-fetch outputs/run42/checkpoint.pt
```

---

## Container

The Singularity image is built locally from `Dockerfile` and rsynced to the
cluster as `${PROJECT_ROOT}/gnn_bench.sif`. Rebuild only when `Dockerfile` or
`requirements.txt` changes:

```bash
cluster-push-container
```

The `.sif` is gitignored — git carries the recipe (`Dockerfile`), not the
image.
````

- [ ] **Step 2: Verify the new content mentions `RUN_ARGS` and does NOT mention the deleted files**

```bash
grep -E 'RUN_ARGS|--export=ALL' slurm/README.md | head -5
grep -E 'config\.sh|submit_job\.sh|sed' slurm/README.md && echo "STILL MENTIONS OLD FILES — FIX" || echo "OK"
```

Expected: the first grep finds several `RUN_ARGS`/`--export=ALL` mentions; the second prints `OK` (no matches for the old files).

- [ ] **Step 3: Commit**

```bash
git add slurm/README.md
git commit -m "$(cat <<'EOF'
docs(slurm): rewrite README for cluster-helper-native flow

Documents \$RUN_ARGS / sbatch --export usage, the three flat slurm scripts,
batch launcher UX, and cluster-* verbs. Removes all references to the
deleted config.sh / submit_job.sh / sed-substituted templates.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Push the feature branch to GitHub

**Files:** none (network operation).

- [ ] **Step 1: Confirm we're on the right branch with a clean working tree**

```bash
git status
git log --oneline main..HEAD
```

Expected: `On branch feature/cluster-helper-onboarding`, working tree clean. `git log` shows the commits from Tasks 2, 3, 4, 5, 6, 7, 8, 9 plus the spec commit (`docs(superpowers): add cluster-helper onboarding design spec`). Total ≈ 9 commits ahead of `main`.

- [ ] **Step 2: Push with upstream tracking**

```bash
git push -u origin feature/cluster-helper-onboarding
```

Expected: prints `Branch 'feature/cluster-helper-onboarding' set up to track 'origin/feature/cluster-helper-onboarding'`. No commits should be rejected.

- [ ] **Step 3: Verify the branch landed on GitHub**

```bash
gh api repos/GiggioIlGriggio/GNN_Bench/branches/feature/cluster-helper-onboarding --jq '.name + " @ " + .commit.sha'
```

Expected: prints `feature/cluster-helper-onboarding @ <40-char SHA>` matching `git rev-parse HEAD`.

- [ ] **Step 4: No commit** (network-only).

---

### Task 11: Set `WANDB_API_KEY` on the cluster

**Files:** `al5165@155.105.223.17:~/.bashrc` (out-of-repo, on cluster).

**Critical: do NOT type the API key in this conversation.** The implementing agent must ask the user to run the command themselves so the key never enters the transcript.

- [ ] **Step 1: Verify the key is not already set on the cluster**

```bash
ssh -o BatchMode=yes al5165@155.105.223.17 'grep -c WANDB_API_KEY ~/.bashrc'
```

Expected: `0`. If `>= 1`, the key is already configured — skip to Step 4.

- [ ] **Step 2: Ask the user to paste the export line directly into their own terminal**

Present the user with this exact command, telling them to substitute their key locally and run it themselves:

```bash
ssh al5165@155.105.223.17 'echo "export WANDB_API_KEY=YOUR_KEY_HERE" >> ~/.bashrc'
```

Do NOT run this command from within the agent. The user runs it locally — that way the key is in their shell history, not the agent's tool-call log.

- [ ] **Step 3: After the user confirms they ran it, verify (without revealing the key)**

```bash
ssh -o BatchMode=yes al5165@155.105.223.17 'grep -c WANDB_API_KEY ~/.bashrc'
```

Expected: `1`. (Not the key itself — just the count.)

- [ ] **Step 4: Verify the key is exported in a non-interactive shell**

```bash
ssh -o BatchMode=yes al5165@155.105.223.17 'source ~/.bashrc && [[ -n "${WANDB_API_KEY:-}" ]] && echo SET || echo MISSING'
```

Expected: `SET`. If `MISSING`, the `~/.bashrc` may not be sourced in non-interactive ssh sessions. In that case, ask the user to move the line into `~/.bash_profile` or `~/.profile` instead — Slurm jobs invoke a login shell so those are sourced.

- [ ] **Step 5: No commit** (out-of-repo).

---

### Task 12: Build and push the Singularity container

**Files:** none in repo; produces `gnn_bench.sif` locally and on the cluster.

- [ ] **Step 1: Verify Docker is running and the user is in the `docker` group**

```bash
docker ps >/dev/null && echo "DOCKER OK" || echo "DOCKER NOT WORKING"
```

Expected: `DOCKER OK`. If `NOT WORKING`, stop and fix Docker before proceeding (likely `sudo systemctl start docker` or membership issue).

- [ ] **Step 2: Verify there's enough free disk for the build (~10 GB transient)**

```bash
df -h . | tail -1
```

Expected: at least ~10 GB free on the partition holding the worktree. The build produces a transient `.tar` in `/tmp/` (auto-cleaned) and a ~3.5 GB `gnn_bench.sif` in the repo root.

- [ ] **Step 3: Run the build + push (long-running; ~10-30 min depending on network)**

```bash
cluster-push-container
```

Expected stderr lines (in order):
- `[cluster] docker build -> gnn_bench:latest`
- `[cluster] docker save -> /tmp/cluster-helper.XXXXXX.tar`
- `[cluster] singularity build -> .../gnn_bench.sif`
- `[cluster] rsync .../gnn_bench.sif -> al5165@...:/data/bdip_ssd/al5165/gnn_bench/`

Expected stdout (parseable):
```
SIF_LOCAL=/home/compa/Documents/working_dir/GNN_Bench-cluster-helper/gnn_bench.sif
SIF_REMOTE=/data/bdip_ssd/al5165/gnn_bench/gnn_bench.sif
```

- [ ] **Step 4: Verify the local `.sif` exists and is gitignored**

```bash
ls -lh gnn_bench.sif
git status --short gnn_bench.sif
git check-ignore gnn_bench.sif
```

Expected: file exists (~3-4 GB), `git status` shows nothing for it (gitignored), `git check-ignore` prints `.gitignore` (or the file path itself, depending on git version) and exits 0.

- [ ] **Step 5: Verify the remote `.sif` exists at the expected path**

```bash
ssh -o BatchMode=yes al5165@155.105.223.17 \
    'ls -lh /data/bdip_ssd/al5165/gnn_bench/gnn_bench.sif'
```

Expected: prints a `-rw-...` line with a ~3-4 GB size. If it errors with "No such file or directory", the rsync failed and Task 12 must be re-run.

- [ ] **Step 6: Verify the container actually runs CUDA on the cluster**

```bash
ssh -o BatchMode=yes al5165@155.105.223.17 \
    'singularity exec /data/bdip_ssd/al5165/gnn_bench/gnn_bench.sif python -c "import torch; print(torch.__version__)"'
```

Expected: prints `2.6.0` (or whatever's in the Dockerfile). If it errors, the container is broken — investigate the build log before proceeding.

- [ ] **Step 7: No commit** (only artifacts, all gitignored).

---

### Task 13: Smoke test — single `cluster-submit`

**Files:** none in repo (consumes `slurm/train.sh` and the manifest).

The point of this task is to validate the **`RUN_ARGS` env-var mechanism end-to-end** before relying on it in the batch launcher. Choose a fast Hydra invocation that exercises CUDA + the data pipeline + wandb.

- [ ] **Step 1: Submit a 2-epoch training run and capture the JOB_ID**

```bash
SUBMIT_OUT=$(cluster-submit slurm/train.sh \
    "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default finetuning=default trainer.epochs=2")
echo "$SUBMIT_OUT"
JOB_ID=$(printf '%s\n' "$SUBMIT_OUT" | grep ^JOB_ID= | cut -d= -f2)
echo "Captured JOB_ID=$JOB_ID"
```

Expected: `$SUBMIT_OUT` contains five `KEY=value` lines on stdout:
```
JOB_ID=<6-digit number>
DEPLOY_SHA=<40-char SHA>
BRANCH=feature/cluster-helper-onboarding
LOG_PATH=/data/bdip_ssd/al5165/gnn_bench/slurm/logs/<JOB_ID>.out
TAIL_CMD=cluster-tail <JOB_ID>
STATUS_CMD=cluster-status
```

Plus `[cluster] ...` progress lines on stderr (not captured into `$SUBMIT_OUT`).

`$JOB_ID` must be a non-empty integer after this step. If empty, the submission failed — re-read the stderr output for the error.

- [ ] **Step 2: Confirm the job is queued or running**

```bash
cluster-status
```

Expected: shows the job ID with state `PD` (pending) or `R` (running). If the job already finished (very fast), `cluster-status` will show empty; check sacct in Step 4 instead.

- [ ] **Step 3: Tail the log until you see CUDA + wandb come up**

```bash
cluster-tail "$JOB_ID"
```

Expected lines (in the `.out` log, in order):
- `[run] SHA=...`
- `[run] host=...`
- `[run] container=/data/bdip_ssd/al5165/gnn_bench/gnn_bench.sif`
- `[run] RUN_ARGS=dataset=orbit model=gcn features=default labels=default finetuning=default trainer.epochs=2`
- `nvidia-smi` output showing the GPU
- Hydra config-printing output
- A wandb run URL like `wandb: View run at https://wandb.ai/teampolpetta/orbitglm/runs/...`

**Critical:** the `[run] RUN_ARGS=...` line must show the full override string. If it shows `<none>` or only partial args, the `--export=ALL,RUN_ARGS=...` quoting through cluster-submit → ssh → sbatch is broken. **In that case: STOP and fall back to the Approach B (per-combo generated scripts via --wip) from the spec's Risks section.**

Press Ctrl-C when you've seen the wandb URL.

- [ ] **Step 4: Confirm the job exits cleanly**

```bash
ssh al5165@155.105.223.17 \
    "sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed --parsable2 --noheader | head -1"
```

Expected: `<JOB_ID>|COMPLETED|0:0|<elapsed>`. If state is `FAILED` or exit code != 0:0, fetch the `.err` log and investigate.

- [ ] **Step 5: Pull the wandb URL from the log (sanity check the metrics landed)**

```bash
cluster-fetch "slurm/logs/${JOB_ID}.out" "/tmp/${JOB_ID}.out"
grep -E "wandb.+View run" "/tmp/${JOB_ID}.out"
```

Expected: prints one line with a wandb URL. Open it in a browser and confirm a 2-epoch run with non-zero metrics is recorded.

- [ ] **Step 6: No commit.**

---

### Task 14: Smoke test — batch launcher

**Files:** none in repo (consumes `slurm/run_experiments.sh`).

- [ ] **Step 1: DRY_RUN preview of the default arrays**

```bash
DRY_RUN=1 bash slurm/run_experiments.sh
```

Expected: prints two `[DRY-RUN] cluster-submit ...` lines (one for `gcn`, one for `gat`) and `Dry run complete — 2 job(s) would be submitted.` No actual submission.

- [ ] **Step 2: Edit the file to use ONE small combination plus `trainer.epochs=1`**

Open `slurm/run_experiments.sh` and temporarily set:

```bash
MODELS=("gcn")
EXTRA_OVERRIDES="trainer.epochs=1"
```

(Do NOT commit this — these are temporary tweaks for the smoke test. Revert in Step 5.)

- [ ] **Step 3: Real submission (1 job, 1 epoch)**

```bash
bash slurm/run_experiments.sh
```

Expected: prints `[submit] slurm/train.sh --export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default finetuning=default trainer.epochs=1` then `JOB_ID=...` from cluster-submit, then `Done — 1 job(s) submitted.`

Capture the JOB_ID:

```bash
# If you already saw it in the output, just note it. Otherwise:
cluster-status
```

- [ ] **Step 4: Tail the log and confirm the override `trainer.epochs=1` reached the Python invocation**

```bash
cluster-tail <JOB_ID>
```

Expected: the `[run] RUN_ARGS=` line includes `trainer.epochs=1`. The Hydra config printout near the start of the python run shows `trainer.epochs: 1`.

- [ ] **Step 5: Revert the temporary edits in `slurm/run_experiments.sh`**

```bash
git checkout slurm/run_experiments.sh
git diff slurm/run_experiments.sh
```

Expected: `git diff` shows nothing (the file is back to the committed state from Task 6).

- [ ] **Step 6: No commit.**

---

### Task 15: Open the PR and merge

**Files:** GitHub PR; merge to `main`.

- [ ] **Step 1: Sanity-check the branch is clean and pushed**

```bash
git status
git log --oneline main..HEAD
```

Expected: clean working tree, ~9 commits ahead of `main`.

- [ ] **Step 2: Open the PR**

```bash
gh pr create --base main --head feature/cluster-helper-onboarding \
    --title "Onboard repo onto /cluster-helper skill" \
    --body "$(cat <<'EOF'
## Summary
- Add `.cluster-helper.yaml` Project Manifest (project_name=gnn_bench, deploy target at `/data/bdip_ssd/al5165/gnn_bench/`).
- Replace bespoke `slurm/{config.sh,submit_job.sh,templates/}` with three flat committed scripts: `slurm/train.sh`, `slurm/sweep.sh`, `slurm/finetune.sh`.
- Rewire `slurm/run_experiments.sh` to call `cluster-submit` per combination; preserves parameter-array UX and `DRY_RUN=1` preview.
- Hydra overrides reach the job script via `$RUN_ARGS`, delivered through `sbatch --export=ALL,RUN_ARGS=...` (because `cluster-submit` forwards extras to sbatch, not to the script).
- Update `.gitignore` for `cluster_outputs/` and `slurm/logs/*` (with `.gitkeep` exception).
- Rewrite `slurm/README.md` for the new flow.
- Design spec: `docs/superpowers/specs/2026-05-21-cluster-helper-onboarding-design.md`.

## Test plan
- [x] `bash -n` syntax-check all four new shell scripts.
- [x] `DRY_RUN=1 bash slurm/run_experiments.sh` prints expected combinations.
- [x] `cluster-push-container` produces `gnn_bench.sif` locally (~3.5GB) and on the cluster.
- [x] Smoke test: 2-epoch `orbit/gcn` job runs to COMPLETED state with the full `RUN_ARGS` line visible in the log.
- [x] Batch launcher smoke test: 1-job/1-epoch real submission propagates `trainer.epochs=1` correctly.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: prints a PR URL like `https://github.com/GiggioIlGriggio/GNN_Bench/pull/<N>`. Save the PR number.

- [ ] **Step 3: Confirm the PR is open and has the right base/head**

```bash
PR_NUM=$(gh pr view feature/cluster-helper-onboarding --json number --jq .number)
gh pr view "$PR_NUM" --json title,baseRefName,headRefName,state
```

Expected: state `OPEN`, base `main`, head `feature/cluster-helper-onboarding`.

- [ ] **Step 4: Hand off to the user for review**

Ask the user to review the PR. If they approve, run Step 5; otherwise iterate on whatever they flag.

- [ ] **Step 5: Squash-merge and delete the branch**

```bash
gh pr merge "$PR_NUM" --squash --delete-branch
```

Expected: prints `Merged pull request #N`. The remote `feature/cluster-helper-onboarding` branch is deleted automatically.

- [ ] **Step 6: Clean up the worktree**

```bash
cd /home/compa/Documents/working_dir/LLM_codebase
git fetch origin
git checkout main
git pull
git worktree remove ../GNN_Bench-cluster-helper
git branch -d feature/cluster-helper-onboarding 2>/dev/null || git branch -D feature/cluster-helper-onboarding
git worktree list
```

Expected: the `GNN_Bench-cluster-helper` worktree no longer appears in `git worktree list`. `main` is up to date with the squash-merged work.

- [ ] **Step 7: No commit** (the merge commit was created by `gh pr merge`).

---

## Done

After Task 15, the repo is fully onboarded. Future workflow:

```bash
# Edit code locally, commit
git add ... && git commit -m "..."

# Submit a job
cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gat ..."
cluster-tail <JOB_ID>

# Batch
bash slurm/run_experiments.sh

# Pull artifacts
cluster-fetch outputs/runXX/checkpoint.pt

# Rebuild container when deps change
cluster-push-container
```
