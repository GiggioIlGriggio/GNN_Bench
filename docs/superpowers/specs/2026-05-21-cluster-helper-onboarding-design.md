# Cluster-helper onboarding for LLM_codebase

**Date:** 2026-05-21
**Branch:** `feature/cluster-helper-onboarding`
**Status:** design approved, pending implementation plan

## Goal

Enable the `/cluster-helper` skill to drive Slurm submissions for this repo, so a user can run `cluster-submit slurm/train.sh ...` from the laptop and have the job execute on `al5165@155.105.223.17`. Replace the bespoke `slurm/` launcher with cluster-helper–native scripts while preserving the parameter-array batch workflow users rely on.

Non-goals: a CI-driven container registry; live-edit / sync-on-save dev loop; rewriting the underlying ML code.

## Decisions (resolved with user)

| Decision | Choice | Why |
|---|---|---|
| `project_name` | `gnn_bench` | Matches GitHub repo (`GNN_Bench`), short, lowercase. |
| Cluster deploy target | Brand-new sibling `/data/bdip_ssd/al5165/gnn_bench/` | Existing `/data/bdip_ssd/al5165/GNNBenchV2/` stays untouched as an archive. |
| Bespoke `slurm/` launcher | Replace with cluster-helper–native scripts | Old `sed`-substitution model is incompatible with cluster-helper's "committed bash script" convention. |
| Batch launcher (`run_experiments.sh`) | Keep, but rewire to call `cluster-submit` | Users rely on the parameter-array loop. |
| Laptop PATH install | Append `export PATH="$HOME/.claude/skills/cluster-helper/bin:$PATH"` to `~/.bashrc` | One-time, works in every new shell. |
| WandB project/entity | Hardcoded into slurm scripts (`logging.project=orbitglm`, `logging.entity=teampolpetta`) | Manifest's `wandb:` block is documentation-only — cluster-helper doesn't inject it. Override per-job via Hydra. |
| Git workflow | Worktree at `../GNN_Bench-cluster-helper`, branch `feature/cluster-helper-onboarding`, merged to `main` via PR | Per repo's `docs/agents/git-workflow.md`. |

## Pre-conditions (one-time, outside this repo)

1. **Laptop `~/.bashrc`:** append `export PATH="$HOME/.claude/skills/cluster-helper/bin:$PATH"`, then `source ~/.bashrc`.
2. **Cluster `~/.bashrc`:** append `export WANDB_API_KEY=<key>` on `al5165@155.105.223.17`. Currently absent — confirmed by `grep -c WANDB_API_KEY` returning 0. User pastes the key once; not committed anywhere.

Both are out-of-band of the deploy loop — they happen during onboarding only, not on every submit.

## In-repo changes

### Files added

- **`.cluster-helper.yaml`** — Project Manifest (committed at repo root):
  ```yaml
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
    gpus: 1
    cpus: 4
    mem: 32G
    time: "23:59:00"

  wandb:
    project: orbitglm
    entity: teampolpetta
  ```

**Args passing mechanism (constraint from cluster-helper internals):** `cluster-submit` forwards extra positional args to **sbatch as flags**, not to the job script (line 148 of `cluster-submit`: `sbatch $SBATCH_ARGS_STR "$JOB_SCRIPT"`). Hydra overrides therefore cannot be passed as positional args. We pass them via Slurm's `--export=ALL,RUN_ARGS=<args>` mechanism: each slurm script reads `$RUN_ARGS` from its environment, and callers (batch launcher or manual `cluster-submit`) construct the `--export=...` flag.

- **`slurm/train.sh`** — standard CV training. Self-contained bash, no template substitution. SBATCH headers hardcoded to `rad2 / 16cpu / rad / gpu:1 / 4 cpu / 32G / 23:59:00`. Body:
  ```bash
  set -euo pipefail
  SIF="$(pwd)/gnn_bench.sif"
  echo "[run] SHA=$(git rev-parse HEAD) host=$(hostname)"
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

- **`slurm/sweep.sh`** — identical shape; python invocation adds `--multirun` after `run_experiment.py`. Wall-time bumped to `47:59:00`.

- **`slurm/finetune.sh`** — identical shape; wall-time `11:59:00`. Hydra overrides `finetuning=<cfg>` and `finetuning.checkpoint_path=<path>` arrive via `$RUN_ARGS` like all other overrides.

- **`slurm/run_experiments.sh`** — kept name, rewired internals. Same parameter-array UX (`DATASETS`, `MODELS`, `FEATURES`, `LABELS`, `MODE`, `DRY_RUN=1`). Loops over combinations and calls `cluster-submit slurm/${MODE}.sh "--export=ALL,RUN_ARGS=dataset=$d model=$m features=$f labels=$l ..."`. Each iteration emits cluster-submit's `JOB_ID=` line so callers can grep them. Sleeps 1s between submits to avoid scheduler overload (preserved from old script).

  **Single-run manual UX:**
  ```bash
  cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default"
  ```

- **`slurm/logs/.gitkeep`** — keep dir in git, gitignore the `.out`/`.err` files inside.

### Files deleted

- `slurm/config.sh`
- `slurm/submit_job.sh`
- `slurm/templates/train.slurm`
- `slurm/templates/sweep.slurm`
- `slurm/templates/finetune.slurm`
- `slurm/templates/` (now empty)
- The *old* `slurm/run_experiments.sh` (replaced by the rewired version above)

### Files modified

- **`slurm/README.md`** — rewrite to document the new flow: single submit via `cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=... model=..."`, batch via `bash slurm/run_experiments.sh`, monitor via `cluster-tail <jobid>`. Explain that `RUN_ARGS` is the only mechanism for passing Hydra overrides through cluster-submit. Remove all references to `config.sh`, `submit_job.sh`, `sed` substitution.
- **`.gitignore`** — add `cluster_outputs/` (where `cluster-fetch` deposits files), add `slurm/logs/*` with `!slurm/logs/.gitkeep` exception. `*.sif` already covered.

### Files NOT changed

- `Dockerfile` — already produces the right environment. cluster-helper's `cluster-push-container` will use it as-is to build `gnn_bench.sif`.
- `requirements.txt` — unchanged.
- `src/`, `scripts/run_experiment.py`, `configs/` — unchanged.

## Bootstrap order (no code runs yet — see implementation plan for the per-step verification gates)

1. Add PATH line to laptop `~/.bashrc`, source.
2. Add `WANDB_API_KEY` to cluster `~/.bashrc`.
3. In the worktree: write `.cluster-helper.yaml`, three slurm scripts, rewired `run_experiments.sh`, `slurm/logs/.gitkeep`, `.gitignore` updates. Delete old bespoke slurm files. Rewrite `slurm/README.md`.
4. Commit + push the `feature/cluster-helper-onboarding` branch.
5. `cluster-push-container` — first container build + rsync to `/data/bdip_ssd/al5165/gnn_bench/gnn_bench.sif`. ~3.5 GB transfer, one-time.
6. `cluster-submit slurm/train.sh "--export=ALL,RUN_ARGS=dataset=orbit model=gcn features=default labels=default"` — smoke test. Read `JOB_ID=` from stdout. **Verify the `RUN_ARGS` env-var passing works end-to-end before relying on it in the batch launcher.**
7. `cluster-tail <JOB_ID>` — confirm CUDA + wandb both initialize cleanly.
8. If green: `gh pr create` → merge to `main` via squash.

## Risks accepted

- **Per-loop git pushes in `run_experiments.sh`:** each `cluster-submit` does `git push`. Looping N combos = N pushes of the same commit. Harmless (no-ops after the first) but visible as network chatter. Decided acceptable vs. introducing a "push once then submit N times" mode that would require changes to `cluster-submit` itself.
- **Shell-quoting of `--export=ALL,RUN_ARGS=...` through `cluster-submit` → ssh heredoc → sbatch:** the chain re-evaluates shell escaping at multiple layers. Our values are restricted to Hydra-style `key=value` overrides separated by spaces — no commas, no quotes, no backslashes. Within that envelope, `printf '%q'` (used by cluster-submit at line 121) should produce safe escaping. **Step 6 of the bootstrap explicitly validates this before the batch launcher relies on it.** If it breaks in practice, fall back to Approach B: per-combo generated scripts in `slurm/runs/` submitted via `--wip`.
- **No `--wip` exposure in batch launcher:** old script didn't have it either. Easy to add later via `WIP=1` env var if needed.
- **WandB project/entity hardcoded in three slurm scripts:** changing them requires editing three files. Trade-off chosen because the manifest's `wandb:` block isn't read by `cluster-submit` (it's template-stamp documentation only). Per-job override via Hydra (`logging.project=foo`) remains available.

## Open questions for the implementation plan

- Should the smoke test (step 6 above) be `dataset=orbit model=gcn` or something smaller/faster? Anything that exercises CUDA + a single forward pass is sufficient; default `orbit/gcn` is fine but takes longer than necessary.
- Should `.gitignore` also ignore `*.sif.tar` / `*.tar.gz` intermediate artifacts that `cluster-push-container` may produce during build? Verify by reading the `cluster-push-container` script during implementation.
- WandB key handoff mechanism: confirm with user whether they want me to ssh in and append the line interactively (they paste, I run `cat >> ~/.bashrc`), or just hand them the exact command to paste themselves.

## Out of scope

- Container registry / GitHub Actions for rebuilds.
- A `cluster-watch <jobid>` poller (cluster-helper roadmap, Phase 5).
- Re-deploying the existing `GNNBenchV2/` artifacts under the new `gnn_bench/` tree. Old runs remain readable from `/data/bdip_ssd/al5165/GNNBenchV2/` but won't be referenced by new jobs.
- Dataset uploads. Datasets remain pinned at their existing cluster paths via config; `cluster-upload-dataset` is available if needed but not exercised here.
