# Design — configurable GNN normalization (`model.norm`) + LayerNorm identity→VWM experiment

**Date:** 2026-06-04
**Branch:** `feature/gnn-norm-layernorm` (stacked on `feature/vwm-identity-r2-decomposition`).
**Status:** approved design, ready to plan + implement.

## Motivation

The GNN backbones (`gcn`, `gat`, `gin`) currently **hardcode** `nn.BatchNorm1d`
as the per-layer normalization. We want to (a) make the norm layer a
first-class, configurable choice and (b) use it to answer a concrete research
question on the identity→VWM task we just decomposed
(`reports/2026-06-04-identity-vwm-r2-decomposition.md`): **does swapping
BatchNorm → LayerNorm change performance?**

BatchNorm over a PyG graph batch normalizes each feature channel across *all
nodes of all graphs in the batch* — coupling samples and making it sensitive to
batch composition. LayerNorm normalizes each node's hidden vector across its
feature channels independently (batch-size-independent). It is a plausible
stability win and a standard A/B for GNNs.

**Scientific honesty (must be carried into the report + handoff).** identity→VWM
sits at a generalization **floor of ≈ 0** (E1 pooled +0.009; E3 −0.05…+0.01).
With no real signal to fit, LayerNorm will most likely *also* land at ≈ 0. The
experiment's value is therefore: (1) confirming the norm choice does **not**
rescue a no-signal feature, and (2) exercising the new `model.norm` knob
end-to-end on the cluster — **not** an expected performance jump. Do not frame a
null result as a failure.

## Goals / non-goals

**Goals**
- A validated `model.norm` config enum (`batch` | `layer` | `none`), default
  `batch`, wired into **all three** backbones via the shared base.
- Default behavior **byte-identical** to today (so every prior result, incl. the
  E3 BatchNorm numbers, stands unchanged).
- TDD coverage for all three norm kinds across all three backbones.
- Run identity→VWM at seeds 42/100/200 with `model.norm=layer` on the cluster;
  update the report, `EXPERIMENTS.md`, and memory once results land.

**Non-goals**
- No change to pooling, JK, heads, or the label normalizer (`label_norm_strategy`
  is a separate, target-side normalization and is out of scope).
- No new backbone architectures. No GLM/feature-set sweep (identity only).
- No graph-aware LayerNorm semantics (see decision below).

## Code design

### 1. Config — `model.norm`
- `src/configs/model_config.py`: add field `norm: Literal["batch", "layer", "none"] = "batch"`.
  Pydantic validation must **reject** any other value with a clear error (mirror
  the style of existing validated fields like `pooling`/`jk_mode`).
- `configs/model/gcn.yaml`, `configs/model/gat.yaml`, `configs/model/gin.yaml`:
  add `norm: batch` (explicit default — keeps the resolved config self-documenting).

### 2. Backbone factory (shared base)
- `src/models/backbones/base_backbone.py`: add a module-level helper
  ```python
  def build_norm(kind: str, dim: int) -> nn.Module:
      if kind == "batch": return nn.BatchNorm1d(dim)
      if kind == "layer": return nn.LayerNorm(dim)
      if kind == "none":  return nn.Identity()
      raise ValueError(f"unknown norm kind: {kind!r}")
  ```
  (Config validation already guards the value; the `raise` is defense-in-depth.)
- `gcn.py`, `gat.py`, `gin.py`: replace each
  `self.norms.append(nn.BatchNorm1d(cfg.hidden_dim))` loop body with
  `self.norms.append(build_norm(cfg.norm, cfg.hidden_dim))`. **No `forward`
  change** — `base_backbone.forward` already does `x = self.norms[i](x)` on a
  `[N, hidden_dim]` tensor, and `BatchNorm1d`/`LayerNorm`/`Identity` are all
  shape-preserving there.
- Note `gat.py` builds `self.norms` sized to `hidden_dim` (head output is
  concatenated to `hidden_dim` before the norm in the current code — verify and
  keep that invariant; the factory takes whatever dim the existing code passes).

### 3. LayerNorm flavor — DECIDED
Use plain **`nn.LayerNorm(hidden_dim)`** (per-node normalization across feature
channels). Rationale: canonical "BatchNorm→LayerNorm" swap, batch-size
independent, cleanest A/B. **Not** `torch_geometric.nn.norm.LayerNorm(mode=...)`
— graph-aware semantics would confound the comparison and are out of scope.

## Test plan (TDD — write tests first)

New `tests/test_backbone_norm.py` (or extend an existing backbone test module if
one exists — check first):
1. For each backbone in {gcn, gat, gin} and each kind in {batch, layer, none}:
   build the backbone with `cfg.norm=kind` and assert `type(backbone.norms[0])`
   is `BatchNorm1d` / `LayerNorm` / `Identity` respectively, and `len(norms) ==
   num_layers`.
2. Forward-pass smoke per (backbone, kind): a small random `Data` batch runs and
   returns `[N, out_dim]` without error (and is finite).
3. Config validation: `ModelConfig(..., norm="bogus")` raises; `norm` defaults to
   `"batch"` when omitted.
4. Default-unchanged regression: with `norm` omitted/`batch`, `norms[0]` is
   `BatchNorm1d` (guards the "prior results still valid" claim).

Run the existing model/training tests to confirm no regression. Expect the two
**pre-existing** `test_training.py` smoke failures (stale ADR-0012 path
assertion — see `reports/2026-06-04-…` §8) to remain, unrelated.

## Experiment (cluster)

Three jobs, mirroring E3 exactly except the norm:
- **Recipe:** `experiment_name=gcn-pnc-sc-vwm-identity-layernorm-seed<S> features=identity dataset=pnc model=gcn labels=pnc_VWMdprime model.norm=layer logging.project=orbitglm trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 trainer.search_space=configs/sweeper/gcn_embedding_dim.yaml trainer.hpo_metric=val_r2 trainer.seed=<S>` for `S in {42,100,200}`.
- **Deploy:** push `feature/gnn-norm-layernorm` first; `cluster-submit` deploys
  from the pushed SHA (which must therefore contain the `model.norm` code).
- **Submission discipline (cluster-helper rules 4–5):** show `cluster-gpus`,
  ask the user for the **node** (+ GPU type) and confirm the **job names** before
  submitting. ~16 h each; `gpunode01/rtx2080` is the E3 precedent.
- **Comparator:** the BatchNorm baseline is the already-completed E3 (jobs
  360854/360855/360856) — **do not re-run BatchNorm.**

## Documentation updates (after results land)

- **Report:** append a section to `reports/2026-06-04-identity-vwm-r2-decomposition.md`
  (e.g. "§9 — LayerNorm vs BatchNorm at the identity floor") with a 3-seed table
  reporting **both** R² flavours (pooled via `scripts/pooled_vs_meanfolds.py`,
  fetched with **project-root-relative** `cluster-fetch` paths) next to the E3
  BatchNorm numbers. State the verdict honestly (likely "no rescue of the floor").
- **EXPERIMENTS.md:** new `## Batch 2026-06-<dd>` entry, 3 per-job blocks +
  summary table + wandb run links (use `backfill-experiment-results`).
- **Memory:** extend `identity_vwm_r2_investigation.md` with the LayerNorm result
  and a new `model.norm` knob note.

## Branch / PR

- Stacked on `feature/vwm-identity-r2-decomposition`; base the PR on that branch
  (or on `main` after #31 merges — confirm at PR time). Per `docs/agents/git-workflow.md`:
  commit per coherent change, explicit filenames, imperative subjects,
  `Co-Authored-By: Claude` trailer, never `git add -A`.
- Suggested commit slices: (1) `model.norm` config + validation; (2) `build_norm`
  factory + backbone wiring + tests; (3) experiment doc/results (later).

## Acceptance criteria

1. `model.norm` ∈ {batch, layer, none} resolves through Hydra and validates;
   unknown values rejected.
2. All three backbones honor it; default `batch` reproduces today's modules.
3. New norm tests green; no new regressions beyond the 2 known stale failures.
4. Three `model.norm=layer` identity→VWM jobs submitted and COMPLETED; both R²
   flavours recomputed.
5. Report + EXPERIMENTS.md + memory updated with an honest BatchNorm-vs-LayerNorm
   comparison.

## Open risks

- **Symlinked venv** in the worktree (`.venv` → the sibling worktree's venv): has
  working torch/PyG companion libs; verify `torch_geometric` imports before
  trusting test runs (see memory `llm_venv_pyg_companion_libs`).
- **GAT norm dim**: confirm `gat.py` passes the correct dim to the norm (head
  concatenation) so `LayerNorm`/`BatchNorm` get matching shapes — covered by the
  forward-pass smoke test.
- **Floor result**: the experiment may show ≈0 for both norms; that is a valid,
  reportable outcome, not a bug.
