# Design: PNC Age→VWM Transfer vs. Concatenation — Implementation in the Nested-CV GNN Pipeline

- **Status:** approved (brainstorm) — ready for implementation plan
- **Date:** 2026-06-13
- **Thesis:** `LLM Wiki/Transfer Learning/wiki/theses/pnc-age-vwm-transfer-vs-concatenation.md`
- **Scope:** Implements the 7-arm experiment matrix (A4, A5, B1–B4, C1, C2) on PNC / Schaefer-400 / SC GCN, under nested CV. D1 (@node surgery) deferred per thesis.

This spec is the *implementation* design. The scientific design, predictions, and falsification criteria live in the thesis. Here we specify **what to build in the codebase** to run the matrix, and the seams/tests that keep it leakage-safe.

---

## 1. Context: what already exists

A prior infrastructure sweep established that ~80% of the matrix is already runnable:

- **Nested CV** (`src/training/nested_cross_validation.py`): 10 reps × 5 outer × 20 inner Optuna, HPO on `val_r2`, refit-on-TrainVal, per-(rep,fold) checkpoints saved with `model_config`/`feature_config` (`nested_cross_validation.py:504–516`). All leakage protection behind `FoldBarrier` (ADR-0009), fit on outer-train only.
- **Flat finetune subsystem** (`src/finetuning/finetuner.py`): leakage-safe per-fold transfer (fold *k* loads pretrain fold *k*), `_reinit_head`, `_freeze_layers`, `build_param_groups` (differential LR). **Runs flat CV only** — *not* nested.
- **Carriers**: `identity` (one-hot) and `glm_diagonal` / `glm_scalar` node features (`src/datasets/feature_builder.py`).
- **Model seam**: `UnimodalBrainGNN` = backbone → pool → head, clean prefix-based freeze (`backbone`).

**Decision (locked):** the transfer arms run under **nested CV**, not the existing flat finetuner, so they are directly comparable to the A3 baseline (which was nested). The flat finetuner is not used for this thesis.

### Arm → infrastructure map

| Arm | Needs | New code? |
|----|-------|-----------|
| A4 (GLM→age) | normal nested run, target=age, GLM carrier | none (config) |
| A5 (age→VWM, no graph) | classical baseline harness, 1 feature = age | none (config) |
| B1/B2 (ID, age→FT/frozen) | transfer infra + ID-age source ckpts | §2, §3 |
| B3/B4 (GLM, age→FT/frozen) | transfer infra + GLM-age source ckpts | §2, §3 |
| C1 (GLM + age-emb @head) | @head covariate (age) | §4 |
| C2 (ID age-FT + GLM @head) | transfer infra + @head covariate (GLM vec) | §2,§3,§4 |
| D1 | — | **deferred** |

---

## 2. Source-checkpoint generation + fold-alignment guard

### Problem
The nested CV computes folds from `bins = stratify_bins(labels)`, and `labels` is the **target**. An age run stratifies on age; a VWM run on VWM → **different fold membership** → loading an age-fold-*k* backbone into VWM-fold-*k* leaks VWM test subjects into the age pretraining. Silent, inflated R².

### Construction (P1 — pre-staged, reuse nested CV twice)
1. **Stratification override.** A new knob forces a run to bin folds on a *named vector* (VWM dprime) regardless of the regression target. Source age runs and all VWM transfer runs use the same override ⇒ byte-identical outer/inner folds.
2. **Source runs.** Two normal nested runs, target=`age`, stratified-on-VWM: one ID-carrier, one GLM-carrier. They already emit per-(rep,fold) refit checkpoints (trained on outer-TrainVal, never outer-test). These are **reused** across B1/B2, B3/B4, and C2.

### Alignment guard (belt + suspenders)
- **Runtime invariant.** Source runs additionally persist `fold_indices.json` per (rep,fold): sorted `test_idx` and `train_val_idx`. The transfer run, when loading source ckpt for (rep,fold), **asserts** its own current `test_idx`/`train_val_idx` match exactly and **hard-fails** on mismatch. The guard verifies *actual subject sets*, not the stratification convention — so drift crashes loudly instead of inflating a number.
- **Tests** (required):
  - unit: same `(n, bins, seeds, n_outer)` ⇒ identical folds; different bins ⇒ different folds (proves the override is necessary).
  - integration: synthetic source+target pair end-to-end — guard passes when stratification is pinned, raises when it isn't.

### Leakage stance (documented)
Outer-test estimate is **unbiased** (source refit never saw outer-test). Residual is *inner-HPO* leakage only: the source backbone saw inner-val connectivity with **age** labels (never VWM). Standard and accepted; it can mildly affect HP selection, not the reported test number.

---

## 3. Transfer integration into nested CV + HPO

### Factory wrapper (per outer fold)
Inside `_run_outer_fold` (where `rep`/`fold`/`train_val_idx` are in scope), wrap the base `model_factory`: build architecture → `load_partial_state_dict(source_ckpt[rep,fold])` → `_reinit_head` → `_freeze_layers`. All inner Optuna trials and the refit of that fold share one source backbone — no counter hack. The wrapper runs the alignment assertion (§2) before loading.

`_freeze_layers` / `build_param_groups` are **lifted out of `finetuner.py`** into a shared module (e.g. `src/finetuning/transfer_ops.py`) so the flat Finetuner and the nested wrapper call one implementation.

### Restricted HPO search space (forced by transfer)
The backbone shape is frozen by the loaded weights → architecture HPs (hidden_dim, num_layers) **cannot** vary. A **new sweeper YAML** for transfer arms ranges only over what is still free:
- `lr`, `weight_decay`
- the reinit'd **head's** dropout / hidden_dim
- early-stop budget
- (FT arms only) backbone/head **differential LR** via `lr_groups`

Frozen-vs-FT (B2/B4 vs B1/B3) is **an arm**, set by `frozen_layers`, **not** an HP.

### Integration nit
The nested `Trainer` must build its optimizer from param-groups / respect `requires_grad`. The flat path threads `ft_trainer._param_groups`; the nested path needs the same hook so frozen params are excluded and differential LR applies.

---

## 4. @head covariate injection (C1, C2)

One generic mechanism, two covariate types. A per-subject covariate vector rides on `data.u`; the model concatenates it to the pooled embedding before the head (`head.embedding_dim = node_dim + cov_emb_dim`), via a small **covariate encoder**.

| Arm | `data.u` | Encoder | Backbone |
|----|----------|---------|----------|
| C1 | chronological age (scalar) | **MLP** 1→k (k HPO'd) | GLM node feats, from scratch |
| C2 | `glm_scalar` flattened → 400-vec (per-node GLM, the classical-baseline representation) | identity/linear | ID, age-pretrained, fine-tuned (§3) |

C2 composes with transfer for free: the head is reinit'd anyway and `load_partial_state_dict` skips it, so the wider head input is a non-issue; the backbone loads cleanly.

### Fold-safe covariate normalization (the real new code here)
A head covariate must be standardized **fit-on-train-only** or it reintroduces leakage. `FoldBarrier` grows covariate support: fit standardizer on outer-train `data.u`, apply to inner-train/val/test. Age → z-score; GLM-vec → per-column z-score (reuse `glm_normalize` logic on a graph-level vector). Tested alongside the existing barrier leakage tests.

### Config
A `head_covariate` block: `type: {none, age_emb, glm_vector}`, `encoder: {mlp, identity}`, dims, normalization. `none` ⇒ today's behavior unchanged (the covariate path is opt-in and inert by default).

---

## 5. Baselines

- **A4 (GLM→age):** normal nested run, target=age, GLM carrier, **age-stratified** (a reported A-table number; distinct from the VWM-stratified GLM-age *source* run used only for checkpoints).
- **A5 (age→VWM, no graph):** classical baseline harness, single input feature = age. Mandatory trivial floor; no GNN.

---

## 6. Components & seams (summary)

| Component | File(s) | Change |
|----------|---------|--------|
| Stratification override | splits / trainer config | new knob: bin on named vector |
| Fold-index manifest + guard | nested CV + transfer wrapper | persist `fold_indices.json`; assert on load |
| Transfer factory wrapper | nested CV `_run_outer_fold` | wrap `model_factory`; load+reinit+freeze |
| Shared freeze/param-group ops | `transfer_ops.py` (lifted from finetuner) | one impl for flat + nested |
| Restricted transfer sweeper | `configs/sweeper/transfer_*.yaml` | new search space |
| Nested optimizer param-groups | `Trainer` | respect requires_grad / accept groups |
| @head covariate path | model (unimodal/head) + `data.u` builder | opt-in concat + encoder |
| Covariate normalization | `FoldBarrier` | fit-on-train standardizer |
| `head_covariate` config | model config | new block, default `none` |

---

## 7. Testing strategy

- **Fold alignment** (§2): unit (splits determinism) + integration (guard pass/raise). **Required, per user.**
- **Transfer wrapper**: a source backbone loads into every fold; head reinit'd; frozen params have `requires_grad=False`; param-groups exclude frozen.
- **Covariate barrier**: standardizer fit on train only (no test stats leak); `none` path is a no-op vs current behavior.
- **Regression guard**: a non-transfer, no-covariate run reproduces current A2/A3 numbers (the changes are inert when off).

---

## 8. Phasing (for writing-plans)

1. **Transfer infra** — stratification override, fold-index manifest + guard + tests, transfer factory wrapper, shared `transfer_ops`, restricted sweeper, nested optimizer hook → unlocks **B1–B4** + source runs.
2. **@head covariate** — `data.u` builder, covariate encoder + model concat, barrier normalization, `head_covariate` config → unlocks **C1, C2**.
3. **Baselines** — A4 (config), A5 (classical harness) → runnable in parallel any time.

---

## 9. Out of scope

- D1 (@node input-dim surgery + differential gradient schedule) — deferred per thesis.
- Cross-dataset transfer, other atlases, self-supervised pretraining, other targets.
- ORBIT replication — underpowered (N≈95–130); optional later, not part of this build.

---

## 10. Risks / open items

- **Compute.** Two source nested runs + 4 transfer arms + 2 C-arms + A4, each 10×5×20. Source runs are reused, but transfer arms re-run the full nested loop. Budget on the cluster accordingly.
- **Source-run HPO.** The source age runs do their own inner HPO; the saved per-fold refit checkpoint is the winning-trial refit. The transfer arms inherit that fixed backbone — confirm a single good age architecture rather than letting source-run HPO vary backbone shape across folds (else different folds yield different backbone widths and the transfer wrapper must build per-fold architectures from each fold's `model_config.json`). **Resolve in the plan:** pin the source backbone architecture (no architectural HPO in the source runs) so all folds share one shape.
