# Runbook: PNC Age→VWM Transfer (arms B1–B4 + source runs)

Leakage-safe nested-CV transfer: an **age-pretrained backbone** is loaded per
outer fold into a **VWM** prediction run. Implements thesis arms **B1–B4** plus
the two pre-staged age "source" runs they reuse. Baselines **A4/A5** are in the
[A4/A5 section](#a4--a5-baselines) below.

Design spec: `docs/superpowers/specs/2026-06-13-pnc-age-vwm-transfer-design.md`.
Plan: `docs/superpowers/plans/2026-06-13-pnc-age-vwm-transfer-infrastructure.md`.

---

## ⚠️ 0. The cohort requirement (READ FIRST — not optional)

The transfer design requires the age source run and every VWM run to operate on
an **identical subject set**, so their nested-CV outer folds match byte-for-byte
— the `SourceBackboneProvider` alignment guard **hard-fails** otherwise
(`ValueError: fold-index mismatch … refusing to transfer (would leak test
subjects)`).

On PNC this is **not** automatic. In `T0/Tabular_data/PNC_ALL_SCORES.csv`:

| column | non-NaN subjects |
|--------|------------------|
| `age_at_cnb` (age target) | 9482 |
| `VWM_overall_dprime` (VWM target) | 1454 |
| both | 1454 (every VWM subject has an age) |
| age-only (age present, VWM NaN) | 8028 |

The dataset NaN-filters on the **current target**, so an unrestricted age run
loads ~thousands of age-only subjects that the VWM run never sees → the two fold
partitions can never coincide. The fix: restrict **all** PNC arms to the
**loadable VWM cohort** — subjects with both a connectivity graph *and* a
non-NaN VWM score — via a subject allowlist.

This is also the scientifically correct choice: a controlled transfer-vs-scratch
comparison must hold the cohort fixed (the age advantage must come from the
*label*, not from extra subjects).

### Generate the cohort allowlist (once)

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/make_pnc_vwm_cohort.py
# writes configs/subject_lists/pnc_vwm_cohort.txt  (973 subjects, identity carrier)
```

Then pass `dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt` to
**every** command below (source age runs, B1–B4, and — for a fair comparison —
the A3 from-scratch and C-arm runs too). The allowlist is **generated locally and
not committed** — `configs/subject_lists/*.txt` is git-ignored, because it lists
PNC (restricted-access) subject IDs. Regenerate it with the command above on each
machine (and whenever the PNC derivatives change).

**GLM-carrier caveat (B3/B4):** the committed cohort is computed with
`features=identity` (973 subjects, needs only an SC graph). The GLM carrier also
needs per-subject GLM maps; if any cohort subject lacks them, regenerate a
GLM-specific allowlist and use it for the GLM source + B3/B4:

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/make_pnc_vwm_cohort.py \
    configs/subject_lists/pnc_vwm_cohort_glm.txt pnc_VWMdprime glm_diagonal
```

---

## 1. Key facts

- **Age column:** `age_at_cnb` — use `labels=pnc_default` (its target).
- **VWM column:** `VWM_overall_dprime` — use `labels=pnc_VWMdprime` (its target).
- **Carriers ↔ arms:** ID source → **B1/B2** (`features=identity`); GLM source →
  **B3/B4** (`features=glm_diagonal`). A B-arm's `features=` **must equal its
  source carrier** so `node_feat_dim` (and thus the input-layer/backbone shape)
  matches the loaded weights, else Task-6's "did not fully load" guard fires.
  (With the current single-contrast `glm_diagonal`, both carriers happen to be
  400-dim, but never rely on that — always pair a B-arm with its own carrier.)
- **Frozen vs fine-tune is the arm, not an HP:** B2/B4 = `transfer=from_age_frozen`
  (`frozen_layers=['backbone']`, head-only); B1/B3 = `transfer=from_age_ft`
  (`frozen_layers=[]`, full fine-tune).
- **HPO search space (nested inner HPO):** pass the sweeper YAML as a *file path*
  via `trainer.search_space=...`. ⚠️ The nested runner reads `trainer.search_space`
  directly — `sweeper=<name>` only configures Hydra's *multirun* plugin and does
  **not** feed the nested inner HPO. Source runs use `source_age_pinned.yaml`
  (pins backbone arch); B-arms use `transfer_finetune.yaml` (optimizer + head
  only — backbone shape is frozen by the loaded weights).
- **Paper preset:** `trainer.n_repetitions=10 trainer.n_outer_folds=5
  trainer.inner_hpo_trials=20`. These belong on the **Slurm cluster**
  (`cluster-helper`), not locally.
- **Logging:** set `logging.project=<wandb-project>` for real runs (the trainer
  default project drifts — see memory `wandb_project_default_drift`).

---

## 2. Step A — the two age source runs (pre-staged, reused)

Each emits per-(rep,fold) backbones + a `fold_indices.json` manifest under
`checkpoints/<experiment_name>-<uid>/`. Both are **stratified on VWM**
(`stratify_target=VWM_overall_dprime`) so their outer folds match every VWM run.

### ID-carrier source (for B1/B2)

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt \
  model=gcn features=identity labels=pnc_default \
  stratify_target=VWM_overall_dprime \
  trainer.search_space=configs/sweeper/source_age_pinned.yaml \
  experiment_name=src_age_id \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  logging.project=age_vwm_transfer
```

### GLM-carrier source (for B3/B4)

⚠️ If the GLM carrier needs subjects to also have GLM maps (see the §0
GLM-carrier caveat), first generate `pnc_vwm_cohort_glm.txt` and use it **here and
for B3/B4** in place of `pnc_vwm_cohort.txt` — the GLM source and its B-arms must
share one cohort just as the ID side does. If every cohort subject has GLM maps,
the identity cohort below is fine.

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt \
  model=gcn features=glm_diagonal labels=pnc_default \
  stratify_target=VWM_overall_dprime \
  trainer.search_space=configs/sweeper/source_age_pinned.yaml \
  experiment_name=src_age_glm \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  logging.project=age_vwm_transfer
```

After each, record the run name: `ls -dt checkpoints/src_age_id-* | head -1`
(and `src_age_glm-*`). That directory is the `source_checkpoint_root` below.

---

## 3. Step B — the four transfer arms

Set `transfer.source_checkpoint_root` to the matching source run directory, and
`features=` to its carrier. Same cohort + `stratify_target` is unnecessary here —
the VWM run stratifies on its own target (VWM) by default, which matches the
source's VWM stratification.

| arm | source | features | transfer preset | frozen |
|-----|--------|----------|-----------------|--------|
| B1 | ID  | identity     | `from_age_ft`     | `[]` (full FT) |
| B2 | ID  | identity     | `from_age_frozen` | `['backbone']` |
| B3 | GLM | glm_diagonal | `from_age_ft`     | `[]` (full FT) |
| B4 | GLM | glm_diagonal | `from_age_frozen` | `['backbone']` |

Template (substitute the carrier, transfer preset, source dir, experiment name):

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt \
  model=gcn features=<identity|glm_diagonal> labels=pnc_VWMdprime \
  transfer=<from_age_ft|from_age_frozen> \
  transfer.source_checkpoint_root=checkpoints/<src_age_id|src_age_glm>-<uid> \
  trainer.search_space=configs/sweeper/transfer_finetune.yaml \
  experiment_name=<b1|b2|b3|b4>_age_vwm \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  logging.project=age_vwm_transfer
```

E.g. **B2** = `features=identity transfer=from_age_frozen
transfer.source_checkpoint_root=checkpoints/src_age_id-<uid>
experiment_name=b2_age_vwm`.

If the run aborts with `fold-index mismatch`, the source and VWM runs used
different cohorts/stratification — confirm both used the **same**
`subject_list_file` and that the source used `stratify_target=VWM_overall_dprime`.
If it aborts with `Source backbone did not fully load`, the B-arm's `features=`
doesn't match the source carrier (backbone shape differs).

---

## 4. Local smoke check (fast preset, plumbing only)

Validates the wiring end-to-end in seconds on a 60-subject subset — **not** a
result. Uses `inner_hpo_trials=0` and `epochs=2`. (This exact sequence was run
to validate the implementation: source writes the manifest, the aligned arm
transfers cleanly, and a misaligned source trips the guard.)

```bash
# 60-subject smoke subset of the cohort
PYTHONPATH=$(pwd) .venv/bin/python -c "from pathlib import Path; \
ids=[l.strip() for l in Path('configs/subject_lists/pnc_vwm_cohort.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]; \
Path('configs/subject_lists/pnc_vwm_smoke60.txt').write_text('\n'.join(ids[:60])+'\n')"

# 1) source age run (stratified on VWM)
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_vwm_smoke60.txt \
  model=gcn features=identity labels=pnc_default stratify_target=VWM_overall_dprime \
  experiment_name=src_age_id_smoke \
  trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=0 trainer.epochs=2 \
  logging.enabled=false
# -> logs "Wrote fold-index manifest: checkpoints/src_age_id_smoke-<uid>/fold_indices.json"

# 2) B2 transfer arm off that source (guard should PASS; backbone frozen, head reinit'd)
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_vwm_smoke60.txt \
  model=gcn features=identity labels=pnc_VWMdprime \
  transfer=from_age_frozen \
  transfer.source_checkpoint_root=checkpoints/src_age_id_smoke-<uid> \
  experiment_name=b2_smoke \
  trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=0 trainer.epochs=2 \
  logging.enabled=false
# -> "Transfer enabled …", "Froze 16 parameters …", no "fold-index mismatch", completes.

# 3) guard fires on misalignment: a source WITHOUT stratify_target (age-stratified folds)
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_vwm_smoke60.txt \
  model=gcn features=identity labels=pnc_default \
  experiment_name=src_age_misaligned_smoke \
  trainer.n_repetitions=1 trainer.n_outer_folds=2 trainer.inner_hpo_trials=0 trainer.epochs=2 \
  logging.enabled=false
# then point B2 at it -> hard crash:
#   ValueError: fold-index mismatch … refusing to transfer (would leak test subjects).
```

To also smoke the **inner-HPO** path, add
`trainer.search_space=configs/sweeper/transfer_finetune.yaml
trainer.inner_hpo_trials=2` to step 2 (validates the restricted sweeper composes
with the frozen backbone — head dims vary, backbone shape stays fixed).

---

## A4 + A5 baselines

Two control baselines for the A-table. Both run on the **same VWM cohort** as the
B-arms (`dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt`) so
every number in the table refers to one population.

### A4 — GLM → age (specificity control)

"Do the GLM node features carry *age* signal?" A normal nested GNN run with the
GLM carrier and target = age, **age-stratified** (no `stratify_target` — this is a
standalone reported number, *not* the VWM-stratified GLM source run of §2). If
GLM→age ≪ GLM→VWM, the GLM vector is VWM-specific rather than a generic
maturation signal. (A prior cross-cohort GLM→age control found r²≈0.04–0.11; this
A4 reruns it within the fixed VWM cohort for an apples-to-apples A-table.)

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_experiment.py \
  dataset=pnc dataset.subject_list_file=configs/subject_lists/pnc_vwm_cohort.txt \
  model=gcn features=glm_diagonal labels=pnc_default \
  trainer.search_space=configs/sweeper/source_age_pinned.yaml \
  experiment_name=a4_glm_to_age \
  trainer.n_repetitions=10 trainer.n_outer_folds=5 trainer.inner_hpo_trials=20 \
  logging.project=age_vwm_transfer
```

(Use `trainer.search_space=…`, not `sweeper=…` — see §1. If you generated a
GLM-specific cohort for B3/B4, use that same allowlist here.)

### A5 — age → VWM, no graph (trivial developmental floor)

"How much of VWM is just chronological age?" The classical-ML harness only
accepts graph-derived inputs (`model.mlp_input ∈ {adjacency, node_features,
both}`) — there is **no scalar-covariate (age-only) input mode**, so A5 is **not**
a classical-harness run. It is the closed-form floor `r² = corr(age, VWM)²`
computed on the fixed VWM cohort, recorded directly in `EXPERIMENTS.md`.

Observed on the 973-subject cohort: **corr(age, VWM_overall_dprime) = 0.229 →
A5 r² = 0.052** (a 5-fold out-of-sample age→VWM linear fit gives r² = 0.050,
confirming the floor). Any B/C arm must beat ~0.05 to demonstrate the graph adds
anything over age alone.

Reproduce:

```bash
PYTHONPATH=$(pwd) .venv/bin/python - <<'PY'
import pandas as pd, numpy as np
from pathlib import Path
csv = "/media/compa/DATA1/Compa/DATA_DERIVATIVES/PNC/T0/Tabular_data/PNC_ALL_SCORES.csv"
df = pd.read_csv(csv, low_memory=False)
df["key"] = df["SUBJID"].astype("Int64").astype(str).str[-10:].str.zfill(10)
cohort = [l.strip().replace("sub-", "")
          for l in Path("configs/subject_lists/pnc_vwm_cohort.txt").read_text().splitlines()
          if l.strip() and not l.startswith("#")]
sub = df[df["key"].isin(cohort)]
age = pd.to_numeric(sub["age_at_cnb"], errors="coerce")
vwm = pd.to_numeric(sub["VWM_overall_dprime"], errors="coerce")
m = age.notna() & vwm.notna()
r = np.corrcoef(age[m], vwm[m])[0, 1]
print(f"n={m.sum()}  corr={r:.4f}  A5 floor r2={r**2:.4f}")
PY
```

