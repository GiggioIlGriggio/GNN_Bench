# ADR-0013: Matched HPO search spaces for cross-backbone comparison

## Status
Accepted — 2026-06-04

## Context
The 2026-06-03 VWM GLM node-feature batch (and its 2026-05-29 predecessor)
compared GLM-derived node-feature encodings on a **single** backbone, GCN,
holding the model and the nested-CV protocol fixed. Its finding — diagonal GLM
forms (`glm_diagonal`, `identity_glm_diagonal`) rank above the scalar forms —
was deflationary: under the ADR-0008 corrected paired t-test the top conditions
were **not** significantly different, and the pipeline is GPU-nondeterministic so
a byte-identical re-run did not reproduce the numbers.

We now want to know whether that node-feature **effect generalises across
architectures**: does diagonal-beats-scalar replicate when the message-passing
layer is swapped? This requires running the same 7-cell node-feature matrix on
GAT, GIN, Graph Transformer, and BrainGNN (GCN stays as the existing reference).

The obstacle is HPO. A robustness claim is only as clean as the tuning protocol
behind it, but the backbones do not expose the same hyperparameters:
- GAT and the Graph Transformer have an attention **`heads`** count (with the
  hard constraint `hidden_dim % heads == 0`, asserted in both backbones) that
  GCN/GIN lack.
- BrainGNN is a different model class entirely: `num_layers` is structurally
  fixed at 2 (two TopK pooling stages), `pooling`/`jk_mode` are no-ops, and it
  carries architecture-specific knobs (`pool_ratio`, `roi_embed_dim`, and the
  auxiliary loss weights `lambda_topk/unit/consist`) that the generic backbones
  do not have.

So a single shared sweeper is impossible, and the *naïve* alternatives are both
bad: reusing `gcn_embedding_dim.yaml` verbatim would leave the attention models
and BrainGNN untuned on their own degrees of freedom, while adopting each
backbone's pre-existing bespoke sweeper (e.g. `braingnn_fc_vwm.yaml`, which also
tunes `lr`/`weight_decay`) would let some backbones tune the optimiser while
others do not — an asymmetry that confounds the comparison.

## Decision
Adopt a **shared architectural base + per-model personalisation, with the
optimiser held out of HPO for every backbone.**

1. **Shared base.** The eight generic `ModelConfig` knobs from
   `gcn_embedding_dim.yaml`: `embedding_dim`, `hidden_dim`, `num_layers`,
   `dropout`, `pooling`, `jk_mode`, `head_hidden_dim`, `head_num_layers`.
2. **GIN** reuses `gcn_embedding_dim.yaml` unchanged (no personalisation needed).
3. **GAT / Transformer** use `gat_embedding_dim.yaml` /
   `transformer_embedding_dim.yaml` = base **+ `model.heads: choice(1, 2, 4, 8)`**.
   Every `hidden_dim` choice (32/64/128/256) is divisible by every `heads`
   choice, so no trial can violate the backbone assertion.
4. **BrainGNN** uses `braingnn_vwm_matched.yaml` = the *applicable* base
   (`embedding_dim`, `hidden_dim`, `dropout`, `head_hidden_dim`,
   `head_num_layers`; dropping `num_layers` and `pooling`/`jk_mode`) **+** its
   personalisation (`model_params.pool_ratio`, `roi_embed_dim`,
   `lambda_topk/unit/consist`). This is `braingnn_fc_vwm.yaml` minus
   `lr`/`weight_decay`, with the base knobs aligned to the generic grid.
5. **Optimiser held out.** No backbone tunes `lr`/`weight_decay`; the generic
   sweeper never did, so BrainGNN must not either.
6. **Everything else identical to 2026-06-03.** `dataset=pnc`,
   `labels=pnc_VWMdprime`, 10×5 nested CV, `inner_hpo_trials=20`,
   `hpo_metric=val_r2`, `features.glm_normalize=true`, 300 epochs,
   `contrast-2back_vs_0back`, edge `weight`, GLM `zmap`. Nondeterminism is
   **not** addressed (kept as-is) so each run is comparable to the GCN reference.

Sweeper files are self-contained (no Hydra defaults inheritance) so each run's
search space is auditable in one file. The `n_trials` field in each sweeper is
vestigial under nested CV — the inner loop count comes from
`trainer.inner_hpo_trials` — but is kept for parallelism and standalone
`--multirun` use.

## Consequences
- **The clean, primary comparison is *within* each backbone:** the
  diagonal-vs-scalar contrast runs under a protocol that is identical except for
  the node features, so the ADR-0008 corrected test is valid there.
- **Cross-backbone *absolute* r² gaps are confounded** and only secondary:
  the attention models and BrainGNN tune extra degrees of freedom (`heads`,
  `model_params`) the generic base lacks, so an absolute-r² ranking partly
  reflects per-backbone tuning surface, not architecture alone. Report it with
  that caveat; do not over-claim a "best backbone".
- The inner HPO remains a noise-dominated single holdout (per the audit), so
  each cell is a single noisy draw, not a reproduction.
- Enabling the Transformer backbone required a new `configs/model/transformer.yaml`
  (none existed); the backbone class already existed.
