# Gate P — route_scale bracket at Configuration B

## Status

Completed on 2026-08-04. Default `route_scale` changed from `1.0` to `0.75`.

## Research basis

The official `config_671B.json` sets `route_scale: 2.5` (V3 calls it `routed_scaling_factor`). No documented
rationale for this specific value was found in the technical report or secondary sources. Confirmed against the
official inference source (`inference/model.py`) that the mechanism matches this project's implementation exactly:
`route_scale` multiplies only the normalized routed-expert weights; the shared expert's output is added completely
unscaled. So `route_scale` controls how much the routed pathway's collective contribution is up- or down-weighted
relative to the always-on shared expert — not tied to `top_k` or expert count by any formula, since normalization
already makes the routed contribution sum to exactly `route_scale` regardless of `top_k`.

Given V3's value comes from a very different regime (256 experts, top-8, presumably tuned at that scale) and this
project's Gate L/Gate O experience already showed literature hyperparameters do not reliably transfer to this
project's much smaller scale, `route_scale=2.5` was treated as a prior to test, not a value to adopt directly —
same approach as Gate L's bias-rate sweep.

## Experiment

Four single-variable runs at Configuration B (155M params, 32 routed experts + 1 shared, top_k=2,
`router_bias_update_rate=1e-4`, `n_dense_layers=1`), otherwise identical: WikiText-2, batch 8, sequence 256,
1,954 steps (~4.0M tokens), seed 1337, MTP disabled.

```text
route_scale=0.75: checkpoints/compact_v3_wikitext_moe_scaleup_1m_gatep_rs075.pt
route_scale=1.0 (Gate O baseline, untouched original default): checkpoints/compact_v3_wikitext_moe_scaleup_1m.pt
route_scale=1.5: checkpoints/compact_v3_wikitext_moe_scaleup_1m_gatep_rs15.pt
route_scale=2.5 (V3's value): checkpoints/compact_v3_wikitext_moe_scaleup_1m_gatep_rs25.pt
```

## Results

| `route_scale` | Final validation PPL | Final load entropy by layer (step 1750) |
|---:|---:|---|
| 0.75 | **296.89** | 0.993, 0.979, 0.938, 0.897, 0.856 |
| 1.0 | 298.88 | 0.989, 0.967, 0.892, 0.813, 0.857 |
| 1.5 | 297.05 | 0.987, 0.907, 0.854, 0.874, 0.843 |
| 2.5 | 311.33 | 0.980, 0.857, 0.756, 0.876, 0.790 |

## Interpretation

`route_scale=2.5` (V3's value) is clearly worse here — 4.2% higher perplexity than the best setting, and visibly
worse load entropy in three of five layers throughout training. The three lower values (0.75, 1.0, 1.5) cluster
within 0.7% of each other on perplexity — likely within run-to-run noise for a single seed — but `route_scale=0.75`
has both the best perplexity and the cleanest, most uniform load entropy of the four, so it is the more decisive
signal rather than perplexity alone. The overall shape matches Gate L's earlier finding: a value tuned at V3's
much larger expert count and batch scale does not transfer to this project's scale, and in this case pushing the
routed pathway to dominate the shared pathway more strongly (higher `route_scale`) actively hurts, rather than
helping, once expert count is in the dozens rather than hundreds.

`route_scale=0.75` is adopted as the new default.

## Validation

```text
Route-scale CLI test: 1 passed
Active V3 suite: 55 passed
Four Configuration B 1M+-token CUDA runs: completed, checkpoints preserved
```

## Next gate

Gate Q: MTP loss-weight annealing (V3 anneals 0.3 -> 0.1 during the training decay phase; this project's
`mtp_weight` is currently a constant 0.3 with MTP itself disabled by default in ablations). Gate R (MLA weight
absorption) remains queued after that, with added urgency from Gate O's generation-VRAM finding.
