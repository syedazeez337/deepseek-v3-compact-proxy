# Gate N — Dense-layer prefix, compute-matched to DeepSeek-V3's own convention

## Status

Completed on 2026-08-04, after a miscalibrated first attempt. Default `n_dense_layers` changed from `0` to `1`.

## Research basis

This is the first gate of the "true to DeepSeek-V3, end to end" roadmap agreed after Gate M. DeepSeek-V3's
official `config_671B.json` sets `n_dense_layers: 3` (of 61 total layers) — the first layers use a plain dense FFN,
the rest use MoE. Critically, the dense FFN's hidden dimension (`inter_dim: 18432`) is not arbitrary: it equals
`(n_shared_experts + n_activated_experts) * moe_inter_dim = (1 + 8) * 2048 = 18432` exactly. DeepSeek sizes the
dense-layer FFN so its *active compute* matches what a routed layer would spend, not so its *stored* parameters
match — a routed layer always has far more stored parameters (256 experts) than a size-matched dense layer.

Sources: https://arxiv.org/pdf/2412.19437 (DeepSeek-V3 Technical Report),
https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json (official config),
https://deepwiki.com/deepseek-ai/DeepSeek-V3/3-model-architecture.

## Implementation

- `v3_config.py`: added `n_dense_layers: int` (validated `0 <= n_dense_layers <= n_layer`).
- `v3_block.py`: `CompactV3Block` now accepts an explicit `use_moe` override and a `dense_hidden_dim` override,
  instead of always reading `config.use_moe`/`config.expert_hidden_dim` directly.
- `compact_v3_model.py`: builds each block with `use_moe = config.use_moe and layer_index >= n_dense_layers`, and
  for layers forced dense by the prefix, sizes the dense FFN's hidden dim as
  `(n_shared_experts + top_k) * expert_hidden_dim` to match MoE active compute, mirroring V3's own ratio.
- `v3_cli.py`: added `--n-dense-layers` (default `1` after this gate).
- Added `test_n_dense_layers_prefix_uses_dense_ffn`, `test_n_dense_layers_out_of_range_rejected`,
  `test_n_dense_layers_matches_moe_active_compute` (`tests/test_compact_v3_model.py`), and
  `test_n_dense_layers_flag_wires_into_config` (`tests/test_v3_cli.py`). Full active suite: 54 passed.
- Verified CUDA prefill/decode equivalence still holds with a mixed dense/MoE model (per-layer caching is
  unaffected by which layers are dense).

## Failure recorded and corrected

The first implementation gave `CompactV3Block`'s dense-layer FFN the plain `config.expert_hidden_dim` (384) —
the same hidden dim as a *single* expert, not matched to the MoE layer's active compute (`(1 shared + 2 top_k) *
384 = 1152`). That first run (`checkpoints/compact_v3_wikitext_moe_1m_gaten_dense1.pt`) measured validation PPL
**593.01**, apparently worse than the Gate M baseline's 580.62 — but this was confounded: the dense layer had only
1/3 the active FFN compute of the MoE layers beside it, so the comparison was "does giving one layer 3x less
compute hurt," not "does a properly-sized dense prefix layer help." This was caught before drawing a conclusion,
fixed to match V3's own compute-matching convention, and rerun. Separately, adopting the new default surfaced a
stale assumption in `test_model_backward` (it read `model.blocks[0].moe...` assuming layer 0 was always MoE);
fixed by passing `n_dense_layers=0` explicitly since that test specifically exercises the MoE gradient path.

## Experiment (corrected)

Single controlled variable relative to Gate M (`n_dense_layers`: 0 -> 1, dense FFN capacity-matched), everything
else identical: dataset/tokenizer, context 64, batch 8, 1,000,448 tokens, 1,954 steps, seed 1337, top_k=2,
`router_bias_update_rate=1e-4`, MTP disabled.

```text
n_dense_layers=0 (Gate M):          checkpoints/compact_v3_wikitext_moe_1m_gatem_topk2.pt
n_dense_layers=1, matched (Gate N): checkpoints/compact_v3_wikitext_moe_1m_gaten_dense1_matched.pt
```

## Results

| Metric | `n_dense_layers=0` (Gate M) | `n_dense_layers=1`, capacity-matched (Gate N) |
|---|---:|---:|
| Final validation perplexity | 580.62 | 581.57 |
| Stored parameters | 15,411,968 | **14,821,120** (3.8% fewer) |
| Active FFN compute (dense layer / MoE layer) | n/a | matched: 1152 hidden units both |
| Generation tokens/sec | 67.32 | 76.38 |
| Generation peak VRAM | 587.21 MB | 566.35 MB |

Load entropy for the three remaining MoE layers converges to 0.99+ by step 1500-1750, matching Gate M's pattern.

## Interpretation

Once compute-matched, the dense-layer prefix gives essentially the same validation perplexity as the
all-MoE baseline (581.57 vs 580.62, within run-to-run noise already observed across this project's other
sweeps) while using 3.8% fewer stored parameters and generating faster. This matches DeepSeek-V3's own
efficiency rationale for the dense-layer prefix: a size-matched dense layer is cheaper to store than a full
routed layer at equal active compute, with no quality cost at this scale. `n_dense_layers=1` is adopted as the
new default.

The initial confounded attempt is a useful lesson for the rest of this roadmap: several of V3's design choices
(this one, and `route_scale` next) come with specific companion ratios/values, not just an on/off mechanism —
porting the mechanism without the matching ratio produces a misleading comparison.

## Next gate

Gate O: retune `route_scale` (still the untouched default `1.0`; V3 uses `2.5` for top-8-of-256, a ratio that
doesn't directly transfer to our top-2-of-4 pool) as the next single-variable step. Gates P (MTP weight
annealing) and Q (MLA weight absorption) remain queued after that, per the roadmap agreed after Gate M.
