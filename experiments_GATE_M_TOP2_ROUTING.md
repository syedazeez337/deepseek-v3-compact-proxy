# Gate M — Top-2 routing, gated on Gate L's stable load balancing

## Status

Completed on 2026-08-04. Default `top_k` changed from `1` to `2`.

## Research basis

Gate J (before load-balancing was fixed) explicitly deferred this: "the result is too small to justify top-2
routing yet." Gate K/L then fixed load balancing. The README has anticipated this step since Gate B: "tests also
verify top-2 behavior before it is used in training." Before running it, the top-k literature was reviewed:

- GShard and Mixtral use top-2 routing; Switch Transformer deliberately uses top-1 to minimize per-token compute
  and routing complexity. The tradeoff is explicit in the literature: top-1 minimizes compute and communication,
  top-k>1 aggregates multiple expert perspectives for better per-token quality at higher compute cost.
- A general MoE survey/review source notes smaller expert counts require more careful load-balancing (matching
  Gate L's own finding) and that naive top-k selection without balancing causes expert collapse — reinforcing
  why this gate only runs after Gate L, not before.
- Caveat noted for interpretation: DeepSeekMoE's actual design principle is *fine-grained segmentation* — many
  small experts (64+ at DeepSeekMoE, 256 at V3) with low top-k density (top-8 of 256 ≈ 3%). This project's 4
  routed experts at top-2 is 50% density, architecturally closer to Mixtral's top-2-of-8 (25%) than to
  DeepSeekMoE/V3's actual sparsity regime. This experiment tests "does raising k on our existing small expert
  pool help," not "does DeepSeekMoE's fine-grained segmentation help" — that remains separate future work.

Sources:

- https://arxiv.org/html/2401.06066v1 (DeepSeekMoE: Towards Ultimate Expert Specialization)
- Mixtral/GShard top-2 vs Switch Transformer top-1 tradeoff (multiple corroborating summaries)
- https://normaluhr.github.io/2025/01/15/moe-load-balancing/ (small-expert-count balancing caution, reused from
  Gate L)

## Implementation

- `v3_cli.py`: added `--top-k` (default `2` after this gate), wired into `CompactV3Config` in both branches;
  `make_config()` takes `top_k` as a parameter.
- `v3_config.py`: default `top_k` changed from `1` to `2`.
- Added `test_top_k_flag_wires_into_config` (`tests/test_v3_cli.py`). Full active suite: 50 passed.

## Experiment

Single controlled variable (`top_k`: 1 -> 2) against the Gate L `router_bias_update_rate=1e-4` baseline. Everything
else identical: dataset/tokenizer, context 64, batch 8, 1,000,448 tokens, 1,954 steps, seed 1337, MTP disabled.

```text
top-1 baseline: checkpoints/compact_v3_wikitext_moe_1m_gatel_u00001.pt  (reused from Gate L, same config otherwise)
top-2 variant:  checkpoints/compact_v3_wikitext_moe_1m_gatem_topk2.pt
```

Parameter check (`CompactV3Model.parameter_report()`), confirming top-k only changes active compute, not stored
weights:

```text
top_k=1: unique_parameters=15,411,968  active_moe_parameters_per_layer_sum=2,363,392
top_k=2: unique_parameters=15,411,968  active_moe_parameters_per_layer_sum=3,543,040
```

## Results

| Metric | top-1 (Gate L) | top-2 (Gate M) |
|---|---:|---:|
| Final validation loss | 6.391349 | 6.364097 |
| Final validation perplexity | 596.661 | **580.620** |
| Min entropy any layer/checkpoint | 0.763 | 0.677 (step 250 only; ≥0.94 from step 750 onward, ≥0.996 by step 1500) |
| Generation tokens/sec | 77.92 | 67.32 |
| Generation peak VRAM | 587.16 MB | 587.21 MB |
| Stored parameters | 15,411,968 | 15,411,968 (unchanged) |
| Active MoE parameters/layer | 2,363,392 | 3,543,040 |

Top-2 load entropy converges to near-perfect uniformity by the second half of training (0.996-1.000 across all
four layers at steps 1500-1750), better than top-1's already-good result from Gate L.

## Interpretation

Top-2 routing improved validation perplexity by 16.0 points (2.7%) over the top-1/tuned-bias-rate baseline, with
zero additional stored parameters — a pure compute-for-quality trade, exactly the tradeoff the literature describes
for Mixtral/GShard's choice of top-2 over Switch Transformer's top-1. This is roughly 10x the perplexity gain Gate
J originally measured for adding MoE over dense (0.27%), at no parameter cost, because it was measured only after
Gate L fixed load balancing — Gate J's own top-1 MoE run had unstable routing that this comparison avoids by
construction. The cost is real: generation throughput dropped ~14% (77.92 -> 67.32 tokens/sec) since decode now
runs two experts per token instead of one. Load balance also improved further under top-2, consistent with the
literature note that spreading more selections per token naturally eases concentration.

This result should not be read as "DeepSeekMoE's architecture benefits from top-2" — it is a narrower finding
specific to this compact 4-expert pool. DeepSeekMoE/V3's actual design keeps top-k density low (~3%) by using far
more, smaller experts, not by raising k on a small pool.

## Next gate

Two directions are now open, both single-variable relative to this gate's config (`top_k=2`,
`router_bias_update_rate=1e-4`):

1. Sweep `sequence_balance_coefficient` (still untouched since the project's original default, `1e-4`) the same
   way Gate L swept the bias rate.
2. Test DeepSeekMoE-style fine-grained segmentation as a distinct architectural change: more, smaller routed
   experts (e.g. 8 experts at half `expert_hidden_dim`, top-4) holding active-parameter budget roughly constant,
   to test the segmentation hypothesis directly rather than conflating it with a plain top-k increase.

Weight-absorbed/fused MLA decode kernels remain separately deferred future work (mentioned in Gates A and E).
