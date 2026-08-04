# Gate L — Router bias update rate sweep, grounded in published MoE load-balancing research

## Status

Completed on 2026-08-04. Default `router_bias_update_rate` changed from `1e-3` to `1e-4`.

## Research basis

Before choosing values to test, the DeepSeek-V3/DeepSeekMoE load-balancing literature was reviewed:

- The bias mechanism this project already implements (`LoadBalancer.update()` in `routing.py`: `bias += rate *
  sign(load - target)`) is exactly DeepSeek's published "Loss-Free Balancing" rule,
  `b_i = b_i + u * sign(e_i)` — no architectural change was needed, only the rate.
- DeepSeek-V3 used bias update speed `γ=0.001` for the first 14.3T of 14.8T training tokens, then annealed to 0.
- The Loss-Free Balancing paper's own ablation (64 experts, large batches) found `u=0.0001` too slow to converge,
  `u=0.001` the tuned sweet spot, and `u=0.01` caused gating to oscillate/thrash.
- Multiple independent sources (a MoE load-balancing survey, Skywork-MoE's writeup) repeat the same warning: too
  strong a correction signal causes the gate to thrash rather than settle, hurting rather than helping balance.
- This project's own config already used `router_bias_update_rate=1e-3` — the literature's large-scale sweet
  spot — yet Gate K showed persistent, unrecovered imbalance in 3 of 4 layers over a 1,954-step run.

Sources:

- https://arxiv.org/pdf/2412.19437 (DeepSeek-V3 Technical Report)
- https://arxiv.org/abs/2408.15664 / https://ar5iv.labs.arxiv.org/html/2408.15664 (Auxiliary-Loss-Free Load
  Balancing Strategy for Mixture-of-Experts)
- https://normaluhr.github.io/2025/01/15/moe-load-balancing/ (review of load-balancing pitfalls across GShard,
  Switch Transformer, DeepSeekMoE, DeepSeek-V3)

The literature's tuned value comes from a very different regime (64 experts, large token batches, hundreds of
billions of training tokens) than this project's compact scale (4 experts, 512 tokens/optimizer step, ~2K steps
total). Rather than assume the number transfers, or blindly guess a new one, this gate treats `u=0.001` as a prior
and runs a small bracketed sweep — the same three points the original paper itself tested (`0.0001`, `0.001`,
`0.01`), plus the Gate K baseline — since a full 1M-token run costs only ~3 minutes on this hardware.

## Implementation

- `v3_cli.py`: added `--bias-update-rate` (default `1e-4` after this gate), wired into `CompactV3Config` in both
  the real-corpus and synthetic branches; `make_config()` takes the rate as a parameter.
- `v3_config.py`: default `router_bias_update_rate` changed from `1e-3` to `1e-4` based on this gate's result.
- Added `test_bias_update_rate_flag_wires_into_config` (`tests/test_v3_cli.py`). Full active suite: 49 passed.

## Experiment

Four runs, otherwise identical to Gate K/J (dataset, tokenizer, context 64, batch 8, 1,000,448 tokens, 1,954 steps,
seed 1337, MTP disabled, top-1 routing over 4 experts + 1 shared expert):

| Rate `u` | Checkpoint | Note |
|---|---|---|
| 0.001 | `checkpoints/compact_v3_wikitext_moe_1m_gatek.pt` | Gate K baseline (current default at the time) |
| 0.003 | `checkpoints/compact_v3_wikitext_moe_1m_gatel_u003.pt` | first step up |
| 0.01 | `checkpoints/compact_v3_wikitext_moe_1m_gatel_u01.pt` | paper's flagged oscillation risk |
| 0.0001 | `checkpoints/compact_v3_wikitext_moe_1m_gatel_u00001.pt` | paper's flagged "too slow" |

## Results

Minimum per-layer entropy observed at any of the 7 periodic checkpoints (step 250-1750), and final metrics:

| Rate `u` | Min entropy (any layer/checkpoint) | Collapse events (entropy < 0.5) | Final val loss | Final val PPL |
|---:|---:|---:|---:|---:|
| 0.0001 | 0.763 | 0 | 6.391349 | 596.661 |
| 0.001 (Gate K) | 0.320 | 1 (layer 1 @750) | 6.391330 | 596.649 |
| 0.003 | 0.051 | 5 across layers 1-3 | 6.395736 | 599.284 |
| 0.01 | **0.000** (full collapse, layer 2 @1750, all 512 tokens to one expert) | 6 across layers 1-3 | 6.395537 | 599.165 |

At `u=0.0001`, no layer ever dropped below entropy 0.76 at any checkpoint — every layer stayed close to uniform
for the entire run. At `u=0.01`, layer 2 fully collapsed onto a single expert by the final logged checkpoint.

## Interpretation

The result is a clean, monotonic trend in the opposite direction from the naive "increase the correction signal to
fix imbalance" instinct: **lower** bias update rate produced **more stable** load balance at this scale, with no
perplexity cost (596.6 vs 596.6-599.3 across the sweep — the higher rates were both worse on quality and on
balance). This directly contradicts the source paper's own finding that `u=0.0001` was "too slow" — but that
finding was tuned for 64 experts and vastly larger per-step batches. At 4 experts and 512 tokens/step, each
optimizer step's expert-load count is a noisier statistic (average 128 tokens/expert), so a bias correction sized
for a low-noise large-scale regime overreacts to noise here rather than tracking a genuine signal, producing
exactly the thrashing failure mode the literature warns about — just at a different absolute rate than where it
was originally observed. This is the concrete lesson: a published hyperparameter is a prior for a different scale,
not a constant, and cheap local ablation beats blind transfer.

`router_bias_update_rate=1e-4` is now the project default. This does not retroactively invalidate Gates I/J/K,
whose configurations remain fully recorded; it changes what "current best config" means for gates from here on.

## Validation

```text
Bias-rate CLI test:   1 passed
Active V3 suite:      49 passed
Compilation:          passed
Four 1M-token CUDA sweep runs: completed, checkpoints preserved
```

## Next gate

With load entropy now stable and near-uniform at the default rate, top-2 routing or additional experts can be
considered as the next single-variable change, holding `router_bias_update_rate=1e-4` fixed. A useful complementary
follow-up (not yet run) would sweep `sequence_balance_coefficient` the same way, since it was left untouched here
to keep this gate a single-variable experiment.
