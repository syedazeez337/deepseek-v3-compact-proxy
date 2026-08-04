# Gate K — MoE load-balancing measurement over a full training run

## Status

Completed on 2026-08-04.

## Motivation

Gate J's routing inspection used a single deterministic 64-token batch taken after training finished. It showed
uneven per-layer expert loads (one expert entirely unused in one layer) but could not distinguish transient noise
from a persistent imbalance, and gave no view of how load balancing behaved over the course of training. Gate J's
interpretation named this the next required step: measure bias-update and sequence-balance behavior and
expert-load entropy over a longer run before increasing active experts or top-k.

## Implementation

- `routing.py`: added `RoutingResult.load_entropy()` — normalized Shannon entropy of the per-expert load
  distribution (`-sum(p*log p) / log(n_experts)`), 1.0 = perfectly uniform load, 0.0 = fully collapsed onto one
  expert. Included in `RoutingResult.summary()`.
- `v3_cli.py`: `routing_report()` was dead code (defined, never called) and was gated on `corpus is not None`
  rather than on whether the model actually uses MoE. Fixed the condition to `model_config.use_moe` and wired it
  into `save_progress()` so every periodic checkpoint now logs `expert_loads` and `load_entropy_by_layer` alongside
  loss/perplexity, not just a single post-hoc snapshot.
- Added `test_load_entropy_uniform_and_concentrated` (`tests/test_routing.py`) and
  `test_cli_progress_reports_routing_entropy` (`tests/test_v3_cli.py`). Full active suite: 48 passed.

This is a pure diagnostic addition — no training math changed.

## Experiment

Exact rerun of Gate J's matched 1M-token MoE configuration, with the new periodic entropy logging enabled. Written
to a new checkpoint path so Gate J's original artifact is preserved.

```text
Dataset:             Salesforce/wikitext, wikitext-2-raw-v1 (same provenance hashes as Gates H/I/J)
Model:                4 routed experts + 1 shared expert, top-1
MTP:                  disabled
Context:              64 tokens
Batch size:           8 sequences
Tokens/optimizer:     512
Train tokens:         1,000,448
Optimizer steps:      1,954
Checkpoint interval:  250 steps
Seed:                 1337 (identical policy to Gates I/J)
Precision:            FP16 autocast + GradScaler
Checkpoint:           checkpoints/compact_v3_wikitext_moe_1m_gatek.pt
Wall clock:           ~3 minutes (RTX 3050 Laptop 6GB, corpus/tokenizer already cached)
```

Reproducibility check: final validation perplexity **596.649** versus Gate J's **596.65** on the identical setup —
matches, confirming the added diagnostics did not alter training behavior.

## Per-layer load entropy over training

Normalized entropy (1.0 = uniform across 4 experts, 0.0 = collapsed onto one):

| Step | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---:|---:|---:|---:|---:|
| 250 | 0.977 | 0.989 | 0.988 | 0.907 |
| 500 | 0.984 | 0.901 | 0.982 | 0.687 |
| 750 | 0.988 | **0.320** | 0.685 | 0.824 |
| 1000 | 0.938 | 0.835 | 0.738 | 0.728 |
| 1250 | 0.977 | 0.845 | 0.733 | 0.748 |
| 1500 | 0.958 | 0.763 | 0.741 | 0.859 |
| 1750 | 0.990 | 0.826 | 0.712 | 0.763 |

At step 750, layer 1's expert loads were `[19, 5, 456, 32]` out of 512 assignments — one expert absorbed 73% of
tokens in that layer for that checkpoint window.

## Interpretation

Layer 0 stays near-uniform (entropy 0.94-0.99) throughout — the bias/sequence-balance mechanism works there.
Layers 1-3 do not: layer 1 shows a sharp collapse at step 750 followed by only partial recovery (settling around
0.76-0.85, never back to its starting ~0.99); layer 2 drops after step 500 and stays persistently in the 0.68-0.74
band for the rest of training, i.e. a structural imbalance rather than transient noise; layer 3 is noisy in the
0.69-0.86 range without a clear trend. None of the imbalanced layers recover to their early-training entropy by
step 1750.

This confirms and extends Gate J's single-snapshot finding: the current load-balancing settings
(`router_bias_update_rate=1e-3`, `sequence_balance_coefficient=1e-4`) are not sufficient to maintain uniform
routing across all layers over a 1,954-step run, and the failure mode is layer-specific rather than uniform across
the model.

## Next gate

Test whether strengthening the load-balancing signal fixes this as a single controlled variable: rerun with an
increased `router_bias_update_rate` and/or `sequence_balance_coefficient`, holding dataset, tokenizer, context,
batch, schedule, and seed identical, and compare the same per-checkpoint entropy trajectory plus final validation
perplexity against this Gate K baseline. Only after load entropy is shown to stay consistently high should top-2
routing or additional experts be considered.
