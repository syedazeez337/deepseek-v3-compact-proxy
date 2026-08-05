# Gate J — 1M-token dense versus 4-expert top-1 MoE comparison

## Status

Completed on 2026-08-04.

## Controlled variables

Both runs used:

```text
Dataset:             Salesforce/wikitext, wikitext-2-raw-v1
Tokenizer:           identical train-only 32K BPE
Train tokens:        1,000,448
Context length:      64
Batch size:          8
Tokens/optimizer:    512
Optimizer:           AdamW
Learning rate:       3e-4 peak with same warmup/cosine schedule
Precision:           FP16 autocast + GradScaler
Seed policy:         identical
MTP:                 disabled
MLA:                 identical
RMSNorm:             identical
```

Only the FFN path changed:

```text
Dense control: one dense SwiGLU path per block
MoE run:       one shared SwiGLU expert + four routed experts, top-1
```

## Results

| Metric | Dense control | 4-expert top-1 MoE |
|---|---:|---:|
| Unique parameters | 10,689,280 | 15,411,968 |
| Final train loss | 6.6613 | 6.6877 |
| Final validation loss | 6.3940 | 6.3913 |
| Validation perplexity | 598.27 | 596.65 |
| Generated tokens | 32 | 32 |
| Generation peak VRAM | 400.92 MB | 581.25 MB |
| MTP | disabled | disabled |

The MoE validation improvement is approximately 1.62 perplexity points, or 0.27%, on this small single-corpus comparison. This is a measured ablation result, not evidence that MoE is generally superior at this scale.

## Routing observations

A deterministic inspection batch of 64 tokens produced these routed loads by layer:

```text
layer 0: [15, 15, 13, 21]
layer 1: [24, 19, 21,  0]
layer 2: [23, 37,  3,  1]
layer 3: [ 2,  4, 29, 29]
```

The routing bias mechanism has not produced uniform per-layer loads after this short run. One layer left an expert unused, while other layers specialized heavily. This is an important negative/diagnostic result, not a failure to hide.

## Validation

```text
Matched MoE tests:    17 passed
Full active suite:    46 passed before experiment
MoE 1M-token run:     completed at 1,000,448 tokens
Periodic checkpoints: passed
Final checkpoint load: passed
Final validation:     passed
Cached generation:    passed
```

## Interpretation

At equal token budget, the compact MoE proxy is marginally better on validation perplexity but costs approximately 44% more stored parameters and higher generation VRAM. The result is too small to justify top-2 routing yet. The next research phase should improve and measure load balancing—bias updates, sequence safeguard, and longer-run expert-load entropy—before increasing active experts.

The failed evaluation command used the archived directory and was corrected by running from the active clean project root; no experiment state was changed by that command.
