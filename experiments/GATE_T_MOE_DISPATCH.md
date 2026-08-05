# Gate T: batched MoE dispatch

Flagged in Gate O and again in Gate R: `moe.py` dispatched routed experts in a Python loop whose cost grew
with expert count even though active compute per token does not. Gate O deferred it; this gate measures it,
fixes it, and records a result that contradicts the obvious optimisation.

## Specification

DeepSeekMoE computes, for each token `t` and each of its `top_k` selected experts `e`:

```text
y[t] = sum_shared E_s(x[t]) + sum_{e in topk(t)} w[t,e] * E_e(x[t])
```

The routed sum only requires that each expert sees the tokens routed to it. Nothing in the mathematics
requires a Python loop, per-expert boolean masks, or device-to-host transfers. The pre-gate implementation
used all three:

```python
for expert_id, expert in enumerate(self.routed_experts):
    token_slots = routing.selected_indices == expert_id
    token_mask = token_slots.any(dim=-1)
    if not token_mask.any():          # device -> host sync, every expert
        continue
    token_indices = token_mask.nonzero(as_tuple=False).squeeze(-1)   # sync
    ...
```

`if not token_mask.any()` and `.nonzero()` each force a synchronisation. At 32 experts across 5 MoE layers
that is up to 160 stalls per forward pass, plus a full-width mask comparison per expert over every token.

## What DeepSeek-V3 actually does

Fetched from the official `inference/model.py` rather than inferred from the paper:

```python
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx]) * weights[idx, top, None]
```

V3 also loops, but takes the counts to the host **once** via `.tolist()` and then branches on a plain Python
list. Their loop is additionally over *local* experts only (`experts_start_idx:experts_end_idx`) because 256
experts are sharded across ranks, so each rank iterates over a handful. This project holds all 32 experts on
one GPU, which is why the same structure costs proportionally far more here.

## Implementation

`DeepSeekMoE._dispatch_routed` groups the `(token, slot)` assignments by expert in a single sorted pass:

- flatten `selected_indices` / `selected_weights` to one entry per assignment
- `torch.argsort(assignments, stable=True)` puts every expert's assignments in one contiguous run
- the router **already** computed `expert_load` as a bincount over those same assignments, so the group
  boundaries are a running offset over `expert_load.tolist()` (one transfer, no second bincount)
- each non-empty expert gets one contiguous slice; empty experts are skipped from the host-side list

One device-to-host transfer per layer, down from up to one per expert. No mask allocation per expert.

The pre-gate implementation is retained verbatim as `DeepSeekMoE.forward_reference()` and used only by tests,
following the `decode_naive()` precedent set in Gate R.

## Rejected alternative: stacked weights + padded `bmm`

The textbook fix is to stack expert weights into `[E, d_model, hidden]` and replace the loop with three
`torch.bmm` calls after sorting and padding each expert's tokens to a common length. Prototyped and measured
before being rejected.

`torch._grouped_mm` exists in torch 2.13 and runs *forward* on this Ampere sm86 card in fp16, bf16 and fp32,
which was unexpected. Its backward fails in every layout tried (`Invalid strides/sizes, got [0, 0] for
strides`), including input-only and weight-only grad, so it is unusable for training without a hand-written
autograd Function. The padded `bmm` route was prototyped instead.

Padding cost is `O(n_experts * largest_expert_load)` rather than `O(total_assignments)`, so it degrades with
routing imbalance. Measured on 32 experts, 2048 tokens, sweeping skew (waste = `E * max_load / total`):

| skew | waste | load entropy | sorted loop | padded bmm | bmm / sorted | bmm peak |
|---|---|---|---|---|---|---|
| 0.00 | 1.2x | 1.00 | 17.17 ms | 9.32 ms | **0.54x** | 409 MiB |
| 0.25 | 8.7x | 0.89 | 17.60 ms | 34.39 ms | 1.95x | 833 MiB |
| 0.50 | 16.6x | 0.68 | 16.15 ms | 61.05 ms | 3.78x | 1271 MiB |
| 0.75 | 24.2x | 0.40 | 19.05 ms | 87.71 ms | 4.60x | 1701 MiB |
| 0.95 | 30.5x | 0.10 | 17.38 ms | 1230.79 ms | 70.8x | 2051 MiB |
| 1.00 | 32.0x | 0.00 | 4.78 ms | 2484.58 ms | 519x | 2135 MiB |

The batched version wins only in a narrow band near perfect balance and is catastrophic outside it: 2.1GB of
padding on a 6GB card, and 519x slower under full collapse. The sorted loop is flat across the entire range
and gets *faster* under collapse, because it does strictly less work when fewer experts are populated.

This is not a hypothetical risk. Gate K measured layer 1 collapsing to load entropy 0.32 in a real run, and
imbalance is worst early in training, exactly where a long run starts. The reliably-fast option is rejected
here in favour of the never-slow one.

## Measured result

RTX 3050 Laptop 6GB, batch 8, sequence 256, top_k=2. Median of 25 (layer) / 20 (model) iterations.

Single MoE layer, forward+backward, expert count varied at constant active compute:

| Routed experts | reference | sorted | speedup | peak MiB (ref / sorted) |
|---|---|---|---|---|
| 4 | 13.87 ms | 11.32 ms | 1.23x | 115.77 / 115.74 |
| 8 | 14.36 ms | 11.38 ms | 1.26x | 129.47 / 129.43 |
| 16 | 24.04 ms | 14.09 ms | 1.71x | 149.00 / 148.96 |
| **32 (default)** | 40.13 ms | 26.52 ms | **1.51x** | 245.16 / 245.16 |
| 64 | 79.87 ms | 44.74 ms | 1.79x | 437.97 / 437.97 |

The speedup grows with expert count, as expected when the removed cost is per-expert.

Full Configuration B training step, FP16 autocast + GradScaler, three repetitions in **fresh processes**:

| | median ms/step | tok/s | peak MiB |
|---|---|---|---|
| reference | 360.39 | 5,683 | 2752.16 |
| **sorted** | **286.70** | **7,143** | 2751.90 |
| | **1.257x** | **+25.7%** | unchanged |

Memory is unchanged (marginally lower). The gain is pure removal of stalls and mask allocations.

Cached generation (batch 1, 16-token prompt, 128 new tokens, 5 repetitions per process, interleaved):

| | tok/s run 1 | tok/s run 2 |
|---|---|---|
| reference | 31.72 | 32.10 |
| **sorted** | **60.94** | **58.98** |

Decode gains **~1.9x**, more than training does. That is the expected direction once the mechanism is clear:
a decode step processes a single token, so the per-expert synchronisations are almost the entire cost of the
MoE layer, while in training they are amortised over 2048 tokens of real matrix work. Confirmed independently
on the Gate Q checkpoint through `complete.py`: 19.22 tok/s before this gate, 23.26 tok/s after, on identical
output text.

### Two methodology corrections inside this gate

**Single-process contamination.** The first full-model benchmark ran both configurations in one process and
produced 1.30x on one invocation and 0.70x on the next: a 2x swing on identical code. Peak allocation had
reached 3.9GB on a 6GB card. This is the failure mode Gate O documented (Windows silently spills past the
VRAM limit into shared system memory rather than raising OOM) compounded by the one Gate R fixed (model
instances stacking unfreed in a single process). Re-run with one configuration per process, peak dropped to
2752MiB and three repetitions agreed to within 2%. The numbers above are from the fresh-process runs; the
single-process numbers were discarded.

**Insufficient warmup on the decode benchmark.** The first decode measurement used 3 repetitions after an
8-token warmup and reported the sorted path as 2.3x *slower* (13.09 vs 29.86 tok/s). That contradicted the
`complete.py` observation on a real checkpoint, which is what prompted re-measurement rather than acceptance.
At 5 repetitions with interleaved ordering the result reversed cleanly and reproducibly to 1.9x faster, with
no overlap between the two sets of samples. The 3-repetition run was measurement error, not a real effect.
Both corrections point the same way: on this hardware a single short measurement is not evidence.

## Equivalence

Six tests added to `tests/test_moe_v3.py`:

- `test_sorted_dispatch_matches_reference[1|2|3]` — bit-exact output, balance loss and routing at top_k 1, 2, 3
  (`torch.equal`, not a tolerance)
- `test_sorted_dispatch_matches_reference_gradients` — every parameter gradient bit-exact after backward
  (`rtol=0, atol=0`)
- `test_sorted_dispatch_matches_reference_on_cuda` — CUDA `index_add_` accumulates with atomics, so this one
  allows `rtol=1e-5, atol=1e-6`
- `test_dispatch_handles_expert_receiving_no_tokens` — router forced to collapse onto a single expert, so 31
  of 32 groups are empty and must be skipped

Full suite: 72 passed (66 before this gate).

## Interpretation

The dispatch loop was never the real cost; the synchronisations inside it were. Removing them recovers 25.7%
end-to-end training throughput and ~1.9x decode throughput for zero memory cost, zero parameter change, and
bit-exact outputs on CPU. The split between the two figures is itself the evidence for that reading: the win
is largest exactly where there is least real work to hide the stalls behind.

The more useful finding is the negative one. The standard advice for MoE dispatch is to batch experts into a
grouped GEMM, and at this scale that advice is actively wrong: padding waste scales with load imbalance, and
this model's imbalance is both real and worst exactly when a run begins. The structure that wins is the one
DeepSeek-V3 already ships, for a reason their code does not state and their scale hides: a sorted loop costs
`O(assignments)` regardless of how skewed the routing is.

This is the third gate (after L and P) where a value or technique taken from the large-scale setting measured
worse than a simpler alternative at this scale.

Caveat: `torch._grouped_mm` having a working forward and a broken backward on sm86 is a torch 2.13
observation, not a permanent property. If a future version fixes the backward, the padded-`bmm` memory
objection still stands but the compute objection would need re-measuring, since a true grouped GEMM does not
pad.

## Changes

- `src/compact_v3/moe.py`: `forward()` now delegates routed dispatch to `_dispatch_routed()`; the previous
  implementation is retained as `forward_reference()` for the equivalence proof.
- `tests/test_moe_v3.py`: six new tests as listed above.

No configuration defaults changed. No checkpoint format change; existing checkpoints load and generate
unchanged.

## Next gate

Gate U: the WikiText-103 training run. `data_v3_103/` already holds 115,241,113 train tokens (1,165,029
documents, 32K BPE trained on the train split only, SHA256 recorded), and the CLI gained
`--dataset-config`/`--dataset-cache-dir` for it. At the throughput measured here:

| Budget | before Gate T | after Gate T |
|---|---|---|
| One epoch of WikiText-103 (115.2M tokens) | 5.63 h | **4.48 h** |
| Chinchilla-optimal on ~36M active params (720M tokens) | 35.19 h | **28.00 h** |

Two decisions are open before that run: the token budget (one epoch as a first datapoint, or push toward the
Chinchilla point), and whether to reuse the WikiText-2 tokenizer for direct comparability with Gates I-Q or
use the better-fitted WikiText-103 tokenizer already cached.

Note for any perplexity comparison against published WikiText-103 results: those are word-level over a ~267K
vocab, this project reports subword perplexity over 32K BPE, and the two are not comparable. The validation
split is 241,580 subword tokens against roughly 218K words, so word-level PPL is approximately
`subword_PPL ** 1.11`. Matching the ~29.4 word-level figure reported for a comparable 6-layer 156M-parameter
decoder means reaching roughly 21 subword PPL. That exponent should be recomputed against an exact word count
before it is used to claim anything.
