# Gate V: WikiText-103 shakedown, and the throughput work that made it affordable

A one-hour run on WikiText-103 to confirm the whole stack works end to end before committing a multi-hour
budget. It also carries the throughput work that preceded it, including a measurement error that had made every
previous estimate in this area wrong.

## The measurement error, first

Three separate probes in this session estimated training throughput and VRAM. **None of them constructed an
optimizer.** They built a model, ran forward and backward, and called `zero_grad`. So all of them omitted AdamW's
two fp32 moment buffers (1184.6 MiB at 155M parameters) and never timed `optimizer.step()`.

Re-measured through this project's own `train_steps` with a real optimizer, at the batch 8 / sequence 512 that
had actually been launched:

| | ms/step | tok/s | peak reserved |
|---|---|---|---|
| manual attention (the launched configuration) | 1232.18 | 3,324 | 7158 MiB |
| fused attention | 721.79 | 5,675 | 6608 MiB |

Both reserve more than the 6144 MiB card. `nvidia-smi` sat pinned at 5991 MiB throughout, which is the signature
of Gate O's Windows failure mode: no OOM, silent spill into shared system memory, large slowdown.

The run that had been launched on a projected 18-hour budget was actually on course for roughly **60 hours**.

This is the second time in two gates that a benchmark omitting real training state produced a confidently wrong
number (Gate T's decode benchmark reported the wrong sign). The rule that follows: **benchmark the real training
step through the real code path, never a hand-written approximation of it.**

## Throughput work

### Fused attention

`reference()` and `prefill()` computed attention explicitly: two matmuls, a causal `masked_fill`, and a float32
softmax, materialising a `[batch, heads, T, T]` tensor per layer and holding it for backward.

Decoupled RoPE makes the fused form exact rather than approximate. The score is

```text
q_content . k_content + q_rope . k_rope
```

which is precisely the dot product of the concatenated `[content; rope]` vectors, and SDPA's default `1/sqrt(E)`
scale already equals `1/sqrt(qk_head_dim)`. This is the property MLA's split exists to provide, and DeepSeek's
own implementation concatenates for the same reason.

| | time | peak |
|---|---|---|
| one MLA layer, manual | 16.34 ms | 349.6 MiB |
| one MLA layer, fused | **5.26 ms** | **120.7 MiB** |
| | 3.10x | 2.90x |

Equivalence: 2.98e-07 in fp32, 4.88e-04 under fp16 autocast, gradients matching to rtol 1e-4.
`reference_manual()` retains the explicit form as the proof, following the `decode_naive()` and
`forward_reference()` precedents.

### Batch size, re-derived against real memory

| batch | tok/s | peak reserved | fits 6144 MiB |
|---|---|---|---|
| 8 (non-fused AdamW) | 5,615 | 6610 MiB | no, spilling |
| 6 | 8,549 | 5178 MiB | yes |
| **7** | **9,400** | **5676 MiB** | yes, 468 MiB spare |
| 8 (fused AdamW) | 9,336 | 6136 MiB | yes, 8 MiB spare |

Batch 8 with fused AdamW does fit, but 8 MiB of headroom on a multi-hour run is not worth 0.7% throughput.

### Fused AdamW

`fused=True` gives 1.16x at identical memory and, by avoiding intermediate tensors, saves enough to raise the
batch from 6 to 7. Guarded to CUDA float parameters with a silent fallback.

### Alternative optimizers, measured and rejected

AdamW state is 1184.6 MiB, 20% of the card, so lighter optimizers were evaluated on the real training step.

| optimizer | state | tok/s at batch 6 |
|---|---|---|
| **AdamW (fused)** | 1184.6 MiB | **8,549** |
| SGD + momentum | 592.3 MiB | 8,138 |
| Adafactor | 2.3 MiB | 5,237 |
| Muon (hybrid, 2D weights) | 654.9 MiB | 4,454 |

`torch.optim.Muon` is native in torch 2.13 and is the current favourite in the literature (~2x compute
efficiency, adopted by Kimi K2, GLM-4.5, INTELLECT-3). Here it is **48% slower per step**: Newton-Schulz
orthogonalisation runs five matrix iterations per 2D parameter, and this model has 96 small expert matrices per
MoE layer, so per-matrix overhead dominates. Muon's advantage is convergence per step on large matrices.

Adafactor's state is effectively free (2.3 MiB, factored second moments) but it is 39% slower per step.

Crucially, none of them help, because **memory stopped being the constraint** once fused AdamW fit batch 7. Even
where a lighter optimizer let batch 8 fit, it was still slower than fused AdamW at batch 6.

Caveat recorded honestly: this measured **wall-clock per step, not convergence per step**. Muon's claim is
reaching a given loss in fewer steps. Testing that means two full training runs and a loss-curve comparison,
which is a gate of its own. AdamW also remains what DeepSeek-V3 actually uses.

Cumulative: **3,324 -> 9,400 tok/s, 2.83x** over the configuration that was actually running.

## The shakedown run

8,000 steps, batch 7, sequence 512, 28,672,000 tokens, WikiText-103 with its own cached tokenizer, MTP enabled,
200 warmup steps, `total_steps` sized to this run so the cosine schedule and MTP anneal complete properly.

| step | tokens | main | MTP | ratio | val PPL | grad norm | min entropy |
|---|---|---|---|---|---|---|---|
| 1000 | 3,584,000 | 5.4803 | 5.6244 | 1.026 | 270.99 | 1.016 | 0.755 |
| 2000 | 7,168,000 | 5.2039 | 5.3675 | 1.031 | 181.39 | 1.166 | 0.813 |
| 3000 | 10,752,000 | 5.0239 | 5.1465 | 1.024 | 147.66 | 1.241 | 0.891 |
| 4000 | 14,336,000 | 4.8195 | 4.9999 | 1.037 | 123.22 | 1.336 | 0.955 |
| 5000 | 17,920,000 | 4.7872 | 5.0074 | 1.046 | 108.29 | 1.524 | 0.943 |
| 6000 | 21,504,000 | 4.9306 | 5.1070 | 1.036 | 97.89 | 1.374 | 0.944 |
| 7000 | 25,088,000 | 4.5454 | 4.8168 | 1.060 | 91.56 | 1.441 | 0.964 |
| 8000 | 28,672,000 | 4.7965 | 5.0634 | 1.056 | **87.85** | 1.516 | 0.958 |

### What each signal confirms

**Perplexity fell at every checkpoint**, 270.99 -> 87.85, with no plateau or divergence. Deterministic
evaluation (68 batches, all 470 windows), so the curve is signal rather than sampling noise.

**The MTP ratio held between 1.024 and 1.060 for the entire run.** This is the load-bearing result. Before Gate
U the same quantity was 0.004, because the head was reading its target off its own input. A value slightly above
1.0 is exactly what a working second-token predictor should produce, since predicting `t+2` is strictly harder
than `t+1`. Gate U's fix is confirmed at full scale under real training, which makes MTP speculative decoding
viable.

**Routing entropy improved rather than collapsed**, minimum across the five MoE layers rising 0.755 -> 0.958.
Gate K observed a layer collapsing to 0.32 on a shorter run; that did not recur at 8,000 steps.

**Grad norm stayed between 1.0 and 1.5** with no spike, vindicating the 200-step warmup. The previous hardcoded
`min(2, steps-1)` would have been two steps.

**Resume works on a real checkpoint**: restarted from step 8000, trained 100 further steps, perplexity 87.85 ->
87.57. The 1.86 GB checkpoint wrote atomically and reloaded clean.

### Generation

Greedy decoding still loops, which is normal at this token budget:

> The cat sat on **the ground . The first two were the first to be built in the early 20th century , and the
> first was the first to be built in the late 19th century .**

Sampled at temperature 0.8, top-k 40:

> In 1943 , the government announced **the formation of a state @-@ owned district in October 1943 to reduce
> health and treatment of the state . The government approved the Act 's government to remove the law by the
> federal government .**

The model carried **1943 from the prompt into the continuation**, which is context tracking rather than fluent
noise. Compare the pre-Gate-V best (292.54 PPL): *"The cat sat on the coast of the United States . The storm was
also used as a tropical storm ."* Coherence still decays after a sentence or two.

`@-@` is WikiText's own hyphen encoding, not a tokenizer defect.

## Interpretation

The stack works end to end: train, checkpoint atomically, resume, evaluate deterministically, generate readable
text. Nothing in the pipeline needed fixing during the run, which is the point of a shakedown.

The perplexity figure is **not comparable to the previous best of 292.54**. That was WikiText-2 validation with a
WikiText-2 tokenizer; this is WikiText-103 validation with its own. Different denominator. The comparable claims
are the shape of the curve and the readability of the samples.

Effective throughput including checkpoint overhead was ~8,000 tok/s, but eight checkpoints in one hour is dense.
At a realistic checkpoint interval the overhead is ~3%, so ~9,200 tok/s is the planning figure.

| Budget | Epochs | Estimated time | Extrapolated PPL |
|---|---|---|---|
| 115M | 1.0 | ~3.5 h | ~60-70 |
| 230M | 2.0 | ~6.9 h | ~45-60 |
| 346M | 3.0 | ~10.4 h | ~40-52 |
| 461M | 4.0 | ~13.9 h | ~38-48 |

Those perplexities apply Chinchilla's `D^-0.28` data exponent to the measured 87.85. The slope observed within
this run was steeper (-0.54), but that is inflated by LR annealing inside a short schedule and will not hold.
They are a range, not a forecast.

Four epochs is the point where [repeated data stops paying full value](https://arxiv.org/pdf/2305.16264) on a
115M-token corpus. Beyond that the budget is better spent elsewhere.

## Changes

- `src/compact_v3/mla.py`: fused `reference()` and `prefill()`; `reference_manual()` retained as the proof.
- `src/compact_v3/training.py`: `make_optimizer` uses fused AdamW where available.
- `v3_cli.py`: `--warmup-steps` added; `--batch-size` default 8 -> 7.
- `tests/`: fused-attention equivalence (output, gradients, causality) and optimizer-selection tests. Suite
  82 -> 87.

Checkpoint: `checkpoints/compact_v3_wikitext103_shakedown.pt` (1.86 GB, step 8000, validation perplexity 87.85).

## Next gate

Gate W: the full WikiText-103 run at 2 epochs (230M tokens, ~6.9 h), which is where repeated data still pays
nearly full value and roughly halves perplexity again.

Then MTP speculative decoding, now that the draft head demonstrably predicts. V3 reports 85-90% acceptance and
1.8x TPS; a 155M model's draft will be weaker, and measuring how much weaker is the experiment.
