# Gate W: the WikiText-103 run, 2 epochs

The first training run at a serious token budget. 229,376,000 tokens, 1.99 epochs of WikiText-103, on the stack
assembled in Gates T through V.

## Setup

```text
64,000 steps, batch 7, sequence 512, 3,584 tokens/step
warmup 1,000, cosine to 10% of peak, AdamW (fused) lr 3e-4, betas (0.9, 0.95), wd 0.1, grad clip 1.0
Configuration B: d_model 512, 6 layers (1 dense + 5 MoE), 32 routed + 1 shared experts, top_k 2
context 520, route_scale 0.75, router_bias_update_rate 1e-4
MTP depth 1, weight 0.3 annealed to 0.1 at 67.6% of steps
WikiText-103 with its own train-split tokenizer (32K byte-level BPE, SHA256 be3accdb...089078)
```

155,271,168 total parameters, ~36M active per token. Checkpoint
`checkpoints/compact_v3_wikitext103_2ep.pt`, 1777.7 MiB.

## Result

| | |
|---|---|
| **validation perplexity** | **41.35** |
| validation loss | 3.7222 |
| evaluation | deterministic, 68 batches, all 470 windows |
| training loss (final) | 3.4381 |
| MTP loss (final) | 3.9634 |
| **MTP / main ratio** | **1.153** |
| learning rate (final) | 3.0e-5, fully annealed |
| word-level equivalent | 67.1 (fertility 1.13, measured) |

Curve across the run:

| step | tokens | val PPL | MTP ratio | min entropy |
|---|---|---|---|---|
| 4,000 | 14.3M | 129.62 | 1.033 | 0.814 |
| 8,000 | 28.7M | 87.88 | 1.042 | 0.900 |
| 12,000 | 43.0M | 74.39 | — | 0.945 |
| 32,000 | 71.7M | 62.15 | 1.069 | 0.961 |
| 64,000 | 229.4M | **41.35** | 1.153 | — |

Against the Gate V shakedown's 87.85 on the identical corpus, tokenizer and context: **2.12x better**.

Projections made during the run were 45-60 (before starting, from Chinchilla's `D^-0.28`), then low-to-mid 40s
(at the halfway point). The result came in just under the tightened range.

## Interpretation

### MTP is confirmed working at scale

The MTP-to-main loss ratio rose monotonically through the run: 1.033 -> 1.042 -> 1.069 -> **1.153**. That is
the signature of a functioning second-token predictor. As the model improves at `t+1`, predicting `t+2` becomes
relatively harder, so the gap widens.

Before Gate U the same quantity was **0.004**, because the module was fed the embedding of its own target and
learned the identity map. This run is the strongest evidence that the fix is correct, and it makes MTP
speculative decoding a viable next gate rather than a theoretical one.

### Load balancing held

Routing entropy improved throughout, from a minimum of 0.814 at step 4,000 to 0.961 at step 32,000, with all
five MoE layers in the 0.96-0.997 band by mid-run. Gate K observed a layer collapsing to 0.32 on a far shorter
run; nothing of the kind recurred across 64,000 steps. The loss-free bias balancer at `1e-4`, chosen in Gate L
against the literature's larger value, holds up at 32 experts over a long run.

### Generation quality

Same prompts, same seed, same sampling parameters, shakedown versus this run:

> **In 1943 , the government announced**
> *shakedown:* that it had been the basis for the local government . The United States Department of Education
> and the University of New York ( DRU ) was the first
> *2 epochs:* that it had been considering a return for the Third Air Force to be a full @-@ time unit .
> `= = = First generation = = =`

> **The bridge was built in**
> *shakedown:* the 19th century and was also the site of the first European @-@ speaking history of the area
> *2 epochs:* the mid @-@ 1990s . The bridge served as the main road for the American @-@ Irish ... communities

The model now commits to specifics that fit the prompt rather than assembling plausible-shaped noun phrases.
Empty constructions like "the first European-speaking history of the area" are largely gone. It still drifts
across paragraphs and invents facts freely, which is expected at this scale.

### Honest calibration

At 41.35 subword perplexity the word-level equivalent is 67.1, using the fertility of **1.1300 tokens/word**
measured directly on the validation split (241,684 subword tokens over 213,886 whitespace words) rather than
the 1.11 estimated in earlier gates.

A published 6-layer, 156M-parameter decoder reaches 29.4 word-level on WikiText-103. This model remains roughly
**2.3x** off that. Real progress from where the project started, not parity, and closing the rest would need
Phase 2's context extension and a substantially larger token budget.

## Two defects the run exposed

### Throughput degraded progressively

Measured step rates, each a clean 4,000-step interval:

| interval | rate |
|---|---|
| steps 12,000-32,000 (average) | 3.19 steps/s |
| steps 32,000-36,000 | 1.387 steps/s |
| steps 36,000-40,000 | 1.332 steps/s |

The run was projected to finish at 03:33 and finished at 06:58, so it degraded further overnight.

Diagnosis at step 40,000: the trainer had **11,242 MiB committed against a 257 MiB resident working set**.
Windows had trimmed it to almost nothing, so `PackedTokenProvider`'s 14 random slices per step into the 922 MiB
token tensor were taking hard page faults against the pagefile. The GPU showed 72-97% utilisation at full 1972
MHz clocks and 60-63C throughout: not compute-bound, not thermally limited, starved waiting on data.

The early "3.19 steps/s average" almost certainly conceals a gradual decline rather than a step change, which
points at memory growth over the life of the process.

Two follow-ups, neither addressed here:

- **`train_tokens.pt` stores 115,241,113 tokens as int64: 922 MiB.** The vocabulary is 32,000, which fits in
  uint16. The same data is 230 MiB, a 4x reduction that would remove the paging pressure outright. This is a
  real pipeline defect that has silently cost throughput on every real-corpus run.
- **Why committed memory reaches 11.2 GiB** is unexplained and is the more interesting question.

A measurement-methodology note, recorded because it was gotten wrong twice in this gate: the slowdown was
initially attributed to this session's own disk activity. Two consecutive intervals with very different external
I/O produced near-identical rates (2,883s and 3,004s), which ruled that out. Attribution needs a controlled
comparison, not a plausible coincidence.

### The final checkpoint records no validation perplexity

`v3_cli.py` writes the last checkpoint from `train_steps`' returned history, which carries training metrics
only. Periodic checkpoints run the deterministic eval sweep and record `validation_perplexity`; the final one
does not. The headline number for this run had to be recomputed separately. Worth closing so the artifact
carries its own result.

## Changes

None to the library. This gate is a training run on the stack as of Gate V.

Checkpoint: `checkpoints/compact_v3_wikitext103_2ep.pt` (1777.7 MiB, step 64,000, validation perplexity 41.35).

## Next

Serving, which this run makes worth doing:

- **int8 export quality check.** Round-tripping this model's weights through int8 with a scale per 32 gives
  0.46-0.93% relative error by tensor group. The perplexity cost needs measuring against the 41.35 baseline
  before anything ships.
- **Inference-only artifact.** The checkpoint is 1777.7 MiB of which 1185 MiB is AdamW state. Weights only is
  592 MiB; int8 is ~165 MiB, a 10.8x reduction. Tied storage must be preserved during export or the file
  inflates to 780 MiB.
- **Hugging Face auto-pull**, so `--checkpoint` resolves against the model repo when the file is absent locally.

Then MTP speculative decoding, now that the draft head demonstrably predicts.
