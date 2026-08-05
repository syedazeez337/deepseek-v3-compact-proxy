# Gate U: MTP objective fix, and three other blockers cleared before a long run

Gate T made a long WikiText-103 run affordable. Before committing 17-28 GPU-hours to it, this gate audited
everything that run depends on. It found one architectural bug that had silently invalidated a prior gate's
interpretation, one failure mode that could have destroyed the run outright, one measurement defect wider than
the effects several gates were reporting, and one pre-existing regression that had made half the checkpoint
archive unloadable.

## 1. The MTP objective was degenerate

### Symptom

Measured on the Gate Q checkpoint (`..._gateq_mtp_anneal.pt`, the current best):

| | |
|---|---|
| main loss | 5.5803 (perplexity 265.1) |
| MTP loss | **0.0240 (perplexity 1.0)** |
| MTP top-1 accuracy against its own input token | **100.00%** |
| MTP loss with the input embedding shuffled, scored against the true target | 17.5075 |
| MTP loss with the input embedding shuffled, scored against the shuffled input | 0.0245 |

The head was a copy function. Given noise it faithfully echoed the noise. 3.68M parameters had learned the
identity map.

### Cause

`model.forward` passed the same tensor as both the module's embedding input and its target:

```python
mtp_hidden = align_hidden_states(hidden_states, horizon=2)     # h_i
mtp_targets = make_future_targets(token_ids, horizon=2)        # t_{i+2}
diagnostics["mtp"] = self.mtp(mtp_hidden, mtp_targets)         # input AND target
```

Inside the module that becomes `token_embedding(t_{i+2})`, merged into the hidden state, decoded through an
output head **tied to that same embedding table**. Recovering `t_{i+2}` from `Emb(t_{i+2})` through a tied head
is close to an identity, so the objective is satisfiable without predicting anything.

### The correct indexing

V3's formulation is `h_i^{'k} = M_k[RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]`. The paper is genuinely
ambiguous about the prediction target: [DeepSeek-V3 issue #655](https://github.com/deepseek-ai/DeepSeek-V3/issues/655)
raises exactly this Figure-3-versus-Equation-22 discrepancy and was closed stale with no maintainer answer.

The ambiguity resolves on its own terms. A module that predicts a token it was handed is degenerate, which is
precisely what this implementation demonstrates. V3 reports MTP improving benchmark scores and supporting
speculative decoding at 85-90% acceptance, neither of which a copy function can do. So module k takes
`Emb(t_{i+k})` and predicts **t_{i+k+1}**. At depth 1: input `h_i` and `Emb(t_{i+1})`, target `t_{i+2}`.

That indexing is also what makes MTP usable as a draft head at inference: `t_{i+1}` is exactly the token the
main model just emitted, so it is available.

Only the input offset was wrong. The target offset was already correct.

### Result after the fix

300 steps on real WikiText-2, Configuration-A-scale model:

| | before (Gate Q checkpoint, fully trained) | after (300 steps) |
|---|---|---|
| MTP loss | 0.0240 | 6.8775 |
| main loss | 5.5803 | 6.8367 |
| MTP / main ratio | 0.004 | **1.01** |
| top-1 vs target t+2 | not measured (it predicted t+2 by copying) | **14.42%** |
| top-1 vs input t+1 | **100.00%** | **0.84%** |

A ratio slightly above 1.0 is the expected signature: predicting two tokens ahead is strictly harder than one.

### Consequence for Gate Q

Gate Q concluded that MTP helps and that annealing its weight helps further (296.89 -> 294.50 -> 292.54). Those
perplexities are real, but the stated mechanism cannot be. A copy objective contributes almost no gradient
through the MTP module itself.

The most plausible surviving explanation is indirect: the MTP loss reaches the **shared** embedding table and
tied output head, and satisfying a copy objective rewards embeddings that are recoverable from themselves,
which is a mild regulariser on the embedding matrix. That is a different claim from "multi-token prediction
improves the model" and it is not what the gate asserted.

**Gate Q's numbers stand; its interpretation does not.** The three-way comparison should be re-run against the
fixed objective before any conclusion about MTP is carried forward. Note also that the Gate Q result sits well
inside the measurement noise documented in section 3 below.

### Regression test, and one that was discarded

`test_model_never_feeds_the_target_token_to_mtp` spies on the arguments `model.forward` passes and asserts the
input tokens are `token_ids[:, 1:-1]` while targets are `token_ids[:, 2:]`. Verified to fail when the bug is
reintroduced.

A behavioural test was also written and then **deleted for not discriminating**: it trained the MTP head on one
batch and checked held-out loss. With the bug deliberately reintroduced, held-out loss stayed at 3.69-4.23
against `ln(64) = 4.16` even after 500 steps, because the tiny test model (d_model 32, vocab 64) only memorises
the batch and never learns the general copy map. The pathology needs a real embedding table at d_model 512 to
emerge. A test that passes both with and without the bug is worse than no test, so the structural assertion is
the only guard kept.

## 2. Checkpoint writes were not atomic

`save_checkpoint` called `torch.save` directly on the destination path, which truncates it immediately. A crash,
OOM, or power loss during the write left an unloadable file and no previous version, because every periodic
checkpoint overwrites the same path. On a 17-28 hour run that is total loss of work.

Now written to a sibling `.tmp` and moved in with `os.replace`, which is atomic on Windows and POSIX when
source and destination share a filesystem. The temp file is removed if the write raises.

`test_checkpoint_write_is_atomic` monkeypatches `torch.save` to succeed and then raise, and asserts the existing
checkpoint is byte-identical afterwards and no temp file remains. Verified to fail against the old code.

## 3. Validation measurement was noisier than the effects being measured

`PackedTokenProvider` draws random windows with replacement and advances its generator between calls, so
`evaluate_provider` returns a different number every time it is called on an unchanged model. Four consecutive
evaluations of the Gate Q checkpoint:

| | perplexity |
|---|---|
| eval 1 | 278.20 |
| eval 2 | 295.46 |
| eval 3 | 318.99 |
| eval 4 | 279.34 |

A 40.8-point spread, 14.7%, from sampling alone.

Within a single run the seed and eval schedule are fixed, so gates comparing configurations at the same step saw
the same windows and those comparisons are internally valid. But Gate P separated its winner from the field by
0.7% and Gate Q by 0.7%, both far inside this band. **Those rankings are reproducible, not robust**: a different
evaluation seed could plausibly reorder them.

`evaluate_tokens` replaces it with a deterministic sweep over fixed, non-overlapping windows of the split.
Verified end to end through the CLI: two separate processes evaluating the same checkpoint returned main loss
`9.113075256347656` and perplexity `9073.154307438404`, identical to the last digit.

## 4. Half the checkpoint archive had been unloadable since Gate N

Found while re-verifying every checkpoint. `CompactV3Config(**payload["model_config"])` fills fields absent from
older checkpoints with **today's** defaults. Gates I-M recorded no `n_dense_layers` because Gate N introduced
it, so they were rebuilt with `n_dense_layers=1`, making block 0 a dense FFN against stored MoE weights.

This was confirmed pre-existing, not caused by this gate: the same verification run against the Gate T commit
`ea90671` produced an identical 7/15.

`config_from_checkpoint` now resolves absent fields to the value they effectively had when the checkpoint was
written (`n_dense_layers=0`, `mtp_weight_final=mtp_weight`, `mtp_decay_step_fraction=1.0`, meaning the annealing
switch never fires). `load_checkpoint` also now compares only the fields a checkpoint actually recorded, rather
than demanding exact dict equality against a config that has since grown.

| | before | after |
|---|---|---|
| checkpoints that load and generate | 7 / 15 | **14 / 15** |

The one remaining failure is `compact_v3_wikitext_moe_1m_gaten_dense1.pt`, Gate N's deliberately-kept confounded
first attempt. Its dense layer was built at width 384 while the current derivation gives 1152, and that width is
computed from config rather than stored, so the architecture is not reconstructible from what the checkpoint
records. Kept as a record; noted here as genuinely unloadable rather than quietly dropped.

The general lesson is recorded for future gates: **anything needed to rebuild the architecture must be stored in
the config, not derived from it**, or a later change to the derivation orphans every existing checkpoint.

## 5. Training context raised from 256 to 512

Not a bug, a measured free gain. Configuration B, FP16 autocast, one config per process:

| sequence | batch | tokens/step | tok/s | peak | % of 6GB |
|---|---|---|---|---|---|
| 256 (old default) | 8 | 2048 | 8,237 | 2752 MiB | 44.8% |
| **512 (new default)** | **8** | 4096 | **11,501** | 4882 MiB | 79.5% |
| 1024 | 4 | 4096 | 9,553 | 5466 MiB | 89.0% |
| 2048 | 2 | 4096 | 3,038 | 6641 MiB | 108% |

512/8 dominates the old setting on both axes: 40% more throughput and double the context, which conditions each
prediction on more text. `context_length` default moves 264 -> 520 and `--sequence-length` 256 -> 512.

The 2048 row is Gate O's Windows trap reproducing exactly: 108% of VRAM, no OOM, just a 2.7x slowdown as the
driver silently spills to shared system memory. Worth keeping the ceiling at 512 for headroom.

## Verification

- Full suite: **81 passed** (72 before this gate).
- Every fix verified to fail against the pre-fix code before being accepted, except the context change, which is
  a measurement rather than a correctness fix.
- All 15 checkpoints re-verified: 14 load and generate, 1 documented as unloadable by construction.
- End-to-end CLI on WikiText-103 at batch 8 / sequence 512: trained, checkpointed, resumed to an absolute step
  target, evaluated deterministically, and generated, with no OOM.
- Deterministic evaluation confirmed identical across two separate processes.

## Changes

- `src/compact_v3/mtp.py`: `forward` takes `input_tokens` separately from `future_targets`; added
  `make_mtp_input_tokens`.
- `src/compact_v3/model.py`: passes `token_ids[:, 1:-1]` as the MTP input.
- `src/compact_v3/training.py`: atomic checkpoint write; `load_checkpoint` compares only recorded fields.
- `src/compact_v3/data.py`: added `evaluate_tokens`; `evaluate_provider` no longer forces the model into train
  mode regardless of its prior state.
- `src/compact_v3/config.py`: added `config_from_checkpoint` and `LEGACY_FIELD_DEFAULTS`; `context_length`
  default 264 -> 520.
- `complete.py`: rebuilds config through `config_from_checkpoint`.
- `v3_cli.py`: `--sequence-length` default 256 -> 512; deterministic validation.

Checkpoint format is unchanged. Existing checkpoints load; their MTP submodules encode a copy function and are
worthless as draft heads, which is expected and requires retraining, not migration.

## Next gate

Gate V: the WikiText-103 training run, now with a working MTP objective, a trustworthy validation number, a
checkpoint that survives a crash, and 40% more throughput at double the context. At 11,501 tok/s one epoch of
115,241,113 tokens is ~2.8 h and the Chinchilla-optimal budget for this model's ~36M active parameters (720M
tokens) is ~17.4 h.

Gate W after that: MTP speculative decoding, which is only now possible. V3 reports 85-90% acceptance and 1.8x
TPS; a 155M model's draft quality will be lower, and measuring how much lower is the point.
