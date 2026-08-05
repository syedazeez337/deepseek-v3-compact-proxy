# Gate I — 1M-token WikiText-2 dense control

## Status

Completed on 2026-08-04 with recoverable checkpoints and final evaluation.

## Experimental specification

```text
Dataset:             Salesforce/wikitext, wikitext-2-raw-v1
Tokenizer:           train-only BPE, 32,000 vocabulary
Model:               compact V3 MLA proxy with dense SwiGLU instead of MoE
MoE:                 disabled
MTP:                 disabled
Context:             64 tokens
Batch size:          8 sequences
Tokens/optimizer:    512
Target budget:       1,000,000 tokens
Actual tokens:       1,000,448
Optimizer steps:     1,954
Checkpoint interval: 250 steps
Validation batches:  32 final evaluation batches
Device:              RTX 3050 Laptop 6GB
Precision:           FP16 autocast + GradScaler
```

The dense control changes only the feed-forward path relative to the compact V3 model: it uses one dense SwiGLU expert in each block, with MLA and the rest of the model unchanged. This isolates the dense data/training control before adding sparse MoE.

## Provenance

```text
train documents:       23,767
validation documents:   2,461
train tokens:       2,277,210
validation tokens:    245,635
train token SHA256: de348dba7822b36fcbc1d79f006e6031e4283c897e570166fabe4f0659c49190
validation SHA256:  88acc0d407783b75292594e2b45ed0303442558e9818cd097d6d572c301c2a54
tokenizer SHA256:    6399670aae15012fc2cebe550e30caa377b4324636c7bc319f9e121a8455980b
```

## Periodic checkpoint milestone

At step 500 / 256,000 tokens:

```text
train loss:       6.6978
validation loss:  6.7829
validation PPL: 882.65
learning rate:    0.0002590
```

## Final checkpoint

```text
Checkpoint: checkpoints/compact_v3_wikitext_dense_1m.pt
Step:       1,954
Tokens:     1,000,448
Final train loss: 6.6613
```

Final no-training reload/evaluation command:

```powershell
uv run python v3_cli.py --real-corpus --dense-control --steps 1954 --batch-size 8 --sequence-length 64 --eval-batches 32 --generate 32 --device cuda --resume --checkpoint checkpoints/compact_v3_wikitext_dense_1m.pt
```

Final evaluation:

```text
validation loss: 6.3940
validation PPL:  598.27
```

Final reload and cached generation succeeded:

```text
generated tokens:       32
generation speed:       108.65 tokens/sec
peak generation VRAM:   400.92 MB
```

The generated token IDs show the model is learning token statistics, but no qualitative language claim is made from one sample. The tokenizer decoder and prompt-quality evaluation are separate follow-up work.

## Validation

```text
Dense/data/CLI tests: 19 passed
Active V3 suite:     46 passed
Periodic checkpoint smoke: passed
1M-token CUDA training: passed
Final checkpoint reload: passed
Final validation/perplexity: passed
Cached generation: passed
```

## Failure recorded

The first 1M-token launch had no periodic checkpointing and produced no recoverable progress, so it was stopped rather than allowed to run opaquely. The CLI was then changed to emit JSON progress and save checkpoints at fixed intervals. The recoverable run completed successfully.

## Interpretation

This establishes the dense MLA control. It does not yet measure MoE improvement. The next experiment should keep this dataset/tokenizer, token budget, context, batch, optimizer, and seed policy fixed while enabling four-expert top-1 MoE and comparing validation loss, perplexity, throughput, VRAM, total/active parameters, and expert loads.
