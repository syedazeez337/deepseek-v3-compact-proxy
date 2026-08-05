# Gate H — WikiText-2 real-corpus smoke

## Status

Passed on 2026-08-03 after correcting checkpoint provider/RNG restoration.

## Research basis

The first real corpus was selected conservatively from the official Salesforce WikiText dataset card:

- dataset: `Salesforce/wikitext`;
- configuration: `wikitext-2-raw-v1`;
- license listed by the card: CC BY-SA 4.0 / GFDL;
- small enough for a transparent first control;
- train/validation/test splits provided;
- raw text retained for a causal LM;
- FineWeb streaming is deferred until the small control is understood.

The tokenizer follows Hugging Face Tokenizers' documented BPE workflow, fitted on the training split only. The loader inserts EOS after non-empty documents and packs fixed causal blocks.

Sources:

- https://huggingface.co/datasets/Salesforce/wikitext
- https://huggingface.co/docs/datasets/en/stream
- https://huggingface.co/docs/tokenizers/quicktour

## Provenance metadata

Cached under `data_v3/`:

```text
train documents:       23,767
validation documents:   2,461
tokenizer vocabulary:  32,000
train tokens:       2,277,210
validation tokens:    245,635
train token SHA256: de348dba7822b36fcbc1d79f006e6031e4283c897e570166fabe4f0659c49190
validation SHA256:  88acc0d407783b75292594e2b45ed0303442558e9818cd097d6d572c301c2a54
tokenizer SHA256:    6399670aae15012fc2cebe550e30caa377b4324636c7bc319f9e121a8455980b
```

The tokenizer was fitted on training documents only. Metadata records the dataset configuration, license, preprocessing, split counts, token counts, and hashes.

## CUDA smoke command

```powershell
uv run python v3_cli.py --real-corpus --steps 1 --batch-size 1 --sequence-length 64 --generate 2 --device cuda --checkpoint $env:TEMP\compact-v3-wikitext-smoke.pt
```

Result:

```text
trained steps:       1
main loss:           10.4714
MTP loss:            10.5580
balance loss:         0.000209
combined loss:       13.6390
grad norm:            8.7223
loaded checkpoint:    step 1
generation speed:     9.27 tokens/sec
peak generation VRAM: 434.54 MB
GPU:                  NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

The temporary smoke checkpoint was deleted after the run. The `data_v3/` corpus/tokenizer cache remains intentionally for reproducibility.

## Validation

```text
Data contract tests: 3 passed
CLI/training/data tests: 11 passed
Active V3 suite: 45 passed
Compilation: passed
uv lock check: passed
Real-corpus CUDA train/reload/generate: passed
```

## Next controlled experiment

Run a real dense-control experiment for 1M–5M tokens with fixed WikiText-2 metadata, validation loss/perplexity, tokens/sec, VRAM, parameter report, and continuous generation. Do not increase model size or add another architectural variable until this control is recorded.
