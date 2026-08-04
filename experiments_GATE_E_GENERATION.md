# Gate E — Cache-aware generation

## Status

Passed for correctness on 2026-08-03.

## Research basis

The official DeepSeek-V3 implementation distinguishes ordinary generation from MTP speculative decoding. MTP is not required for the ordinary generation path, so this gate uses the main model only. PyTorch inference uses `torch.inference_mode()` to avoid autograd bookkeeping during generation.

Sources:

- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md
- https://docs.pytorch.org/docs/2.13/generated/torch.autograd.grad_mode.inference_mode.html
- https://docs.pytorch.org/docs/2.13/generated/torch.multinomial.html

## Implementation

Created `v3_generation.py` with:

- cached prefill/decode generation;
- uncached full-recomputation reference generation;
- greedy decoding;
- temperature sampling;
- top-k filtering;
- top-p filtering;
- seeded sampling;
- context-bound checks;
- CUDA peak-memory/timing reporting;
- explicit inference mode.

## Validation

```text
Generation tests: 6 passed
Active V3 suite:  34 passed
Compilation:      passed
uv lock check:     passed
CUDA generation:   passed
```

Cached and uncached greedy generation produced identical token IDs for the fixed test prompt.

## CUDA smoke measurement

Configuration A, FP16, prompt length 32, 8 generated tokens:

```text
cached tokens/sec:    8.69
uncached tokens/sec: 30.30
cached peak VRAM:    40.03 MB
output shape:        (1, 40)
cache compressed KV: (1, 32, 64)
cache RoPE keys:     (1, 32, 16)
```

The cached path is slower in this reference implementation. This is expected and recorded: `decode()` currently reconstructs all historical content K/V projections from the compressed cache at every token. It proves cache correctness and storage format, but it is not yet the absorbed MLA optimization. A future performance phase may absorb the up-projections into query-side operations; no speed claim is made before that work.

## Next gate

Implement the training and checkpoint pipeline only after this correctness gate. Training must preserve model, optimizer, scheduler, GradScaler, RNG, configuration, tokenizer/data identity, routing bias, and main/MTP metrics.
