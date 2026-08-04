# Gate A — Compact MLA

## Status

Passed on 2026-08-03.

## Environment

```text
OS: Windows 11
Python: 3.13.14
PyTorch: 2.13.0+cu130
CUDA runtime: 13.0
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Driver: 610.88
```

## Configuration A

```text
context_length: 256
d_model: 256
n_heads: 4
q_lora_rank: 64
kv_lora_rank: 64
qk_nope_head_dim: 32
qk_rope_head_dim: 16
v_head_dim: 32
```

## Validation

```text
MLA tests: 6 passed
Python compilation: passed
uv lock check: passed
CUDA prefill/decode: passed
```

The valid CUDA smoke used a 255-token prefill followed by one-token decode, reaching context length 256.

```text
prefill output: (1, 255, 256)
cache compressed_kv: (1, 255, 64)
cache rope_keys: (1, 255, 16)
next cache length: 256
peak allocated VRAM: 11.57 MB
```

Cache accounting at batch 1 and sequence length 255:

```text
MLA compressed KV + RoPE values: 20,400
Full per-head K/V values:         81,600
Reduction:                        4x fewer values
```

The implementation currently prioritizes readable reference/cache equivalence. Weight absorption and kernel optimization are deferred until after the MoE/model integration gates.

## Research-engineering note

An initial smoke command intentionally attempted a 257th position after a 256-token prefill and was rejected by the explicit context bound. The corrected 255+1 test passed. The failure was recorded as a boundary-validation check, not hidden.
