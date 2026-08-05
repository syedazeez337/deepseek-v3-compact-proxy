# Gate C — Compact V3 decoder integration

## Status

Passed on 2026-08-03.

## Integrated architecture

```text
token embedding
  -> repeated RMSNorm -> MLA -> residual blocks
  -> RMSNorm -> shared-plus-routed DeepSeekMoE -> residual blocks
  -> final RMSNorm
  -> tied output projection
```

Created:

```text
v3_block.py
compact_v3_model.py
tests/test_compact_v3_model.py
```

The model exposes:

- full-sequence forward and causal loss;
- MLA prefill and per-token decode cache orchestration;
- exact unique parameter reporting;
- shared output/input embedding weights;
- routing and sequence-balance diagnostics.

## Configuration A measurement

```text
Configuration: CompactV3Config defaults
Unique parameters: 14,490,368
CUDA dtype:        FP16 autocast
Input:             (1, 64)
Logits:            (1, 64, 32000)
Loss:              8.9080
Peak VRAM:         108.39 MB
Forward/backward:  648.21 ms
```

## Validation

```text
Model tests:        7 passed
Active V3 tests:    22 passed
Compilation:        passed
uv lock check:      passed
CUDA FP16 step:     passed
```

The backward test intentionally verifies shared/router/attention paths and every selected routed expert. Unselected sparse experts are expected to have no gradient on a single top-k batch.

A previous strict all-parameters-gradient test failed for that expected sparse behavior; the failure and correction are recorded rather than hidden.

## Next gate

Implement sequential depth-1 MTP as a separate objective/module. Do not begin long training until MTP target offsets, loss separation, gradients, and disabled-path behavior pass.
