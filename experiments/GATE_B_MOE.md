# Gate B — V3-style shared-plus-routed MoE

## Status

Passed on 2026-08-03.

## Scope

Implemented and tested separately from the decoder:

- SwiGLU experts;
- one always-on shared expert;
- sigmoid router affinity;
- routing-only expert bias;
- top-k selection;
- normalization of original selected affinities;
- route scaling;
- token dispatch and weighted merge;
- routing diagnostics;
- sequence-balance safeguard loss;
- bias load balancer.

## Validation

```text
Active V3 tests: 15 passed
Compilation:      passed
uv lock check:    passed
CUDA MLA smoke:   passed
CUDA FP16 MoE:    passed
```

MoE CUDA smoke with Configuration A dimensions:

```text
output:       (1, 256, 256)
top_k:        1
assignments:  256
expert load:  [72, 52, 61, 71]
parameters:   1,475,584
peak VRAM:    22.90 MB
```

Top-2 behavior is tested separately and conserves exactly `tokens × top_k` assignments.

## Failure recorded

The first full-suite invocation recursively collected archived legacy tests. Those tests imported the archived `rope.py` API and failed against the new MLA API. The clean V3 module was renamed to `v3_rope.py`, MLA imports were namespaced, and the active suite is now run explicitly with:

```powershell
uv run pytest -q tests
```

This keeps the archived prior project reversible without allowing legacy modules to shadow the new V3 proxy.

## Next gate

Integrate MLA and DeepSeekMoE into a pre-norm V3 decoder block and compact language model. Do not begin long training before model shape, loss, parameter-report, and FP16 forward/backward tests pass.
