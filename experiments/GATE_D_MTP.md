# Gate D — Sequential depth-1 MTP

## Status

Passed on 2026-08-03.

## Implementation

Created `mtp.py` with:

- horizon-2 target construction;
- hidden-state alignment;
- RMSNorm for hidden and future-token embeddings;
- concatenation and projection back to model width;
- one causal Transformer refinement layer;
- shared token embedding and output head;
- separate main and MTP losses;
- configurable MTP weight;
- explicit depth-0 disabled path.

`CompactV3Model` now reports the MTP result in diagnostics without changing the base model's main-loss API. The caller forms the combined objective explicitly:

```text
combined_loss = main_loss + mtp_weight * mtp_loss
```

## CUDA Configuration A measurement

```text
main loss:       8.8851
MTP loss:       10.3972
combined loss:  12.0045
MTP logits:     (1, 62, 32000)
unique params:  15,411,968
peak VRAM:      119.88 MB
```

The MTP logits have sequence length `64 - 2 = 62`, proving horizon-2 alignment.

## Validation

```text
MTP tests:              6 passed
Integrated V3 tests:   22 passed
Active V3 suite:       28 passed
Compilation:            passed
uv lock check:          passed
CUDA FP16 MTP step:     passed
```

## Next gate

Implement cache-aware generation as a separate module, compare cached and uncached greedy decoding, then add training/checkpoint infrastructure only after generation equivalence passes.
