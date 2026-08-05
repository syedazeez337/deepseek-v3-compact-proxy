# Gate G — Synthetic train → checkpoint → resume → generate lifecycle

## Status

Passed on 2026-08-03.

## Research basis

The lifecycle follows the already researched PyTorch guidance:

- `torch.autocast` and `GradScaler` remain inside the training path;
- checkpoint state is loaded before resuming;
- generation uses `torch.inference_mode()`;
- ordinary generation uses the main model and MLA cache; MTP speculative decoding is deferred until the main path is stable;
- CUDA bitwise reproducibility is not claimed because sparse CUDA operations may be nondeterministic.

## Implementation

Created `v3_cli.py` with a continuous synthetic lifecycle:

```text
create model/provider
  -> train target steps
  -> save full checkpoint
  -> create fresh model/optimizer/scaler/provider
  -> load checkpoint
  -> generate through cached MLA path
  -> print metrics and environment metadata
```

The checkpoint contains model, optimizer, scaler, model/training configuration, batch-provider state, RNG state, step, tokens seen, and metrics.

## CPU lifecycle measurement

Command:

```powershell
uv run python v3_cli.py --steps 2 --batch-size 1 --sequence-length 8 --generate 2 --device cpu --checkpoint $env:TEMP\compact-v3-lifecycle-check.pt
```

Result:

```text
trained steps:       2
loaded step:         2
main loss:           10.4409
MTP loss:            10.3662
combined loss:       13.5510
grad norm:           23.6278
generation output:   prompt plus 2 generated token IDs
generation speed:    96.04 tokens/sec
```

The temporary checkpoint was deleted after validation.

## Validation

```text
CLI/lifecycle tests: 3 passed
Training tests:      4 passed
Active V3 suite:     41 passed
Compilation:         passed
uv lock check:       passed
Real CPU lifecycle:  passed
```

## Scope boundary

This gate proves lifecycle mechanics only. It uses synthetic random tokens, so repeated output is expected and no language-quality claim is made. The next phase is a controlled short real-corpus smoke run, beginning with dataset/tokenizer provenance and a 1M–5M token dense-control experiment.
