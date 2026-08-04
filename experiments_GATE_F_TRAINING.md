# Gate F — Reproducible training and checkpoint pipeline

## Status

Passed for the synthetic training gate on 2026-08-03.

## Research basis

The implementation follows PyTorch 2.13 guidance:

- `torch.autocast` handles FP16 forward computation;
- `torch.amp.GradScaler` scales the loss and gradients;
- gradient accumulation keeps scaled gradients and a constant scale across microbatches;
- `unscale_` and gradient clipping occur once after accumulation and before `step`;
- `step`/`update` occur once per effective batch;
- checkpoint state includes model, optimizer, scaler, scheduler-equivalent progress, RNG, and data-provider state;
- non-reentrant activation checkpointing is the selected future option;
- exact bitwise CUDA determinism is not promised because sparse CUDA operations such as `index_add_` can be nondeterministic.

Sources:

- https://docs.pytorch.org/docs/2.13/notes/amp_examples.html
- https://docs.pytorch.org/docs/2.13/notes/randomness.html
- https://docs.pytorch.org/docs/2.13/checkpoint.html
- https://pytorch.org/tutorials/beginner/saving_loading_models.html

## Implementation

Created `v3_training.py` with:

- `TrainingConfig` validation;
- deterministic synthetic token batch provider;
- AdamW parameter groups;
- warmup/cosine learning-rate calculation;
- FP16 autocast and GradScaler support;
- accumulation-level loss scaling;
- one-time unscale and clipping per optimizer step;
- main, MTP, and balance loss metrics;
- gradient norm and learning-rate metrics;
- Python/NumPy/PyTorch/CUDA RNG capture and restore;
- batch-provider RNG/state capture and restore;
- model/config/optimizer/scaler/metric checkpointing;
- configuration mismatch rejection;
- environment metadata helper.

## Synthetic CUDA measurement

Configuration A, one effective optimizer step, microbatch 1, sequence 32:

```text
main loss:       10.4631
MTP loss:        10.4096
balance loss:     0.000205
combined loss:   13.5862
grad norm:       10.9615
learning rate:    0.0003
unique params:   15,411,968
peak VRAM:       314.02 MB
```

## Validation

```text
Training tests:   4 passed
Active V3 suite: 38 passed
Compilation:      passed
uv lock check:    passed
CUDA AMP step:    passed
```

## Failure recorded and corrected

An initial manual smoke cast the model parameters to FP16 while enabling `GradScaler`, which correctly raised `ValueError: Attempting to unscale FP16 gradients.` PyTorch's documented pattern keeps model parameters in FP32 and uses FP16 autocast for computation. The pipeline now explicitly rejects manually half-cast parameters when GradScaler is enabled, and the corrected FP32-parameter/FP16-autocast smoke passes.

## Scope boundary

This gate validates the training mechanics only. It did not download a corpus, run a long training job, or claim language quality. The next step is a tiny continuous synthetic run with checkpoint save/resume and generated output, followed by a documented 1M–5M-token dense-control experiment.
