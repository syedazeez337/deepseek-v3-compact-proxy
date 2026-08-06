# Gate X — instrumentation, provenance, and release artifacts

Status: experimental, on branch `mlops-experiments`. Nothing in `src/compact_v3/` was modified.

## Specification

Every gate so far produced its numbers by printing JSON to a terminal and having them copied into a document by
hand. That works, and it produced 23 defensible gates. It has three failure modes this gate tests for:

1. A run's numbers are separated from the code that produced them. Nothing records the commit.
2. A single-seed comparison cannot state its own noise floor, so an effect smaller than seed variance can be
   read as a result. Gate U already found this had happened twice.
3. The published artifact carries optimizer state it does not need and is deserialized with `weights_only=False`,
   which executes arbitrary code on load.

The question: can these be closed additively, without touching the model code, and does the instrumentation
reproduce a conclusion the repo already reached by hand?

## Implementation

New directory `mlops/`, six files, no changes to `src/compact_v3/` or to `v3_cli.py`.

| File | Role |
|---|---|
| `provenance.py` | Commit, branch, dirty flag, argv, torch/CUDA/GPU, and the corpus SHA256 triple as one dict |
| `track.py` | `start/log/summary/finish` over trackio, MLflow, or nothing, selected by `MLOPS_BACKEND` |
| `run_experiment.py` | One tracked arm; imports the library directly rather than going through `v3_cli.py` |
| `compare.py` | Reads arm results and reports the effect against the measured seed spread |
| `export.py` | `.pt` to weights-only `.safetensors` plus a provenance sidecar |
| `verify_export.py` | Proves the export is the same model as its source |

`.github/workflows/tests.yml` runs the existing 90 tests on `macos-latest` and `ubuntu-latest`.

## Experiment 1: does the tooling know when it cannot tell two settings apart?

A 2x2 over `route_scale` and seed. The seed arms exist only to measure the noise floor that the `route_scale`
effect has to clear. 15.8M parameters (d_model 256, n_layer 4, 8 routed experts, top_k 2), WikiText-2, 600 steps
at batch 8 sequence 256, so 1,228,800 tokens per arm. Validation is the repo's deterministic `evaluate_tokens`
over 20 fixed windows.

```bash
MLOPS_BACKEND=trackio uv run python mlops/run_experiment.py --name rs075_s1337 --route-scale 0.75 --seed 1337 --steps 600
```

### Results

| arm | `route_scale` | seed | validation PPL | seconds |
|---|---:|---:|---:|---:|
| rs075_s999 | 0.75 | 999 | 583.70 | 72.7 |
| rs250_s999 | 2.5 | 999 | 584.56 | 71.7 |
| rs075_s1337 | 0.75 | 1337 | 592.47 | 72.3 |
| rs250_s1337 | 2.5 | 1337 | 593.55 | 72.6 |

| grouping | mean PPL | spread |
|---|---:|---:|
| `route_scale` 0.75 | 588.09 | 8.77 across seeds |
| `route_scale` 2.5 | 589.05 | 8.99 across seeds |
| seed 999 | 584.13 | |
| seed 1337 | 593.01 | |

**`route_scale` effect: 0.96 PPL. Seed effect: 8.88 PPL.** The seed moved perplexity 9.3x more than the
hyperparameter did. `compare.py` reports `NOT RESOLVED at this budget`.

### Interpretation

This does not contradict Gate P and must not be read as doing so. Gate P ran Configuration B, 155M parameters,
1,954 steps at roughly 4.0M tokens, and measured 296.89 for `route_scale=0.75` against 311.33 for 2.5, a 14.4 PPL
effect. This gate ran a 10x smaller model on a third of the tokens, where the same effect is 0.96 PPL and
invisible.

What it does establish is the number Gate P could only estimate. Gate P wrote that 0.75, 1.0, and 1.5 landed
"within 0.7% of each other, likely within run-to-run noise for a single seed", and declined to rank them on
perplexity alone, ranking on load entropy instead. That judgement was correct. The measured seed spread here is
1.53% of the mean, roughly twice the 0.7% band Gate P set aside, so Gate P was right to set it aside and right
that its 4.2% result for 2.5 sat well clear of it.

The tooling passed the test it was given: it declined to report an effect it could not see, and it did so from a
run table rather than from an author's judgement.

## Experiment 2: what does a checkpoint know about itself?

`mlops/inspect_checkpoints.py` on the flagship artifact.

```text
compact_v3_wikitext103_2ep.pt   1864.1 MB   step 64000   tokens_seen 229,376,000
  has_validation_perplexity: false
  has_provenance: false
  metric_keys: balance_loss, combined_loss, corpus_metadata_path, grad_norm,
               learning_rate, main_loss, mtp_loss, mtp_weight, tokenizer_path
```

The README's Gate W defect list is confirmed against the file. The checkpoint the README headlines at 41.35
validation perplexity, and which is published on Hugging Face, does not contain that number. `serve.py:50` reads
`validation_perplexity` and gets `None`. The same is true of `compact_v3_wikitext_moe_1m_gatem_topk2.pt`, so this
is not specific to Gate W.

No checkpoint on disk records a commit hash. The 41.35 figure is attributable to a run, but not by the artifact
itself to a revision of the code.

## Experiment 3: the release artifact

`mlops/export.py` on `compact_v3_wikitext_moe_1m.pt`.

```text
177,751,872 bytes  ->  61,662,280 bytes    reduction 65.3%
134 tensors written
dropped as tied: output.weight, mtp.token_embedding.weight, mtp.output_head.weight
```

Three tensors alias `token_embedding.weight`, not one. `model.py:33` ties the output head, and `MTPObjective`
takes both the embedding and the output head by reference, so the MTP pair aliases the same storage again.

safetensors refuses to write aliased tensors at all, which is what surfaced this. Writing them as independent
copies instead, which is the obvious way to make the error go away, would have added 98,304,000 bytes to a
61,662,280-byte file: a 2.6x inflation, silently, with no error and no accuracy change to reveal it.

`mlops/verify_export.py` rebuilds the model from each artifact and compares logits on fixed input:

```text
max_abs_logit_difference: 0.0
bit_identical: true
```

## Experiment 4: CI cost

`uv pip compile --python-platform linux` resolves 15 `nvidia-*` wheels, because `pyproject.toml` scopes torch to
the `pytorch-cu130` index on Windows and Linux. A Linux runner therefore downloads the entire CUDA stack to run
tests that skip without a GPU. `uv --torch-backend cpu` does not override this; the `[tool.uv.sources]` pin wins.

`--python-platform macos` resolves zero `nvidia-*` wheels, because that marker routes macOS to PyPI. The workflow
runs both, with macOS as the cheap path. No change to `pyproject.toml` was needed.

## What this cost

Trackio: 13 packages, 3.6 seconds to install. Four arms: 290 seconds of GPU. Total new code: six files, none of
them touching the model.

## Open issues

- Trackio writes to `~/.cache/huggingface/trackio/*.db`, outside the repository and outside git. The experiment
  record is therefore not covered by the provenance discipline this gate is trying to establish.
- `run_experiment.py` duplicates the training setup that `v3_cli.py` already does. That was deliberate, to keep
  this gate additive, but it is not the shape to keep.
- MLflow was implemented in `track.py` but not exercised. Only trackio was run.
- n=2 per cell is enough to show the seed dominates, not enough to put an interval on either.
