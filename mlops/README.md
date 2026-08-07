# mlops/

Experimental instrumentation for this repo. Additive: nothing here is imported by `src/compact_v3/`,
`v3_cli.py`, `serve.py`, or `complete.py`, and deleting the directory returns the project to its previous state.

Measured results are in `experiments/GATE_X_MLOPS.md` and `experiments/GATE_Y_COMPARABLE_EVALUATION.md`.

## Install

```bash
uv sync --group eval
```

The eval stack (`transformers`, `lm-eval`, `wandb`, `torchao`, `safetensors`, `accelerate`) is a dependency
group, so a plain `uv sync` still gives the lean training environment, and reverts this one.

## Run a tracked arm

```bash
WANDB_MODE=offline uv run python mlops/run_experiment.py --name rs075_s1337 --route-scale 0.75 --seed 1337 --steps 600
```

`MLOPS_BACKEND` selects the tracker: `wandb` (default), `trackio`, or `none`. `WANDB_MODE=offline` needs no
account, and `wandb sync wandb/offline-run-*` uploads those runs later.

About 72 seconds per arm on an RTX 3050 at 15.8M parameters. Results land in `mlops/results/<name>.json` as
config plus summary only; per-step history belongs in the tracker rather than in git.

## Compare arms

```bash
uv run python mlops/compare.py
```

Reports each arm's perplexity, then the hyperparameter effect against the measured seed spread, and refuses to
call an effect resolved when it is smaller than the noise floor.

## Evaluate against published numbers

```bash
uv run python mlops/eval_harness.py --checkpoint checkpoints/compact_v3_wikitext103_2ep.pt
```

Runs lm-evaluation-harness through `hf_wrapper.py`. Read `experiments/GATE_Y_COMPARABLE_EVALUATION.md` before
interpreting the output: lm-eval detokenizes WikiText before scoring, and this model was trained on the raw
form, so `word_perplexity` from this command is not comparable to published figures.

## Measure quantization cost

```bash
uv run python mlops/quantize_int8.py --group-size 0
```

int8 weight-only against the repo's own deterministic evaluation. `--group-size 0` is per-row, which measured
better than per-group-32 on both accuracy and size.

## Export a release artifact

```bash
uv run python mlops/export.py checkpoints/compact_v3_wikitext_moe_1m.pt
uv run python mlops/verify_export.py checkpoints/compact_v3_wikitext_moe_1m.pt mlops/exports/compact_v3_wikitext_moe_1m.safetensors
```

Largely superseded by `hf_wrapper.py` plus `save_pretrained`, which handles the same job through transformers.
Kept because it detects tied weights from storage pointers rather than from a config flag, which is what made it
immune to the silent save/load corruption recorded in Gate Y, and because `verify_export.py`'s logit comparison
is what caught the RoPE buffer bug.

## Inspect what a checkpoint records

```bash
uv run python mlops/inspect_checkpoints.py checkpoints/compact_v3_wikitext103_2ep.pt
```
