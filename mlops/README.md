# mlops/

Experimental instrumentation for this repo. Additive: nothing here is imported by `src/compact_v3/`,
`v3_cli.py`, `serve.py`, or `complete.py`, and deleting the directory returns the project to its previous state.

Measured results are in `experiments/GATE_X_MLOPS.md`.

## Install

```bash
uv pip install trackio safetensors
```

Both are installed into the existing `.venv` and deliberately not added to `pyproject.toml`, so `uv sync` reverts
them. `MLOPS_BACKEND=none` runs everything untracked if neither is present.

## Run a tracked arm

```bash
MLOPS_BACKEND=trackio uv run python mlops/run_experiment.py --name rs075_s1337 --route-scale 0.75 --seed 1337 --steps 600
```

About 72 seconds per arm on an RTX 3050 at 15.8M parameters. Results land in `mlops/results/<name>.json` and in
trackio's store.

## See the dashboard

```bash
uv run trackio show --project compact-v3-mlops
```

## Compare arms

```bash
uv run python mlops/compare.py
```

Reports each arm's perplexity, then the hyperparameter effect against the measured seed spread. It refuses to
call an effect resolved when it is smaller than the noise floor.

## Export a release artifact

```bash
uv run python mlops/export.py checkpoints/compact_v3_wikitext_moe_1m.pt
uv run python mlops/verify_export.py checkpoints/compact_v3_wikitext_moe_1m.pt mlops/exports/compact_v3_wikitext_moe_1m.safetensors
```

The first writes weights-only safetensors plus a provenance sidecar. The second proves the two artifacts produce
identical logits.

## Inspect what a checkpoint records

```bash
uv run python mlops/inspect_checkpoints.py checkpoints/compact_v3_wikitext103_2ep.pt
```
