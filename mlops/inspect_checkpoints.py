"""What does a checkpoint actually know about itself?

The repo's README lists as a Gate W defect that `v3_cli.py` writes its final
checkpoint from training history, so the shipped artifact carries no validation
perplexity. This checks that claim against the checkpoints on disk rather than
taking it on faith, and is the concrete argument for stamping provenance and
summary metrics into the payload.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def inspect(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics") or {}
    return {
        "checkpoint": path.name,
        "megabytes": round(path.stat().st_size / 1e6, 1),
        "step": payload.get("step"),
        "tokens_seen": payload.get("tokens_seen"),
        "has_validation_perplexity": "validation_perplexity" in metrics,
        "validation_perplexity": metrics.get("validation_perplexity"),
        "has_tokenizer_path": "tokenizer_path" in metrics,
        "has_corpus_metadata_path": "corpus_metadata_path" in metrics,
        "has_provenance": "provenance" in payload,
        "metric_keys": sorted(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report what each checkpoint records about itself.")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.checkpoints:
        if not path.exists():
            print(json.dumps({"checkpoint": str(path), "error": "missing"}))
            continue
        print(json.dumps(inspect(path), indent=2), flush=True)


if __name__ == "__main__":
    main()
