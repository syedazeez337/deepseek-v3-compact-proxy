"""Export a training checkpoint to a weights-only safetensors artifact.

A `.pt` here is a resume artifact: weights plus AdamW state plus GradScaler plus
RNG plus the data provider position. Serving needs none of that. This writes the
weights alone, in a format that executes no code on load, plus a sidecar JSON
carrying the config and provenance that the `.pt` kept inside its pickle.

Two things make this less mechanical than it looks:

1. `model.py:33` ties the output head to the token embedding, so those two
   entries share one storage. safetensors refuses to write aliased tensors, and
   writing both copies would inflate the file by a full vocab x d_model matrix.
   Ties are dropped here and recorded in the sidecar so loading can restore them.
2. `torch.load(weights_only=False)` is required to read the old pickle, because
   the payload holds a `TrainingConfig` dataclass and numpy RNG state. That is
   precisely the property this export removes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file


def find_shared_storage(state_dict: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Group parameter names by the storage they alias."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        groups[(tensor.untyped_storage().data_ptr(), tensor.shape, str(tensor.dtype))].append(name)
    return {names[0]: names[1:] for names in groups.values() if len(names) > 1}


def export(checkpoint: Path, destination: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["model"]

    ties = find_shared_storage(state_dict)
    dropped = [name for aliases in ties.values() for name in aliases]
    weights = {
        name: tensor.contiguous()
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor) and name not in dropped
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    # The safetensors header takes str -> str only, so the real metadata goes in
    # the sidecar and only a pointer lives in the header.
    save_file(weights, destination, metadata={"format": "pt", "sidecar": destination.with_suffix(".json").name})

    metrics = payload.get("metrics") or {}
    sidecar = {
        "source_checkpoint": checkpoint.name,
        "step": payload.get("step"),
        "tokens_seen": payload.get("tokens_seen"),
        "model_config": payload.get("model_config"),
        "tied_weights": ties,
        "validation_perplexity": metrics.get("validation_perplexity"),
        "tokenizer_path": metrics.get("tokenizer_path"),
        "corpus_metadata_path": metrics.get("corpus_metadata_path"),
        "provenance": payload.get("provenance"),
    }
    destination.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "export": str(destination),
        "export_bytes": destination.stat().st_size,
        "tensors_written": len(weights),
        "tensors_dropped_as_tied": dropped,
        "payload_keys": sorted(k for k in payload if k != "model"),
        "validation_perplexity_in_checkpoint": metrics.get("validation_perplexity"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a .pt checkpoint to weights-only safetensors.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    destination = args.out or Path("mlops/exports") / (args.checkpoint.stem + ".safetensors")
    report = export(args.checkpoint, destination)
    report["reduction"] = f"{1 - report['export_bytes'] / report['checkpoint_bytes']:.1%}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
