"""Prove the safetensors export is the same model as the .pt it came from.

Size reduction is worthless if the weights moved. This rebuilds the model from
each artifact and compares logits on a fixed input, which is the only claim that
matters for an inference-only export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from compact_v3.config import config_from_checkpoint
from compact_v3.model import CompactV3Model


def from_pt(path: Path, device: torch.device) -> CompactV3Model:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = CompactV3Model(config_from_checkpoint(payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def from_safetensors(path: Path, device: torch.device) -> CompactV3Model:
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model = CompactV3Model(config_from_checkpoint(sidecar["model_config"])).to(device)
    state = load_file(str(path), device=str(device))
    # Re-tie: the aliases were dropped at export precisely because they are the
    # same storage. Restoring them from the canonical entry is what makes the
    # smaller file lossless rather than lossy.
    for canonical, aliases in sidecar["tied_weights"].items():
        for alias in aliases:
            state[alias] = state[canonical]
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a .pt and its safetensors export.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    reference = from_pt(args.checkpoint, device)
    exported = from_safetensors(args.export, device)

    torch.manual_seed(0)
    tokens = torch.randint(reference.config.vocab_size, (2, 64), device=device)
    with torch.inference_mode():
        reference_logits, _, _ = reference(tokens)
        exported_logits, _, _ = exported(tokens)

    difference = (reference_logits - exported_logits).abs().max().item()
    tied = {
        "reference_ties_output_to_embedding": reference.tied_output_embedding(),
        "exported_ties_output_to_embedding": exported.tied_output_embedding(),
    }
    print(json.dumps({
        "max_abs_logit_difference": difference,
        "bit_identical": bool(torch.equal(reference_logits, exported_logits)),
        **tied,
        "embedding_bytes": reference.token_embedding.weight.numel() * 4,
        "naive_export_would_add_bytes": reference.token_embedding.weight.numel() * 4 * len(
            json.loads(args.export.with_suffix(".json").read_text(encoding="utf-8"))["tied_weights"].get(
                "token_embedding.weight", [])
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
