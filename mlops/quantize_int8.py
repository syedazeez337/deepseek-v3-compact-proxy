"""Measure what int8 costs, against the repo's own 41.35 baseline.

The roadmap says: "int8 with a scale per 32 values is about 165MB, so the
perplexity cost needs measuring against the 41.35 baseline before anything
ships." This measures it, on the repo's own deterministic evaluation, in the
same process as the fp32 baseline so nothing about the data or windowing can
differ between the two numbers.

Note on latency: torchao's int8 weight-only kernel relies on torch.compile for
its speedup. Measuring wall clock without compile shows the size win and no
latency win, which reads as a failed result but is not one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compact_v3.config import config_from_checkpoint
from compact_v3.data import evaluate_tokens
from compact_v3.model import CompactV3Model


def _tensor_bytes(tensor: torch.Tensor, seen: set[int]) -> int:
    """Bytes held by a tensor, following torchao's subclasses to their innards.

    A quantized parameter is a tensor subclass wrapping int8 data plus scales,
    and calling `untyped_storage()` on it raises. `__tensor_flatten__` names the
    inner tensors, which is the supported way to reach them.
    """
    if hasattr(tensor, "__tensor_flatten__"):
        inner_names, _ = tensor.__tensor_flatten__()
        return sum(_tensor_bytes(getattr(tensor, name), seen) for name in inner_names)
    storage = tensor.untyped_storage()
    if storage.data_ptr() in seen:
        return 0
    seen.add(storage.data_ptr())
    return storage.nbytes()


def parameter_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    return sum(_tensor_bytes(t, seen) for t in list(model.parameters()) + list(model.buffers()))


def load(checkpoint: Path, device: torch.device) -> CompactV3Model:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CompactV3Model(config_from_checkpoint(payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="int8 weight-only quantization cost.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/compact_v3_wikitext103_2ep.pt"))
    parser.add_argument("--tokens", type=Path, default=Path("data_v3_103/validation_tokens.pt"))
    parser.add_argument("--group-size", type=int, default=32, help="scale per N weights; 0 uses per-row")
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("mlops/results/int8.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    tokens = torch.load(args.tokens, weights_only=True)

    baseline_model = load(args.checkpoint, device)
    baseline_bytes = parameter_bytes(baseline_model)
    baseline = evaluate_tokens(baseline_model, tokens, args.batch_size, args.sequence_length, device)
    baseline_ppl = float(np.exp(baseline["main_loss"]))
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    from torchao.quantization.granularity import PerGroup

    quantized_model = load(args.checkpoint, device)
    # torchao 0.18 config version 2 rejects the `group_size` argument and wants
    # a granularity object. The assertion says so, which is more than most
    # deprecations manage.
    config = Int8WeightOnlyConfig(granularity=PerGroup(args.group_size)) if args.group_size else Int8WeightOnlyConfig()

    # `output`, `mtp.output_head` and `mtp.token_embedding` all alias
    # `token_embedding.weight`. quantize_ walks modules, so it reaches that one
    # storage three times, and the second visit tries to quantize an Int8Tensor
    # and dies inside aten.view. Skipping the aliases is also the right call on
    # accuracy grounds: the tied output head is the most quantization-sensitive
    # matrix in the model.
    embedding_ptr = quantized_model.token_embedding.weight.data_ptr()

    def not_tied_to_embedding(module: torch.nn.Module, name: str) -> bool:
        weight = getattr(module, "weight", None)
        return (
            isinstance(module, torch.nn.Linear)
            and weight is not None
            and weight.data_ptr() != embedding_ptr
        )

    quantize_(quantized_model, config, filter_fn=not_tied_to_embedding)
    quantized_bytes = parameter_bytes(quantized_model)
    quantized = evaluate_tokens(quantized_model, tokens, args.batch_size, args.sequence_length, device)
    quantized_ppl = float(np.exp(quantized["main_loss"]))

    report = {
        "checkpoint": str(args.checkpoint),
        "group_size": args.group_size or "per-row",
        "windows": baseline["windows"],
        "fp32_perplexity": round(baseline_ppl, 4),
        "int8_perplexity": round(quantized_ppl, 4),
        "perplexity_delta": round(quantized_ppl - baseline_ppl, 4),
        "perplexity_delta_percent": round(100 * (quantized_ppl - baseline_ppl) / baseline_ppl, 3),
        "fp32_parameter_megabytes": round(baseline_bytes / 1e6, 1),
        "int8_parameter_megabytes": round(quantized_bytes / 1e6, 1),
        "size_reduction_percent": round(100 * (1 - quantized_bytes / baseline_bytes), 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
