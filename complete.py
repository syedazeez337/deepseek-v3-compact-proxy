from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from compact_v3.model import CompactV3Model
from compact_v3.data import load_tokenizer, resolve_tokenizer_path
from compact_v3.config import CompactV3Config, config_from_checkpoint
from compact_v3.generation import generate_cached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete a text prompt using a trained checkpoint.")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="defaults to the tokenizer recorded in the checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def complete_text(
    model: CompactV3Model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    generator: torch.Generator | None = None,
) -> dict:
    was_training = model.training
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt).ids
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    result = generate_cached(
        model,
        prompt_tensor,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        generator=generator,
    )
    if was_training:
        model.train()
    generated_ids = result.tokens[0].tolist()
    return {
        "prompt": prompt,
        "completion": tokenizer.decode(generated_ids),
        "continuation_only": tokenizer.decode(generated_ids[len(prompt_ids):]),
        "tokens_per_second": result.tokens_per_second,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = load_tokenizer(resolve_tokenizer_path(args.tokenizer, payload))
    config = config_from_checkpoint(payload["model_config"])
    model = CompactV3Model(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.do_sample else None
    result = complete_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=args.do_sample,
        generator=generator,
    )
    result["checkpoint_step"] = payload["step"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
