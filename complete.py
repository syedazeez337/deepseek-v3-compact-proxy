from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from compact_v3_model import CompactV3Model
from data_v3 import load_tokenizer
from v3_config import CompactV3Config
from v3_generation import generate_cached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete a text prompt using a trained checkpoint.")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data_v3/tokenizer.json"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = CompactV3Config(**payload["model_config"])
    model = CompactV3Model(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt).ids
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.do_sample else None
    result = generate_cached(
        model,
        prompt_tensor,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=args.do_sample,
        generator=generator,
    )
    generated_ids = result.tokens[0].tolist()
    print(json.dumps({
        "prompt": args.prompt,
        "completion": tokenizer.decode(generated_ids),
        "continuation_only": tokenizer.decode(generated_ids[len(prompt_ids):]),
        "checkpoint_step": payload["step"],
        "tokens_per_second": result.tokens_per_second,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
