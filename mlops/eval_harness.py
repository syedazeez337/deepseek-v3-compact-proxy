"""Run lm-evaluation-harness against a Compact V3 checkpoint.

The point is `word_perplexity`. The repo's headline 41.35 is subword perplexity
under its own 32K BPE tokenizer, which is not comparable to any published
number. The README currently bridges that gap by hand:

    at a measured fertility of 1.13 tokens per word, 41.35 subword perplexity
    is 67.1 word-level, against 29.4 for a published 6-layer 156M decoder

lm-eval's wikitext task computes word_perplexity directly from rolling
loglikelihoods over the raw text, normalising by whitespace word count rather
than by tokens. That is tokenizer-independent by construction, so it replaces
the hand conversion with the metric the papers actually report.

    uv run python mlops/eval_harness.py --checkpoint checkpoints/compact_v3_wikitext103_2ep.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hf_wrapper import CompactV3ForCausalLM, tokenizer_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description="lm-eval against a Compact V3 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/compact_v3_wikitext103_2ep.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data_v3_103/tokenizer.json"))
    parser.add_argument("--tasks", default="wikitext")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="documents per task; omit for the full set")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("mlops/results/lm_eval.json"))
    args = parser.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    model = CompactV3ForCausalLM.from_compact_checkpoint(args.checkpoint).to(args.device)
    tokenizer = tokenizer_from_json(args.tokenizer, model.config.context_length)

    # max_length must be the model's real window. HFLM would otherwise infer a
    # default far larger than 520 and every rolling window would overrun it.
    harness = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=model.config.context_length,
        device=args.device,
    )

    started = time.perf_counter()
    results = lm_eval.simple_evaluate(
        model=harness,
        tasks=args.tasks.split(","),
        limit=args.limit,
        bootstrap_iters=0,
    )
    elapsed = time.perf_counter() - started

    report = {
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "max_length": model.config.context_length,
        "limit": args.limit,
        "seconds": round(elapsed, 1),
        "results": results["results"],
        "n_samples": {k: v for k, v in (results.get("n-samples") or {}).items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
