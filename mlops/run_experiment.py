"""One tracked training arm, small enough to iterate on.

Imports the library directly rather than going through `v3_cli.py`, so nothing
in the existing code path changes. Each arm runs in its own process; VRAM and
RNG state do not leak between them.

    uv run python mlops/run_experiment.py --name rs075 --route-scale 0.75 --seed 1337
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import track
from provenance import provenance

from compact_v3.config import CompactV3Config
from compact_v3.data import DataConfig, PackedTokenProvider, evaluate_tokens, prepare_wikitext2
from compact_v3.model import CompactV3Model
from compact_v3.training import TrainingConfig, make_optimizer, seed_everything, train_steps

# Deliberately smaller than Configuration A: the question here is whether the
# tooling can tell two arms apart, which needs many short runs rather than one
# faithful one.
SEQUENCE_LENGTH = 256
BATCH_SIZE = 8
EVAL_EVERY = 50
EVAL_BATCHES = 20


def build_config(vocab_size: int, route_scale: float, top_k: int) -> CompactV3Config:
    config = CompactV3Config(
        vocab_size=vocab_size,
        context_length=SEQUENCE_LENGTH + 8,
        n_layer=4,
        d_model=256,
        n_heads=4,
        n_dense_layers=1,
        n_routed_experts=8,
        top_k=top_k,
        expert_hidden_dim=256,
        route_scale=route_scale,
    )
    config.validate()
    return config


def mean_load_entropy(model: CompactV3Model) -> float:
    entropies = [
        block.moe.last_routing.load_entropy()
        for block in model.blocks
        if block.moe is not None and block.moe.last_routing is not None
    ]
    return float(np.mean(entropies)) if entropies else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="One tracked experiment arm.")
    parser.add_argument("--name", required=True, help="run name in the tracker")
    parser.add_argument("--project", default="compact-v3-mlops")
    parser.add_argument("--route-scale", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache-dir", default="data_v3")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results", type=Path, default=Path("mlops/results"))
    args = parser.parse_args()

    device = torch.device(args.device)
    seed_everything(args.seed)

    corpus = prepare_wikitext2(
        DataConfig(cache_dir=args.cache_dir, context_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE, seed=args.seed)
    )
    model_config = build_config(corpus.metadata["tokenizer_vocab_size"], args.route_scale, args.top_k)
    training_config = TrainingConfig(total_steps=args.steps, warmup_steps=min(50, args.steps - 1))

    model = CompactV3Model(model_config).to(device)
    optimizer = make_optimizer(model, training_config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    provider = PackedTokenProvider(corpus.train_tokens, BATCH_SIZE, SEQUENCE_LENGTH, args.seed + 1)

    parameters = sum(p.numel() for p in model.parameters())
    record = provenance(args.cache_dir)
    run_config = {
        "arm": args.name,
        "route_scale": args.route_scale,
        "top_k": args.top_k,
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "parameters": parameters,
        "model": asdict(model_config),
        "provenance": record,
    }
    run = track.start(args.project, args.name, run_config)
    print(json.dumps({"run": args.name, "backend": run.backend, "parameters": parameters}, indent=2), flush=True)

    started = time.perf_counter()
    history: list[dict[str, float]] = []

    def on_step(step: int, metrics: dict[str, float], batches_drawn: int) -> None:
        elapsed = time.perf_counter() - started
        row = {
            **{key: float(value) for key, value in metrics.items()},
            "tokens_seen": batches_drawn * BATCH_SIZE * SEQUENCE_LENGTH,
            "tokens_per_second": batches_drawn * BATCH_SIZE * SEQUENCE_LENGTH / max(elapsed, 1e-9),
            "load_entropy": mean_load_entropy(model),
        }
        if step % EVAL_EVERY == 0 or step == args.steps:
            validation = evaluate_tokens(
                model, corpus.validation_tokens, BATCH_SIZE, SEQUENCE_LENGTH, device, max_batches=EVAL_BATCHES
            )
            row["validation_loss"] = validation["main_loss"]
            row["validation_perplexity"] = float(np.exp(validation["main_loss"]))
            print(json.dumps({"step": step, "validation_perplexity": row["validation_perplexity"]}), flush=True)
        run.log(step, row)
        history.append({"step": step, **row})

    train_steps(model, optimizer, scaler, provider, training_config, device, steps=args.steps, progress_callback=on_step)

    final = evaluate_tokens(
        model, corpus.validation_tokens, BATCH_SIZE, SEQUENCE_LENGTH, device, max_batches=EVAL_BATCHES
    )
    summary = {
        "final_validation_loss": final["main_loss"],
        "final_validation_perplexity": float(np.exp(final["main_loss"])),
        "wall_clock_seconds": time.perf_counter() - started,
        "parameters": parameters,
    }
    run.summary(summary)
    run.finish()

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"{args.name}.json").write_text(
        json.dumps({"config": run_config, "summary": summary, "history": history}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run": args.name, **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
