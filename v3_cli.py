from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from compact_v3.model import CompactV3Model
from complete import complete_text
from compact_v3.data import DataConfig, PackedTokenProvider, evaluate_provider, load_tokenizer, prepare_wikitext2
from compact_v3.config import CompactV3Config
from compact_v3.generation import generate_cached
from compact_v3.training import (
    SyntheticBatchProvider,
    TrainingConfig,
    environment_metadata,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    seed_everything,
    train_steps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact V3 synthetic train-resume-generate lifecycle.")
    parser.add_argument("--steps", type=int, default=3, help="Total optimizer steps to reach")
    parser.add_argument("--max-tokens", type=int, default=0, help="Real-corpus token budget; overrides steps")
    parser.add_argument("--eval-batches", type=int, default=0, help="Fixed validation batches after training")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save progress every N optimizer steps")
    parser.add_argument("--dense-control", action="store_true", help="Disable MoE for the dense control")
    parser.add_argument("--enable-mtp", action="store_true", help="Enable MTP; off by default for matched architecture ablations")
    parser.add_argument("--bias-update-rate", type=float, default=1e-4, help="Router load-balancing bias update speed (DeepSeek-V3 loss-free balancing gamma)")
    parser.add_argument("--top-k", type=int, default=2, help="Number of routed experts activated per token")
    parser.add_argument("--n-dense-layers", type=int, default=1, help="Number of leading layers that use a dense FFN instead of MoE")
    parser.add_argument("--route-scale", type=float, default=0.75, help="Scale applied to normalized routed-expert weights before mixing with the shared expert")
    parser.add_argument("--mtp-weight-final", type=float, default=0.1, help="MTP loss weight after the decay-phase fraction of training is reached; set equal to mtp_weight (0.3) to disable annealing")
    parser.add_argument("--mtp-decay-fraction", type=float, default=0.6757, help="Fraction of total_steps after which the MTP loss weight switches to mtp-weight-final")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--generate", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/compact_v3_synthetic.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--real-corpus", action="store_true", help="Download/cache the configured WikiText corpus and train on it")
    parser.add_argument("--force-data", action="store_true", help="Rebuild the cached tokenizer/tokens")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Salesforce/wikitext config name, e.g. wikitext-2-raw-v1 or wikitext-103-raw-v1")
    parser.add_argument("--dataset-cache-dir", type=str, default="data_v3", help="Cache directory for the tokenizer/tokens; use a distinct dir per dataset-config to avoid overwriting another corpus's cache")
    parser.add_argument("--sample-prompt", type=str, default="The", help="Fixed prompt decoded and logged at every periodic checkpoint (real-corpus only), so generation quality can be watched over training")
    parser.add_argument("--sample-tokens", type=int, default=20, help="Number of tokens to generate for the periodic sample completion")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def make_config(
    sequence_length: int,
    bias_update_rate: float = 1e-4,
    top_k: int = 2,
    n_dense_layers: int = 1,
    route_scale: float = 0.75,
    mtp_weight_final: float = 0.1,
    mtp_decay_step_fraction: float = 0.6757,
) -> CompactV3Config:
    config = CompactV3Config(
        context_length=max(256, sequence_length + 8),
        router_bias_update_rate=bias_update_rate,
        top_k=top_k,
        n_dense_layers=n_dense_layers,
        route_scale=route_scale,
        mtp_weight_final=mtp_weight_final,
        mtp_decay_step_fraction=mtp_decay_step_fraction,
    )
    config.validate()
    return config


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.generate < 0:
        raise ValueError("steps must be positive and generate must be non-negative")
    if args.batch_size < 1 or args.sequence_length < 3:
        raise ValueError("batch-size must be positive and sequence-length must be at least 3")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed)
    corpus = None
    if args.real_corpus:
        data_config = DataConfig(
            dataset_config=args.dataset_config,
            cache_dir=args.dataset_cache_dir,
            context_length=args.sequence_length,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        corpus = prepare_wikitext2(data_config, force=args.force_data)
        model_config = CompactV3Config(
            vocab_size=corpus.metadata["tokenizer_vocab_size"],
            context_length=max(256, args.sequence_length + 8),
            use_moe=not args.dense_control,
            mtp_depth=1 if args.enable_mtp else 0,
            router_bias_update_rate=args.bias_update_rate,
            top_k=args.top_k,
            n_dense_layers=args.n_dense_layers,
            route_scale=args.route_scale,
            mtp_weight_final=args.mtp_weight_final,
            mtp_decay_step_fraction=args.mtp_decay_fraction,
        )
        provider = PackedTokenProvider(corpus.train_tokens, args.batch_size, args.sequence_length, args.seed + 1)
        tokenizer = load_tokenizer(corpus.tokenizer_path)
        print(json.dumps({"corpus": corpus.metadata}, indent=2))
    else:
        tokenizer = None
        model_config = make_config(
            args.sequence_length,
            args.bias_update_rate,
            args.top_k,
            args.n_dense_layers,
            args.route_scale,
            args.mtp_weight_final,
            args.mtp_decay_fraction,
        )
        provider = SyntheticBatchProvider(model_config.vocab_size, args.batch_size, args.sequence_length, args.seed + 1)
    model_config.validate()
    if args.max_tokens:
        args.steps = max(1, (args.max_tokens + args.batch_size * args.sequence_length - 1) // (args.batch_size * args.sequence_length))
    training_config = TrainingConfig(total_steps=args.steps, warmup_steps=min(2, args.steps - 1))
    model = CompactV3Model(model_config).to(device)
    optimizer = make_optimizer(model, training_config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_step = 0
    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
        payload = load_checkpoint(args.checkpoint, model, optimizer, scaler, provider, device)
        start_step = int(payload["step"])
        print(json.dumps({"resumed_from": str(args.checkpoint), "step": start_step}, indent=2))
    validation_provider = None
    if model_config.use_moe:
        def routing_report() -> dict[str, object]:
            loads = []
            entropies = []
            for block in model.blocks:
                if block.moe is not None and block.moe.last_routing is not None:
                    loads.append(block.moe.last_routing.expert_load.detach().cpu().tolist())
                    entropies.append(block.moe.last_routing.load_entropy())
            return {"expert_loads": loads, "load_entropy_by_layer": entropies}
    else:
        routing_report = lambda: {"expert_loads": [], "load_entropy_by_layer": []}

    if corpus is not None and args.eval_batches > 0:
        validation_provider = PackedTokenProvider(corpus.validation_tokens, args.batch_size, args.sequence_length, args.seed + 2)

    def save_progress(step: int, metrics: dict[str, float], batches_drawn: int) -> None:
        if args.checkpoint_every <= 0 or step % args.checkpoint_every != 0:
            return
        checkpoint_metrics = dict(metrics)
        checkpoint_metrics["tokens_seen"] = batches_drawn * args.batch_size * args.sequence_length
        checkpoint_metrics["routing"] = routing_report()
        if tokenizer is not None:
            checkpoint_metrics["sample"] = complete_text(model, tokenizer, args.sample_prompt, max_new_tokens=args.sample_tokens)
        if validation_provider is not None:
            validation = evaluate_provider(model, validation_provider, args.eval_batches, device)
            checkpoint_metrics.update({f"validation_{key}": value for key, value in validation.items()})
            checkpoint_metrics["validation_perplexity"] = float(np.exp(validation["main_loss"]))
        if corpus is not None:
            checkpoint_metrics["corpus_metadata_path"] = str(corpus.metadata_path)
            checkpoint_metrics["tokenizer_path"] = str(corpus.tokenizer_path)
        save_checkpoint(
            args.checkpoint, model, optimizer, scaler, start_step + step,
            checkpoint_metrics["tokens_seen"], training_config, provider, checkpoint_metrics,
        )
        print(json.dumps({"progress_step": start_step + step, "metrics": checkpoint_metrics}, indent=2), flush=True)

    remaining = max(args.steps - start_step, 0)
    history = []
    if remaining:
        history = train_steps(
            model,
            optimizer,
            scaler,
            provider,
            training_config,
            device,
            steps=remaining,
            start_step=start_step,
            progress_callback=save_progress,
        )
        last_metrics = history[-1]
        completed_step = start_step + len(history)
        tokens_seen = provider.batches_drawn * args.batch_size * args.sequence_length
        checkpoint_metrics = dict(last_metrics)
        if corpus is not None:
            checkpoint_metrics["corpus_metadata_path"] = str(corpus.metadata_path)
            checkpoint_metrics["tokenizer_path"] = str(corpus.tokenizer_path)
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            scaler,
            completed_step,
            tokens_seen,
            training_config,
            provider,
            checkpoint_metrics,
        )
        print(json.dumps({"trained_steps": len(history), "checkpoint": str(args.checkpoint), "metrics": last_metrics}, indent=2))
    else:
        completed_step = start_step
        print(json.dumps({"trained_steps": 0, "checkpoint": str(args.checkpoint), "reason": "target step already reached"}, indent=2))
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if corpus is not None and args.eval_batches > 0:
        validation_provider = PackedTokenProvider(corpus.validation_tokens, args.batch_size, args.sequence_length, args.seed + 2)
        validation_model = CompactV3Model(model_config).to(device)
        validation_optimizer = make_optimizer(validation_model, training_config)
        validation_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        load_checkpoint(args.checkpoint, validation_model, validation_optimizer, validation_scaler, provider, device)
        validation = evaluate_provider(validation_model, validation_provider, args.eval_batches, device)
        validation["perplexity"] = float(np.exp(validation["main_loss"]))
        print(json.dumps({"validation": validation}, indent=2))
        del validation_model, validation_optimizer, validation_scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not args.checkpoint.exists():
        raise RuntimeError("training did not produce a checkpoint")
    reload_model = CompactV3Model(model_config).to(device)
    reload_optimizer = make_optimizer(reload_model, training_config)
    reload_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if args.real_corpus:
        reload_provider = PackedTokenProvider(corpus.train_tokens, args.batch_size, args.sequence_length, args.seed + 1)
    else:
        reload_provider = SyntheticBatchProvider(model_config.vocab_size, args.batch_size, args.sequence_length, args.seed + 1)
    payload = load_checkpoint(args.checkpoint, reload_model, reload_optimizer, reload_scaler, reload_provider, device)
    loaded_step = payload["step"]
    del reload_optimizer, reload_scaler, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if tokenizer is not None:
        sample = complete_text(reload_model, tokenizer, args.sample_prompt, max_new_tokens=args.generate)
        prompt_report: dict[str, object] = {"prompt": sample["prompt"], "completion": sample["completion"]}
        generation_tokens_per_second = sample["tokens_per_second"]
        generation_peak_allocated_mb = None
    else:
        prompt = torch.randint(model_config.vocab_size, (1, min(8, model_config.context_length - args.generate)), device=device)
        result = generate_cached(reload_model, prompt, args.generate)
        prompt_report = {"prompt": prompt.detach().cpu().tolist(), "generated_tokens": result.tokens.detach().cpu().tolist()}
        generation_tokens_per_second = result.tokens_per_second
        generation_peak_allocated_mb = result.peak_allocated_mb
    print(json.dumps({
        "environment": environment_metadata(device),
        "loaded_step": loaded_step,
        **prompt_report,
        "generation_tokens_per_second": generation_tokens_per_second,
        "generation_peak_allocated_mb": generation_peak_allocated_mb,
        "completed_step": completed_step,
    }, indent=2))


if __name__ == "__main__":
    main()
