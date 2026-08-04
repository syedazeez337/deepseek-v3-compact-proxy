from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from compact_v3_model import CompactV3Model
from mtp import mtp_weight_schedule
from v3_config import CompactV3Config


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 2
    total_steps: int = 10
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0
    seed: int = 1337
    use_checkpointing: bool = False

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.total_steps < 1 or self.warmup_steps < 0:
            raise ValueError("training steps must be valid")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")


class SyntheticBatchProvider:
    def __init__(self, vocab_size: int, batch_size: int, sequence_length: int, seed: int = 1337) -> None:
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.batches_drawn = 0

    def next(self, device: torch.device) -> tuple[Tensor, Tensor]:
        tokens = torch.randint(
            self.vocab_size,
            (self.batch_size, self.sequence_length + 1),
            generator=self.generator,
            device="cpu",
        )
        self.batches_drawn += 1
        return tokens[:, :-1].to(device), tokens[:, 1:].to(device)

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state(), "batches_drawn": self.batches_drawn}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        generator_state = torch.as_tensor(state["generator_state"], dtype=torch.uint8, device="cpu").contiguous()
        self.generator.set_state(generator_state)
        self.batches_drawn = state["batches_drawn"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate,
        betas=config.betas,
    )


def learning_rate(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.total_steps - config.warmup_steps - 1, 1)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_microbatch(
    model: CompactV3Model,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    inputs: Tensor,
    targets: Tensor,
    accumulation_steps: int,
    device: torch.device,
    use_amp: bool,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    with autocast_context(device, use_amp):
        _, main_loss, diagnostics = model(inputs, targets, update_balancer=True)
        mtp_result = diagnostics["mtp"]
        mtp_weight = mtp_weight_schedule(
            step, total_steps, model.config.mtp_weight, model.config.mtp_weight_final, model.config.mtp_decay_step_fraction
        )
        combined = model.mtp.combined_loss(main_loss, mtp_result, mtp_weight) + diagnostics["balance_loss"]
        scaled_loss = combined / accumulation_steps
    if scaler.is_enabled():
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()
    return {
        "main_loss": float(main_loss.detach().float()),
        "mtp_loss": float(mtp_result.loss.detach().float()) if mtp_result is not None else 0.0,
        "mtp_weight": mtp_weight,
        "balance_loss": float(diagnostics["balance_loss"].detach().float()),
        "combined_loss": float(combined.detach().float()),
    }


def optimizer_step(
    model: CompactV3Model,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> float:
    if scaler.is_enabled() and any(parameter.dtype == torch.float16 for parameter in model.parameters()):
        raise ValueError("keep model parameters in FP32 when using GradScaler; use autocast for FP16 computation")
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().float())


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu").contiguous()
    torch.set_rng_state(torch_state)
    if state["cuda"] is not None and torch.cuda.is_available():
        cuda_states = [torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous() for value in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    path: str | Path,
    model: CompactV3Model,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    tokens_seen: int,
    training_config: TrainingConfig,
    batch_provider: SyntheticBatchProvider,
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "model_config": asdict(model.config),
            "training_config": asdict(training_config),
            "batch_provider": batch_provider.state_dict(),
            "rng": capture_rng_state(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: CompactV3Model,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch_provider: SyntheticBatchProvider,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["model_config"] != asdict(model.config):
        raise ValueError("checkpoint model configuration does not match")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    batch_provider.load_state_dict(payload["batch_provider"])
    restore_rng_state(payload["rng"])
    return payload


def train_steps(
    model: CompactV3Model,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch_provider: SyntheticBatchProvider,
    training_config: TrainingConfig,
    device: torch.device,
    steps: int | None = None,
    start_step: int = 0,
    progress_callback=None,
) -> list[dict[str, float]]:
    training_config.validate()
    model.train()
    use_amp = device.type == "cuda"
    history = []
    optimizer.zero_grad(set_to_none=True)
    count = steps or training_config.total_steps
    for step_offset in range(count):
        step = start_step + step_offset
        set_learning_rate(optimizer, learning_rate(step, training_config))
        aggregate = {"main_loss": 0.0, "mtp_loss": 0.0, "mtp_weight": 0.0, "balance_loss": 0.0, "combined_loss": 0.0}
        for _ in range(training_config.gradient_accumulation_steps):
            inputs, targets = batch_provider.next(device)
            metrics = train_microbatch(
                model, optimizer, scaler, inputs, targets,
                training_config.gradient_accumulation_steps, device, use_amp,
                step, training_config.total_steps,
            )
            for key in aggregate:
                aggregate[key] += metrics[key] / training_config.gradient_accumulation_steps
        aggregate["grad_norm"] = optimizer_step(model, optimizer, scaler, training_config.grad_clip)
        aggregate["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(aggregate)
        if progress_callback is not None:
            progress_callback(step + 1, aggregate, batch_provider.batches_drawn)
    return history


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
    }


__all__ = [
    "SyntheticBatchProvider",
    "TrainingConfig",
    "capture_rng_state",
    "environment_metadata",
    "load_checkpoint",
    "make_optimizer",
    "restore_rng_state",
    "save_checkpoint",
    "seed_everything",
    "train_steps",
]
