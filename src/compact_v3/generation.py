from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from compact_v3.model import CompactV3Model
from compact_v3.block import BlockCache


@dataclass
class GenerationResult:
    tokens: Tensor
    seconds: float
    tokens_per_second: float
    peak_allocated_mb: float | None


def _validate_sampling(temperature: float, top_k: int | None, top_p: float) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    _validate_sampling(temperature, top_k, top_p)
    logits = logits.float() / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def greedy_next_token(logits: Tensor) -> Tensor:
    return logits.argmax(dim=-1, keepdim=True)


def generate_uncached(
    model: CompactV3Model,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    do_sample: bool = False,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError("prompt must have shape (batch, sequence) with a non-empty sequence")
    if max_new_tokens < 0 or prompt.size(1) + max_new_tokens > model.config.context_length:
        raise ValueError("requested generation exceeds configured context_length")
    model.eval()
    tokens = prompt.clone()
    if tokens.device != next(model.parameters()).device:
        raise ValueError("prompt and model must be on the same device")
    if tokens.size(1) + max_new_tokens > model.config.context_length:
        raise ValueError("generation exceeds configured context_length")
    if tokens.is_cuda:
        torch.cuda.reset_peak_memory_stats(tokens.device)
        torch.cuda.synchronize(tokens.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, _, _ = model(tokens)
            next_logits = logits[:, -1]
            next_token = sample_next_token(next_logits, temperature, top_k, top_p, generator) if do_sample else greedy_next_token(next_logits)
            tokens = torch.cat((tokens, next_token), dim=1)
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)
    seconds = time.perf_counter() - start
    return GenerationResult(tokens, seconds, max_new_tokens / max(seconds, 1e-12), _peak_memory(tokens))


def generate_cached(
    model: CompactV3Model,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    do_sample: bool = False,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError("prompt must have shape (batch, sequence) with a non-empty sequence")
    if max_new_tokens < 0 or prompt.size(1) + max_new_tokens > model.config.context_length:
        raise ValueError("requested generation exceeds configured context_length")
    model.eval()
    if prompt.device != next(model.parameters()).device:
        raise ValueError("prompt and model must be on the same device")
    if prompt.size(1) + max_new_tokens > model.config.context_length:
        raise ValueError("generation exceeds configured context_length")
    if prompt.is_cuda:
        torch.cuda.reset_peak_memory_stats(prompt.device)
        torch.cuda.synchronize(prompt.device)
    start = time.perf_counter()
    tokens = prompt.clone()
    with torch.inference_mode():
        logits, caches = model.prefill(tokens)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1]
            next_token = sample_next_token(next_logits, temperature, top_k, top_p, generator) if do_sample else greedy_next_token(next_logits)
            tokens = torch.cat((tokens, next_token), dim=1)
            logits, caches = model.decode(next_token, caches)
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)
    seconds = time.perf_counter() - start
    return GenerationResult(tokens, seconds, max_new_tokens / max(seconds, 1e-12), _peak_memory(tokens))


def _peak_memory(tokens: Tensor) -> float | None:
    if not tokens.is_cuda:
        return None
    return torch.cuda.max_memory_allocated(tokens.device) / 1024**2


def cache_shapes(caches: list[BlockCache]) -> list[dict[str, tuple[int, ...]]]:
    return [
        {
            "compressed_kv": tuple(cache.mla.compressed_kv.shape),
            "rope_keys": tuple(cache.mla.rope_keys.shape),
        }
        for cache in caches
    ]


__all__ = [
    "GenerationResult",
    "cache_shapes",
    "generate_cached",
    "generate_uncached",
    "greedy_next_token",
    "sample_next_token",
]
