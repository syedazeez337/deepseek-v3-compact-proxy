import pytest
import torch

from compact_v3.model import CompactV3Model
from compact_v3.config import CompactV3Config
from compact_v3.generation import cache_shapes, generate_cached, generate_uncached, sample_next_token


def config() -> CompactV3Config:
    return CompactV3Config(
        vocab_size=64,
        context_length=32,
        n_layer=2,
        d_model=32,
        n_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=1,
        expert_hidden_dim=16,
        mtp_depth=0,
    )


def test_cached_and_uncached_greedy_generation_match() -> None:
    torch.manual_seed(10)
    model = CompactV3Model(config()).eval()
    prompt = torch.randint(64, (1, 5))
    uncached = generate_uncached(model, prompt, 4)
    cached = generate_cached(model, prompt, 4)
    assert torch.equal(uncached.tokens, cached.tokens)
    assert uncached.tokens.shape == (1, 9)


def test_sampling_is_seeded_and_different_from_greedy() -> None:
    torch.manual_seed(11)
    model = CompactV3Model(config()).eval()
    prompt = torch.randint(64, (1, 4))
    generator_a = torch.Generator().manual_seed(22)
    generator_b = torch.Generator().manual_seed(22)
    first = generate_cached(model, prompt, 4, temperature=0.8, top_k=10, do_sample=True, generator=generator_a)
    second = generate_cached(model, prompt, 4, temperature=0.8, top_k=10, do_sample=True, generator=generator_b)
    assert torch.equal(first.tokens, second.tokens)


def test_sampling_validation() -> None:
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(1, 8), temperature=0)
    with pytest.raises(ValueError):
        sample_next_token(torch.randn(1, 8), top_p=0)


def test_cache_shapes() -> None:
    model = CompactV3Model(config()).eval()
    prompt = torch.randint(64, (1, 4))
    _, caches = model.prefill(prompt)
    shapes = cache_shapes(caches)
    assert shapes[0]["compressed_kv"] == (1, 4, 8)
    assert shapes[0]["rope_keys"] == (1, 4, 4)


def test_generation_context_bound() -> None:
    model = CompactV3Model(config()).eval()
    with pytest.raises(ValueError):
        generate_cached(model, torch.randint(64, (1, 30)), 3)


def test_cuda_cached_generation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CompactV3Model(config()).cuda().half().eval()
    prompt = torch.randint(64, (1, 5), device="cuda")
    result = generate_cached(model, prompt, 4)
    assert result.tokens.is_cuda
    assert result.peak_allocated_mb is not None
    torch.cuda.synchronize()
