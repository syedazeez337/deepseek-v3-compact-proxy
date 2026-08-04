import pytest
import torch

from mla import MLACache, MultiHeadLatentAttention
from v3_config import CompactV3Config
from v3_rope import RotaryEmbedding


@pytest.fixture
def config() -> CompactV3Config:
    return CompactV3Config(
        vocab_size=64,
        context_length=32,
        d_model=32,
        n_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )


def test_config_cache_accounting(config: CompactV3Config) -> None:
    assert config.cache_values_per_token() == 12
    assert config.full_kv_values_per_token() == 40


def test_reference_and_prefill_match(config: CompactV3Config) -> None:
    torch.manual_seed(1)
    mla = MultiHeadLatentAttention(config).eval()
    x = torch.randn(2, 7, config.d_model)
    reference = mla.reference(x)
    prefill, cache = mla.prefill(x)
    assert cache.compressed_kv.shape == (2, 7, config.kv_lora_rank)
    assert cache.rope_keys.shape == (2, 7, config.qk_rope_head_dim)
    assert torch.allclose(reference, prefill, atol=1e-5, rtol=1e-5)


def test_cached_decode_matches_full_recomputation(config: CompactV3Config) -> None:
    torch.manual_seed(2)
    mla = MultiHeadLatentAttention(config).eval()
    prompt = torch.randn(1, 5, config.d_model)
    next_token = torch.randn(1, 1, config.d_model)
    _, cache = mla.prefill(prompt)
    decoded, next_cache = mla.decode(next_token, cache)
    full = mla.reference(torch.cat((prompt, next_token), dim=1))[:, -1:]
    assert torch.allclose(decoded, full, atol=1e-5, rtol=1e-5)
    assert next_cache.sequence_length == 6


def test_absorbed_decode_matches_naive_decode(config: CompactV3Config) -> None:
    torch.manual_seed(6)
    mla = MultiHeadLatentAttention(config).eval()
    prompt = torch.randn(1, 5, config.d_model)
    next_token = torch.randn(1, 1, config.d_model)
    _, cache = mla.prefill(prompt)
    absorbed_out, absorbed_cache = mla.decode(next_token, cache)
    naive_out, naive_cache = mla.decode_naive(next_token, cache)
    assert torch.allclose(absorbed_out, naive_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(absorbed_cache.compressed_kv, naive_cache.compressed_kv, atol=1e-6)
    assert torch.allclose(absorbed_cache.rope_keys, naive_cache.rope_keys, atol=1e-6)


def test_absorbed_decode_matches_naive_over_multiple_steps(config: CompactV3Config) -> None:
    torch.manual_seed(7)
    mla = MultiHeadLatentAttention(config).eval()
    prompt = torch.randn(1, 4, config.d_model)
    _, absorbed_cache = mla.prefill(prompt)
    _, naive_cache = mla.prefill(prompt)
    for _ in range(4):
        token = torch.randn(1, 1, config.d_model)
        absorbed_out, absorbed_cache = mla.decode(token, absorbed_cache)
        naive_out, naive_cache = mla.decode_naive(token, naive_cache)
        assert torch.allclose(absorbed_out, naive_out, atol=1e-5, rtol=1e-5)


def test_causality_is_preserved(config: CompactV3Config) -> None:
    torch.manual_seed(3)
    mla = MultiHeadLatentAttention(config).eval()
    first = torch.randn(1, 6, config.d_model)
    second = first.clone()
    second[:, -1] += 10.0
    first_output = mla.reference(first)
    second_output = mla.reference(second)
    assert torch.allclose(first_output[:, :-1], second_output[:, :-1], atol=1e-5, rtol=1e-5)


def test_cache_report_shows_compression(config: CompactV3Config) -> None:
    mla = MultiHeadLatentAttention(config)
    report = mla.cache_report(batch_size=2, sequence_length=16)
    assert report["mla_values"] < report["full_kv_values"]


def test_cuda_forward_when_available(config: CompactV3Config) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    mla = MultiHeadLatentAttention(config).cuda().half().eval()
    x = torch.randn(1, 8, config.d_model, device="cuda", dtype=torch.float16)
    output, cache = mla.prefill(x)
    assert output.is_cuda and output.dtype == torch.float16
    assert isinstance(cache, MLACache)
    torch.cuda.synchronize()
