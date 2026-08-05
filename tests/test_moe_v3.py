import pytest
import torch

from compact_v3.experts import SwiGLUExpert
from compact_v3.moe import DeepSeekMoE
from compact_v3.config import CompactV3Config


def make_config(top_k: int = 1) -> CompactV3Config:
    return CompactV3Config(
        d_model=16,
        n_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=top_k,
        expert_hidden_dim=12,
    )


def test_swiglu_expert_shape() -> None:
    expert = SwiGLUExpert(8, 12)
    output = expert(torch.randn(2, 5, 8))
    assert output.shape == (2, 5, 8)


def test_shared_and_routed_moe_conserves_assignments() -> None:
    torch.manual_seed(1)
    moe = DeepSeekMoE(make_config(top_k=2))
    x = torch.randn(2, 3, 16)
    output, routing, balance_loss = moe(x)
    assert output.shape == x.shape
    assert int(routing.expert_load.sum()) == 2 * 3 * 2
    assert torch.isfinite(balance_loss)
    assert moe.parameter_report()["shared_parameters"] > 0


def test_shared_expert_is_active() -> None:
    moe = DeepSeekMoE(make_config())
    x = torch.randn(1, 2, 16)
    output, _, _ = moe(x)
    shared_only = moe.shared_experts[0](x.reshape(-1, 16)).reshape_as(x)
    assert output.shape == x.shape
    assert not torch.allclose(output, torch.zeros_like(output))
    assert shared_only.abs().sum() > 0


def test_cuda_fp16_moe() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    moe = DeepSeekMoE(make_config(top_k=2)).cuda().half()
    x = torch.randn(2, 8, 16, device="cuda", dtype=torch.float16)
    output, routing, loss = moe(x, update_balancer=True)
    assert output.is_cuda and output.dtype == torch.float16
    assert torch.isfinite(loss)
    assert int(routing.expert_load.sum()) == 2 * 8 * 2
    torch.cuda.synchronize()


def test_top_k_two_is_not_top_one() -> None:
    moe = DeepSeekMoE(make_config(top_k=2))
    _, routing, _ = moe(torch.randn(1, 4, 16))
    assert routing.selected_indices.size(-1) == 2
    assert routing.assignments == 8
