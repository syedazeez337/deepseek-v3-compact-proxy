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


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_sorted_dispatch_matches_reference(top_k: int) -> None:
    """Gate T: batched dispatch must be exactly equivalent to the old loop."""
    torch.manual_seed(3)
    moe = DeepSeekMoE(make_config(top_k=top_k)).eval()
    x = torch.randn(4, 7, 16)
    fast, fast_routing, fast_loss = moe(x)
    reference, reference_routing, reference_loss = moe.forward_reference(x)
    assert torch.equal(fast, reference)
    assert torch.equal(fast_loss, reference_loss)
    assert torch.equal(fast_routing.selected_indices, reference_routing.selected_indices)
    assert torch.equal(fast_routing.expert_load, reference_routing.expert_load)


def test_sorted_dispatch_matches_reference_gradients() -> None:
    """Equal outputs are not enough; the backward pass must match too."""
    torch.manual_seed(5)
    config = make_config(top_k=2)
    x = torch.randn(3, 5, 16)

    grads = []
    for forward in ("forward", "forward_reference"):
        torch.manual_seed(11)
        moe = DeepSeekMoE(config)
        output, _, balance_loss = getattr(moe, forward)(x)
        (output.square().sum() + balance_loss).backward()
        grads.append([p.grad.clone() for p in moe.parameters() if p.grad is not None])

    assert len(grads[0]) == len(grads[1]) and len(grads[0]) > 0
    for fast, reference in zip(*grads):
        torch.testing.assert_close(fast, reference, rtol=0, atol=0)


def test_sorted_dispatch_matches_reference_on_cuda() -> None:
    """index_add_ on CUDA accumulates with atomics, so allow float tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(13)
    moe = DeepSeekMoE(make_config(top_k=2)).cuda().eval()
    x = torch.randn(8, 32, 16, device="cuda")
    fast, _, _ = moe(x)
    reference, _, _ = moe.forward_reference(x)
    torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-6)


def test_dispatch_handles_expert_receiving_no_tokens() -> None:
    """A collapsed router leaves most experts empty; those must be skipped."""
    torch.manual_seed(7)
    moe = DeepSeekMoE(make_config(top_k=1)).eval()
    with torch.no_grad():
        moe.router.projection.weight.zero_()
        moe.router.expert_bias.zero_()
        moe.router.expert_bias[2] = 10.0  # force every token onto expert 2
    x = torch.randn(2, 6, 16)
    output, routing, _ = moe(x)
    assert int(routing.expert_load[2]) == 12
    assert int(routing.expert_load.sum()) == 12
    assert torch.equal(output, moe.forward_reference(x)[0])
