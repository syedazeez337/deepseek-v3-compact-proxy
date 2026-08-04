import pytest
import torch

from compact_v3_model import CompactV3Model
from mtp import MTPObjective, align_hidden_states, make_future_targets, mtp_weight_schedule
from v3_config import CompactV3Config


def small_config(mtp_depth: int = 1) -> CompactV3Config:
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
        mtp_depth=mtp_depth,
        mtp_weight=0.3,
    )


def test_future_target_alignment() -> None:
    tokens = torch.arange(10).view(1, 10)
    hidden = torch.randn(1, 10, 8)
    assert torch.equal(make_future_targets(tokens), tokens[:, 2:])
    assert align_hidden_states(hidden, horizon=2).shape == (1, 8, 8)


def test_mtp_has_sequential_refinement_and_separate_loss() -> None:
    model = CompactV3Model(small_config())
    tokens = torch.randint(64, (1, 10))
    logits, main_loss, diagnostics = model(tokens, tokens)
    result = diagnostics["mtp"]
    assert logits.shape == (1, 10, 64)
    assert result is not None
    assert result.target_offset == 2
    assert result.logits.shape == (1, 8, 64)
    assert torch.isfinite(main_loss)
    assert torch.isfinite(result.loss)
    combined = model.mtp.combined_loss(main_loss, result)
    assert torch.allclose(combined, main_loss + 0.3 * result.loss)


def test_combined_loss_accepts_weight_override() -> None:
    model = CompactV3Model(small_config())
    tokens = torch.randint(64, (1, 10))
    _, main_loss, diagnostics = model(tokens, tokens)
    result = diagnostics["mtp"]
    combined = model.mtp.combined_loss(main_loss, result, weight=0.1)
    assert torch.allclose(combined, main_loss + 0.1 * result.loss)


def test_mtp_weight_schedule_switches_at_decay_fraction() -> None:
    assert mtp_weight_schedule(step=0, total_steps=100, initial_weight=0.3, final_weight=0.1, decay_step_fraction=0.6757) == 0.3
    assert mtp_weight_schedule(step=67, total_steps=100, initial_weight=0.3, final_weight=0.1, decay_step_fraction=0.6757) == 0.3
    assert mtp_weight_schedule(step=68, total_steps=100, initial_weight=0.3, final_weight=0.1, decay_step_fraction=0.6757) == 0.1
    assert mtp_weight_schedule(step=99, total_steps=100, initial_weight=0.3, final_weight=0.1, decay_step_fraction=0.6757) == 0.1


def test_mtp_disabled_preserves_base_path() -> None:
    model = CompactV3Model(small_config(mtp_depth=0))
    tokens = torch.randint(64, (1, 10))
    _, loss, diagnostics = model(tokens, tokens)
    assert torch.isfinite(loss)
    assert diagnostics["mtp"] is None


def test_mtp_and_shared_output_receive_gradients() -> None:
    model = CompactV3Model(small_config())
    tokens = torch.randint(64, (1, 10))
    _, main_loss, diagnostics = model(tokens, tokens)
    total = model.mtp.combined_loss(main_loss, diagnostics["mtp"])
    total.backward()
    assert model.mtp.merge.weight.grad is not None
    assert model.output.weight.grad is not None
    assert model.token_embedding.weight.data_ptr() == model.output.weight.data_ptr()


def test_mtp_direct_shape_validation() -> None:
    model = CompactV3Model(small_config())
    with pytest.raises(ValueError):
        model.mtp(torch.randn(1, 5, 32), torch.randint(64, (1, 4)))


def test_cuda_fp16_mtp() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CompactV3Model(small_config()).cuda().half()
    tokens = torch.randint(64, (1, 10), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, main_loss, diagnostics = model(tokens, tokens)
        total = model.mtp.combined_loss(main_loss, diagnostics["mtp"])
    total.backward()
    assert torch.isfinite(total)
    torch.cuda.synchronize()
