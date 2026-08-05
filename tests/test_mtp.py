import pytest
import torch

from compact_v3.model import CompactV3Model
from compact_v3.mtp import (
    MTPObjective,
    align_hidden_states,
    make_future_targets,
    make_mtp_input_tokens,
    mtp_weight_schedule,
)
from compact_v3.config import CompactV3Config


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


def test_mtp_input_token_is_one_step_before_target() -> None:
    """Gate U: the module must never be handed the token it is scored against."""
    tokens = torch.arange(10).view(1, 10)
    inputs = make_mtp_input_tokens(tokens, horizon=2)
    targets = make_future_targets(tokens, horizon=2)
    assert inputs.shape == targets.shape
    assert torch.equal(inputs, tokens[:, 1:-1])
    assert torch.equal(targets, inputs + 1)
    assert not torch.equal(inputs, targets)


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
        model.mtp(torch.randn(1, 5, 32), torch.randint(64, (1, 5)), torch.randint(64, (1, 4)))
    with pytest.raises(ValueError):
        model.mtp(torch.randn(1, 5, 32), torch.randint(64, (1, 4)), torch.randint(64, (1, 5)))


def test_model_never_feeds_the_target_token_to_mtp() -> None:
    """Gate U regression, stated directly: input tokens must not be the targets.

    The bug was in this wiring, not in the module. `model.forward` passed
    token_ids[:, 2:] as both the embedding input and the target, so the module
    could read the answer off its own input through the tied output head.
    """
    model = CompactV3Model(small_config())
    tokens = torch.randint(64, (2, 12))
    seen = {}
    original = model.mtp.forward

    def spy(hidden_states, input_tokens=None, future_targets=None):
        seen["inputs"] = input_tokens
        seen["targets"] = future_targets
        return original(hidden_states, input_tokens, future_targets)

    model.mtp.forward = spy
    model(tokens, tokens)

    assert seen["inputs"] is not None and seen["targets"] is not None
    assert seen["inputs"].shape == seen["targets"].shape
    assert not torch.equal(seen["inputs"], seen["targets"])
    assert torch.equal(seen["inputs"], tokens[:, 1:-1])
    assert torch.equal(seen["targets"], tokens[:, 2:])


def test_mtp_prediction_depends_on_hidden_state_not_just_input_token() -> None:
    """If predictions ignore the hidden state, the head cannot be a real draft head."""
    torch.manual_seed(1)
    model = CompactV3Model(small_config()).eval()
    tokens = torch.randint(64, (2, 20))
    with torch.no_grad():
        _, _, diagnostics = model(tokens, tokens)
        baseline = diagnostics["mtp"].logits.clone()

        hidden = align_hidden_states(model.final_norm(model.token_embedding(tokens)), horizon=2)
        perturbed = model.mtp(
            hidden + 5.0,
            make_mtp_input_tokens(tokens, horizon=2),
            make_future_targets(tokens, horizon=2),
        ).logits
    assert not torch.allclose(baseline, perturbed, atol=1e-4)


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
