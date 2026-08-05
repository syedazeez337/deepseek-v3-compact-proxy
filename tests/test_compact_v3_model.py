import pytest
import torch

from compact_v3.model import CompactV3Model
from compact_v3.norms import RMSNorm
from compact_v3.config import CompactV3Config


def config(**overrides) -> CompactV3Config:
    values = dict(
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
    )
    values.update(overrides)
    return CompactV3Config(**values)


def test_model_forward_loss_and_tied_embeddings() -> None:
    model = CompactV3Model(config())
    tokens = torch.randint(0, 64, (2, 8))
    logits, loss, diagnostics = model(tokens, tokens)
    assert logits.shape == (2, 8, 64)
    assert loss is not None and torch.isfinite(loss)
    assert torch.isfinite(diagnostics["balance_loss"])
    assert model.tied_output_embedding()
    assert isinstance(model.final_norm, RMSNorm)


def test_model_backward() -> None:
    torch.manual_seed(5)
    model = CompactV3Model(config(n_dense_layers=0))
    tokens = torch.randint(0, 64, (1, 8))
    _, loss, diagnostics = model(tokens, tokens)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attention.q_down.weight.grad is not None
    assert model.blocks[0].moe.router.projection.weight.grad is not None
    assert model.blocks[0].moe.shared_experts[0].gate.weight.grad is not None
    selected = diagnostics["routing"][0].selected_indices.unique().tolist()
    for expert_id in selected:
        assert model.blocks[0].moe.routed_experts[expert_id].gate.weight.grad is not None


def test_model_prefill_decode_matches_full_logits() -> None:
    torch.manual_seed(4)
    model = CompactV3Model(config()).eval()
    prompt = torch.randint(0, 64, (1, 5))
    next_token = torch.randint(0, 64, (1, 1))
    _, caches = model.prefill(prompt)
    decoded, _ = model.decode(next_token, caches)
    full, _, _ = model(torch.cat((prompt, next_token), dim=1))
    assert torch.allclose(decoded, full[:, -1:], atol=1e-5, rtol=1e-5)


def test_dense_control_constructs() -> None:
    dense = CompactV3Model(CompactV3Config(**{**config().__dict__, "use_moe": False, "mtp_depth": 0}))
    tokens = torch.randint(0, 64, (1, 8))
    logits, loss, diagnostics = dense(tokens, tokens)
    assert logits.shape == (1, 8, 64)
    assert torch.isfinite(loss)
    assert diagnostics["routing"][0] is None


def test_n_dense_layers_prefix_uses_dense_ffn() -> None:
    model = CompactV3Model(config(n_layer=4, n_dense_layers=2))
    assert model.blocks[0].moe is None and model.blocks[0].dense_ffn is not None
    assert model.blocks[1].moe is None and model.blocks[1].dense_ffn is not None
    assert model.blocks[2].moe is not None and model.blocks[2].dense_ffn is None
    assert model.blocks[3].moe is not None and model.blocks[3].dense_ffn is None
    tokens = torch.randint(0, 64, (1, 8))
    logits, loss, diagnostics = model(tokens, tokens)
    assert logits.shape == (1, 8, 64)
    assert torch.isfinite(loss)
    assert diagnostics["routing"][0] is None and diagnostics["routing"][1] is None
    assert diagnostics["routing"][2] is not None and diagnostics["routing"][3] is not None


def test_n_dense_layers_matches_moe_active_compute() -> None:
    cfg = config(n_layer=2, n_dense_layers=1, n_routed_experts=4, n_shared_experts=1, top_k=2, expert_hidden_dim=16)
    model = CompactV3Model(cfg)
    expected_hidden_dim = (cfg.n_shared_experts + cfg.top_k) * cfg.expert_hidden_dim
    assert model.blocks[0].dense_ffn.gate.out_features == expected_hidden_dim
    assert model.blocks[0].dense_ffn.gate.out_features == 48


def test_n_dense_layers_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        config(n_dense_layers=5).validate()


def test_mtp_decay_step_fraction_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        config(mtp_decay_step_fraction=1.5).validate()


def test_mtp_weight_final_negative_rejected() -> None:
    with pytest.raises(ValueError):
        config(mtp_weight_final=-0.1).validate()


def test_configuration_b_constructs() -> None:
    model = CompactV3Model(
        CompactV3Config(
            vocab_size=128,
            context_length=64,
            n_layer=4,
            d_model=64,
            n_heads=4,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            n_routed_experts=8,
            n_shared_experts=1,
            top_k=2,
            expert_hidden_dim=32,
        )
    )
    assert model.parameter_report()["unique_parameters"] > 0


def test_context_bound() -> None:
    model = CompactV3Model(config(context_length=4))
    with pytest.raises(ValueError):
        model(torch.randint(0, 64, (1, 5)))


def test_cuda_fp16_model() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CompactV3Model(config()).cuda().half()
    tokens = torch.randint(0, 64, (1, 8), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, loss, _ = model(tokens, tokens)
    assert logits.is_cuda and torch.isfinite(loss)
    loss.backward()
    torch.cuda.synchronize()


def test_parameter_report_is_stable() -> None:
    c = config()
    first = CompactV3Model(c).parameter_report()
    second = CompactV3Model(c).parameter_report()
    assert first == second


def test_config_from_checkpoint_applies_pre_gate_defaults() -> None:
    """Gate U: a checkpoint saved before a field existed must not get today's default.

    Gates I-M recorded no n_dense_layers. Rebuilding with the current default
    (1) makes block 0 dense against stored MoE weights, so those checkpoints
    stopped loading entirely from Gate N onward.
    """
    from compact_v3.config import config_from_checkpoint

    legacy = {
        "vocab_size": 32000, "context_length": 256, "n_layer": 4, "d_model": 256,
        "n_heads": 8, "q_lora_rank": 64, "kv_lora_rank": 64, "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 16, "v_head_dim": 32, "rope_base": 10000.0,
        "rms_norm_eps": 1e-6, "use_moe": True, "n_routed_experts": 4,
        "n_shared_experts": 1, "top_k": 2, "expert_hidden_dim": 384,
        "route_scale": 1.0, "router_bias_update_rate": 1e-4,
        "sequence_balance_coefficient": 1e-4, "mtp_depth": 1, "mtp_weight": 0.3,
    }
    config = config_from_checkpoint(legacy)
    assert config.n_dense_layers == 0, "pre-Gate-N checkpoints had every layer as MoE"
    assert config.mtp_weight_final == config.mtp_weight, "annealing did not exist pre-Gate-Q"
    assert config.mtp_decay_step_fraction == 1.0, "a fraction of 1.0 never fires the switch"
    assert config.context_length == 256 and config.n_routed_experts == 4


def test_config_from_checkpoint_preserves_recorded_values() -> None:
    from dataclasses import asdict

    from compact_v3.config import CompactV3Config, config_from_checkpoint

    original = CompactV3Config(n_dense_layers=1, mtp_weight_final=0.1, mtp_decay_step_fraction=0.5)
    assert config_from_checkpoint(asdict(original)) == original


def test_config_from_checkpoint_ignores_unknown_fields() -> None:
    from dataclasses import asdict

    from compact_v3.config import CompactV3Config, config_from_checkpoint

    stored = asdict(CompactV3Config())
    stored["some_field_from_the_future"] = 123
    assert config_from_checkpoint(stored) == CompactV3Config()
