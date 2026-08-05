from pathlib import Path

import pytest
import torch

from compact_v3.model import CompactV3Model
from compact_v3.config import CompactV3Config
from compact_v3.training import (
    SyntheticBatchProvider,
    TrainingConfig,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    seed_everything,
    train_steps,
)


def model_config() -> CompactV3Config:
    return CompactV3Config(
        vocab_size=32,
        context_length=16,
        n_layer=1,
        d_model=16,
        n_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        n_routed_experts=2,
        n_shared_experts=1,
        top_k=1,
        expert_hidden_dim=8,
        mtp_depth=0,
    )


def train_config() -> TrainingConfig:
    return TrainingConfig(total_steps=5, warmup_steps=1, gradient_accumulation_steps=2, weight_decay=0.0)


def make_training_objects(seed: int):
    seed_everything(seed)
    model = CompactV3Model(model_config())
    optimizer = make_optimizer(model, train_config())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    provider = SyntheticBatchProvider(32, 1, 8, seed=seed + 1)
    return model, optimizer, scaler, provider


def mtp_model_config() -> CompactV3Config:
    return CompactV3Config(
        **{
            **model_config().__dict__,
            "mtp_depth": 1,
            "mtp_weight": 0.3,
            "mtp_weight_final": 0.1,
            "mtp_decay_step_fraction": 0.4,
        }
    )


def test_mtp_weight_anneals_partway_through_training() -> None:
    seed_everything(20)
    model = CompactV3Model(mtp_model_config())
    optimizer = make_optimizer(model, train_config())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    provider = SyntheticBatchProvider(32, 1, 8, seed=21)
    history = train_steps(model, optimizer, scaler, provider, train_config(), torch.device("cpu"), steps=5)
    weights = [item["mtp_weight"] for item in history]
    assert weights == [0.3, 0.3, 0.1, 0.1, 0.1]


def test_tiny_training_updates_parameters() -> None:
    model, optimizer, scaler, provider = make_training_objects(10)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history = train_steps(model, optimizer, scaler, provider, train_config(), torch.device("cpu"), steps=2)
    assert len(history) == 2
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.isfinite(torch.tensor(item["combined_loss"])) for item in history)


def test_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    full_model, full_optimizer, full_scaler, full_provider = make_training_objects(20)
    full_history = train_steps(
        full_model, full_optimizer, full_scaler, full_provider, train_config(), torch.device("cpu"), steps=5
    )

    split_model, split_optimizer, split_scaler, split_provider = make_training_objects(20)
    first_history = train_steps(
        split_model, split_optimizer, split_scaler, split_provider, train_config(), torch.device("cpu"), steps=2
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint, split_model, split_optimizer, split_scaler, 2, 2 * 2 * 8,
        train_config(), split_provider, first_history[-1],
    )
    resumed_model, resumed_optimizer, resumed_scaler, resumed_provider = make_training_objects(999)
    payload = load_checkpoint(
        checkpoint, resumed_model, resumed_optimizer, resumed_scaler, resumed_provider, torch.device("cpu")
    )
    assert payload["step"] == 2
    train_steps(
        resumed_model, resumed_optimizer, resumed_scaler, resumed_provider,
        train_config(), torch.device("cpu"), steps=3, start_step=2,
    )
    for name, value in full_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[name]), name
    assert full_provider.batches_drawn == resumed_provider.batches_drawn
    for key, state in full_optimizer.state_dict()["state"].items():
        for field, value in state.items():
            assert torch.equal(value, resumed_optimizer.state_dict()["state"][key][field])


def test_checkpoint_rejects_model_configuration_mismatch(tmp_path: Path) -> None:
    model, optimizer, scaler, provider = make_training_objects(30)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, scaler, 0, 0, train_config(), provider, {})
    other = CompactV3Model(CompactV3Config(**{**model_config().__dict__, "vocab_size": 33}))
    other_optimizer = make_optimizer(other, train_config())
    other_scaler = torch.amp.GradScaler("cuda", enabled=False)
    other_provider = SyntheticBatchProvider(33, 1, 8, seed=31)
    try:
        load_checkpoint(checkpoint, other, other_optimizer, other_scaler, other_provider, torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("configuration mismatch was accepted")


def test_cuda_training_step() -> None:
    if not torch.cuda.is_available():
        return
    model, optimizer, scaler, provider = make_training_objects(40)
    model.cuda()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = train_steps(model, optimizer, scaler, provider, train_config(), torch.device("cuda"), steps=1)
    assert torch.isfinite(torch.tensor(history[0]["combined_loss"]))
    torch.cuda.synchronize()


def test_checkpoint_write_is_atomic(tmp_path) -> None:
    """Gate U: a crash mid-write must not destroy the existing checkpoint.

    torch.save straight to the destination truncates it immediately, so a
    failure part-way through a multi-hour run left an unloadable file and no
    way back. The write now goes to a sibling temp file and is renamed in.
    """
    import compact_v3.training as training_module

    config = model_config()
    model = CompactV3Model(config)
    training_config = train_config()
    optimizer = make_optimizer(model, training_config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    provider = SyntheticBatchProvider(config.vocab_size, 1, 8, seed=0)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, scaler, 1, 8, training_config, provider, {"loss": 1.0})
    good = path.read_bytes()

    real_save = torch.save

    def exploding_save(payload, target, *args, **kwargs):
        real_save(payload, target, *args, **kwargs)
        raise RuntimeError("simulated crash mid-write")

    training_module.torch.save = exploding_save
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            save_checkpoint(path, model, optimizer, scaler, 2, 16, training_config, provider, {"loss": 0.5})
    finally:
        training_module.torch.save = real_save

    assert path.read_bytes() == good, "existing checkpoint was damaged by a failed write"
    assert not (tmp_path / "ckpt.pt.tmp").exists(), "temp file left behind after a failed write"
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    assert reloaded["step"] == 1


def test_make_optimizer_falls_back_off_cuda() -> None:
    """fused AdamW needs CUDA float params; a CPU model must still build an optimizer."""
    model = CompactV3Model(model_config())
    optimizer = make_optimizer(model, train_config())
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["fused"] in (False, None)
    # and it must actually be able to step
    loss = model(torch.randint(0, 32, (1, 8)), torch.randint(0, 32, (1, 8)))[1]
    loss.backward()
    optimizer.step()


def test_make_optimizer_uses_fused_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CompactV3Model(model_config()).cuda()
    optimizer = make_optimizer(model, train_config())
    assert optimizer.param_groups[0]["fused"] is True
