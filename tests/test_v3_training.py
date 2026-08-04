from pathlib import Path

import torch

from compact_v3_model import CompactV3Model
from v3_config import CompactV3Config
from v3_training import (
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
