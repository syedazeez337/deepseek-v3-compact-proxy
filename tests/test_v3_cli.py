import subprocess
import sys
from pathlib import Path

import torch

from compact_v3_model import CompactV3Model
from data_v3 import _train_tokenizer, DataConfig
from v3_cli import build_parser, make_config
from v3_config import CompactV3Config
from v3_training import SyntheticBatchProvider, TrainingConfig, make_optimizer, save_checkpoint


def test_synthetic_lifecycle_cli(tmp_path: Path) -> None:
    checkpoint = tmp_path / "compact_v3.pt"
    command = [
        sys.executable,
        "v3_cli.py",
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--sequence-length",
        "8",
        "--generate",
        "2",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert checkpoint.exists()
    assert '"loaded_step": 1' in completed.stdout
    assert '"generated_tokens"' in completed.stdout


def test_cli_help() -> None:
    completed = subprocess.run([sys.executable, "v3_cli.py", "--help"], capture_output=True, text=True, check=True)
    assert "synthetic train-resume-generate lifecycle" in completed.stdout
    assert "--real-corpus" in completed.stdout


def test_cli_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "compact_v3.pt"
    base = [sys.executable, "v3_cli.py", "--steps", "1", "--batch-size", "1", "--sequence-length", "8", "--generate", "1", "--device", "cpu", "--checkpoint", str(checkpoint)]
    subprocess.run(base, capture_output=True, text=True, check=True)
    resumed = subprocess.run([*base, "--resume", "--steps", "2"], capture_output=True, text=True, check=True)
    assert '"resumed_from"' in resumed.stdout
    assert '"loaded_step": 2' in resumed.stdout


def test_cli_progress_reports_routing_entropy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "compact_v3.pt"
    command = [
        sys.executable,
        "v3_cli.py",
        "--steps",
        "2",
        "--checkpoint-every",
        "1",
        "--batch-size",
        "1",
        "--sequence-length",
        "8",
        "--generate",
        "1",
        "--device",
        "cpu",
        "--checkpoint",
        str(checkpoint),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert '"load_entropy_by_layer"' in completed.stdout
    assert '"expert_loads"' in completed.stdout


def test_bias_update_rate_flag_wires_into_config() -> None:
    args = build_parser().parse_args(["--bias-update-rate", "0.01"])
    assert args.bias_update_rate == 0.01
    config = make_config(args.sequence_length, args.bias_update_rate)
    assert config.router_bias_update_rate == 0.01


def test_top_k_flag_wires_into_config() -> None:
    args = build_parser().parse_args(["--top-k", "2"])
    assert args.top_k == 2
    config = make_config(args.sequence_length, args.bias_update_rate, args.top_k)
    assert config.top_k == 2


def test_n_dense_layers_flag_wires_into_config() -> None:
    args = build_parser().parse_args(["--n-dense-layers", "1"])
    assert args.n_dense_layers == 1
    config = make_config(args.sequence_length, args.bias_update_rate, args.top_k, args.n_dense_layers)
    assert config.n_dense_layers == 1


def test_route_scale_flag_wires_into_config() -> None:
    args = build_parser().parse_args(["--route-scale", "2.5"])
    assert args.route_scale == 2.5
    config = make_config(
        args.sequence_length, args.bias_update_rate, args.top_k, args.n_dense_layers, args.route_scale
    )
    assert config.route_scale == 2.5


def test_mtp_annealing_flags_wire_into_config() -> None:
    args = build_parser().parse_args(["--mtp-weight-final", "0.05", "--mtp-decay-fraction", "0.5"])
    assert args.mtp_weight_final == 0.05
    assert args.mtp_decay_fraction == 0.5
    config = make_config(
        args.sequence_length,
        args.bias_update_rate,
        args.top_k,
        args.n_dense_layers,
        args.route_scale,
        args.mtp_weight_final,
        args.mtp_decay_fraction,
    )
    assert config.mtp_weight_final == 0.05
    assert config.mtp_decay_step_fraction == 0.5


def test_dataset_config_flags_default_and_override() -> None:
    default_args = build_parser().parse_args([])
    assert default_args.dataset_config == "wikitext-2-raw-v1"
    assert default_args.dataset_cache_dir == "data_v3"
    overridden = build_parser().parse_args(["--dataset-config", "wikitext-103-raw-v1", "--dataset-cache-dir", "data_v3_103"])
    assert overridden.dataset_config == "wikitext-103-raw-v1"
    assert overridden.dataset_cache_dir == "data_v3_103"


def test_real_corpus_cli_requires_no_synthetic_provider(tmp_path: Path) -> None:
    source = Path("v3_cli.py").read_text(encoding="utf-8")
    assert "PackedTokenProvider(corpus.train_tokens" in source
    assert "if args.real_corpus:" in source


def test_complete_cli_produces_readable_text(tmp_path: Path) -> None:
    texts = ["the cat sat on the mat", "the dog ran in the park", "a bird flew over the tree"]
    tokenizer_path = tmp_path / "tokenizer.json"
    _train_tokenizer(texts, DataConfig(vocab_size=64), tokenizer_path)

    config = CompactV3Config(
        vocab_size=64, context_length=32, n_layer=1, d_model=16, n_heads=2,
        q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
        n_routed_experts=2, n_shared_experts=1, top_k=1, expert_hidden_dim=8, mtp_depth=0,
    )
    model = CompactV3Model(config)
    optimizer = make_optimizer(model, TrainingConfig(total_steps=1))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    provider = SyntheticBatchProvider(config.vocab_size, 1, 8, seed=1)
    checkpoint_path = tmp_path / "tiny.pt"
    save_checkpoint(checkpoint_path, model, optimizer, scaler, 0, 0, TrainingConfig(total_steps=1), provider, {})

    command = [
        sys.executable, "complete.py", "the cat sat on",
        "--checkpoint", str(checkpoint_path),
        "--tokenizer", str(tokenizer_path),
        "--max-new-tokens", "4",
        "--device", "cpu",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert '"completion"' in completed.stdout
    assert '"continuation_only"' in completed.stdout
