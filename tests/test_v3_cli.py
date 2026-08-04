import subprocess
import sys
from pathlib import Path

from v3_cli import build_parser, make_config


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


def test_real_corpus_cli_requires_no_synthetic_provider(tmp_path: Path) -> None:
    source = Path("v3_cli.py").read_text(encoding="utf-8")
    assert "PackedTokenProvider(corpus.train_tokens" in source
    assert "if args.real_corpus:" in source
