"""Everything needed to say where a number came from.

A checkpoint that records its metrics but not the code that produced them can be
compared against nothing. This collects the identifying facts once, so they can
be stamped into a run's config and into the checkpoint payload.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args), capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit hash does not describe the code that ran.
        # Recording the flag is the difference between a reproducible run and one
        # that merely looks reproducible.
        "dirty": bool(status) if status is not None else None,
    }


def corpus_state(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir) / "metadata.json"
    if not path.exists():
        return {"metadata_path": str(path), "present": False}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return {
        "metadata_path": str(path),
        "present": True,
        "train_token_sha256": metadata.get("train_token_sha256"),
        "validation_token_sha256": metadata.get("validation_token_sha256"),
        "tokenizer_sha256": metadata.get("tokenizer_sha256"),
        "train_tokens": metadata.get("train_tokens"),
    }


def provenance(cache_dir: str | Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "git": git_state(),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if cache_dir is not None:
        record["corpus"] = corpus_state(cache_dir)
    return record


if __name__ == "__main__":
    print(json.dumps(provenance("data_v3"), indent=2))
