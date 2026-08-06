from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    validation_split: str = "validation"
    vocab_size: int = 32_000
    context_length: int = 256
    batch_size: int = 1
    seed: int = 1337
    cache_dir: str = "data_v3"


@dataclass
class PackedCorpus:
    train_tokens: torch.Tensor
    validation_tokens: torch.Tensor
    tokenizer_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _dataset_text(dataset) -> list[str]:
    return [text for text in dataset["text"] if isinstance(text, str) and text.strip()]


def _train_tokenizer(texts: list[str], config: DataConfig, path: Path) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[EOS]"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(path))
    return tokenizer


def _encode_documents(tokenizer: Tokenizer, texts: list[str]) -> torch.Tensor:
    eos_id = tokenizer.token_to_id("[EOS]")
    encoded: list[int] = []
    for text in texts:
        encoded.extend(tokenizer.encode(text).ids)
        encoded.append(eos_id)
    return torch.tensor(encoded, dtype=torch.long)


def _pack(tokens: torch.Tensor, context_length: int) -> torch.Tensor:
    usable = (tokens.numel() // (context_length + 1)) * (context_length + 1)
    if usable == 0:
        raise ValueError("corpus does not contain enough tokens for one causal block")
    return tokens[:usable].contiguous()


def prepare_wikitext2(config: DataConfig, force: bool = False) -> PackedCorpus:
    root = Path(config.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_path = root / "tokenizer.json"
    metadata_path = root / "metadata.json"
    train_path = root / "train_tokens.pt"
    validation_path = root / "validation_tokens.pt"
    if not force and all(path.exists() for path in (tokenizer_path, metadata_path, train_path, validation_path)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return PackedCorpus(torch.load(train_path, weights_only=True), torch.load(validation_path, weights_only=True), tokenizer_path, metadata_path, metadata)

    dataset = load_dataset(config.dataset_id, config.dataset_config)
    train_texts = _dataset_text(dataset[config.train_split])
    validation_texts = _dataset_text(dataset[config.validation_split])
    tokenizer = _train_tokenizer(train_texts, config, tokenizer_path)
    train_tokens = _pack(_encode_documents(tokenizer, train_texts), config.context_length)
    validation_tokens = _pack(_encode_documents(tokenizer, validation_texts), config.context_length)
    torch.save(train_tokens, train_path)
    torch.save(validation_tokens, validation_path)
    metadata = {
        "dataset": asdict(config),
        "license": "CC BY-SA 4.0 / GFDL as listed by Salesforce/wikitext dataset card",
        "train_documents": len(train_texts),
        "validation_documents": len(validation_texts),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "train_tokens": int(train_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
        "train_token_sha256": sha256_bytes(train_tokens.numpy().tobytes()),
        "validation_token_sha256": sha256_bytes(validation_tokens.numpy().tobytes()),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "preprocessing": "non-empty documents, tokenizer trained on train split only, EOS after each document, fixed causal blocks",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PackedCorpus(train_tokens, validation_tokens, tokenizer_path, metadata_path, metadata)


class PackedTokenProvider:
    def __init__(self, tokens: torch.Tensor, batch_size: int, context_length: int, seed: int = 1337) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.batches_drawn = 0
        self.max_start = tokens.numel() - context_length - 1
        if self.max_start < 1:
            raise ValueError("not enough packed tokens for requested context length")

    def next(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(self.max_start, (self.batch_size,), generator=self.generator)
        inputs = torch.stack([self.tokens[start : start + self.context_length] for start in starts])
        targets = torch.stack([self.tokens[start + 1 : start + self.context_length + 1] for start in starts])
        self.batches_drawn += 1
        return inputs.to(device), targets.to(device)

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state(), "batches_drawn": self.batches_drawn}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        generator_state = torch.as_tensor(state["generator_state"], dtype=torch.uint8, device="cpu").contiguous()
        self.generator.set_state(generator_state)
        self.batches_drawn = state["batches_drawn"]


def evaluate_tokens(
    model,
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Deterministic evaluation over fixed, non-overlapping windows.

    `PackedTokenProvider` draws random windows with replacement and advances its
    generator, so repeated evaluations of one unchanged model disagree: four
    consecutive runs measured perplexity 278.20, 295.46, 318.99, 279.34, a 14.7%
    spread that is pure sampling noise. That is larger than the differences
    several gates were drawing conclusions from. This walks the split in order
    instead, so the number depends only on the weights.
    """
    stride = context_length + 1
    usable = tokens.numel() // stride
    if usable < 1:
        raise ValueError("not enough validation tokens for one window")
    windows = tokens[: usable * stride].reshape(usable, stride)

    was_training = model.training
    model.eval()
    total_main = 0.0
    total_combined = 0.0
    counted = 0
    with torch.inference_mode():
        for start in range(0, usable, batch_size):
            if max_batches is not None and counted >= max_batches:
                break
            chunk = windows[start : start + batch_size]
            if chunk.size(0) == 0:
                break
            batch = chunk.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            _, loss, diagnostics = model(inputs, targets)
            combined = model.mtp.combined_loss(loss, diagnostics["mtp"])
            total_main += float(loss)
            total_combined += float(combined)
            counted += 1
    if was_training:
        model.train()
    return {
        "main_loss": total_main / counted,
        "combined_loss": total_combined / counted,
        "batches": counted,
        "windows": min(usable, counted * batch_size),
    }


def evaluate_provider(model, provider: PackedTokenProvider, batches: int, device: torch.device) -> dict[str, float]:
    if batches < 1:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            inputs, targets = provider.next(device)
            _, loss, diagnostics = model(inputs, targets)
            combined = model.mtp.combined_loss(loss, diagnostics["mtp"])
            losses.append({"main_loss": float(loss), "combined_loss": float(combined)})
    if was_training:
        model.train()
    return {
        "main_loss": sum(item["main_loss"] for item in losses) / len(losses),
        "combined_loss": sum(item["combined_loss"] for item in losses) / len(losses),
    }


def load_tokenizer(path: str | Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


TOKENIZER_FALLBACKS = (
    Path("checkpoints/tokenizer_wikitext103.json"),
    Path("data_v3_103/tokenizer.json"),
    Path("data_v3/tokenizer.json"),
)


def resolve_tokenizer_path(requested: str | Path | None, payload: dict) -> Path:
    """Pick the tokenizer a checkpoint was trained with.

    Decoding with the wrong tokenizer produces plausible-looking noise rather
    than an error, so a wrong default is worse than a missing file. The
    checkpoint records the path it was trained with; that is tried first, then
    the download location, then the corpus caches.
    """
    recorded = (payload.get("metrics") or {}).get("tokenizer_path")

    if requested is not None:
        requested = Path(requested)
        if requested.exists():
            return requested
        candidates = ([Path(recorded)] if recorded else []) + list(TOKENIZER_FALLBACKS)
        found = list(dict.fromkeys(str(p) for p in candidates if p.exists()))
        hint = f"  found instead: {', '.join(found)}" if found else (
            "  none found locally; download one with:\n"
            "    uv run huggingface-cli download syedazeez/deepseek-v3-compact-proxy "
            "tokenizer_wikitext103.json --local-dir checkpoints")
        raise FileNotFoundError(f"--tokenizer {requested} does not exist.\n{hint}")

    if recorded and Path(recorded).exists():
        return Path(recorded)
    for candidate in TOKENIZER_FALLBACKS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no tokenizer found. The checkpoint records {recorded!r}, which is not on this machine.\n"
        "  download it with:\n"
        "    uv run huggingface-cli download syedazeez/deepseek-v3-compact-proxy "
        "tokenizer_wikitext103.json --local-dir checkpoints\n"
        "  or point at your own with --tokenizer")


__all__ = ["DataConfig", "PackedCorpus", "PackedTokenProvider", "evaluate_provider", "evaluate_tokens", "load_tokenizer", "prepare_wikitext2", "resolve_tokenizer_path"]
