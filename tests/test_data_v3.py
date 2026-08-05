import json

import torch

from compact_v3.data import DataConfig, PackedTokenProvider, _encode_documents, _pack, _train_tokenizer, load_tokenizer


def test_tokenizer_and_packing_contract(tmp_path) -> None:
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    path = tmp_path / "tokenizer.json"
    tokenizer = _train_tokenizer(texts, DataConfig(vocab_size=32), path)
    tokens = _encode_documents(tokenizer, texts)
    packed = _pack(tokens, context_length=4)
    assert path.exists()
    assert packed.numel() % 5 == 0
    assert tokenizer.token_to_id("[EOS]") in tokens.tolist()


def test_tokenizer_decode_round_trips_to_readable_text(tmp_path) -> None:
    texts = ["the cat sat on the mat", "the dog ran in the park"]
    path = tmp_path / "tokenizer.json"
    tokenizer = _train_tokenizer(texts, DataConfig(vocab_size=64), path)
    ids = tokenizer.encode("the cat sat on the mat").ids
    assert tokenizer.decode(ids) == "the cat sat on the mat"
    reloaded = load_tokenizer(path)
    assert reloaded.decode(ids) == "the cat sat on the mat"


def test_packed_provider_shapes_and_resume() -> None:
    tokens = torch.arange(100)
    first = PackedTokenProvider(tokens, batch_size=2, context_length=8, seed=42)
    a_inputs, a_targets = first.next(torch.device("cpu"))
    state = first.state_dict()
    b_inputs, b_targets = first.next(torch.device("cpu"))
    second = PackedTokenProvider(tokens, batch_size=2, context_length=8, seed=999)
    second.load_state_dict(state)
    c_inputs, c_targets = second.next(torch.device("cpu"))
    assert a_inputs.shape == (2, 8)
    assert torch.equal(a_targets, a_inputs + 1)
    assert torch.equal(b_inputs, c_inputs)
    assert torch.equal(b_targets, c_targets)


def test_metadata_is_json_serializable() -> None:
    config = DataConfig()
    assert json.dumps(config.__dict__)


def test_evaluate_tokens_is_deterministic() -> None:
    """Gate U: repeated evaluation of one unchanged model must agree exactly.

    evaluate_provider draws random windows with replacement and advances its
    generator, so four consecutive evaluations of a fixed checkpoint measured
    perplexity 278.20, 295.46, 318.99 and 279.34 - a 14.7% spread wider than
    the differences several gates were drawing conclusions from.
    """
    from compact_v3.config import CompactV3Config
    from compact_v3.model import CompactV3Model
    from compact_v3.data import evaluate_provider, evaluate_tokens

    config = CompactV3Config(
        vocab_size=64, context_length=24, n_layer=1, d_model=16, n_heads=2,
        q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4,
        v_head_dim=4, n_routed_experts=2, n_shared_experts=1, top_k=1,
        expert_hidden_dim=8, mtp_depth=0,
    )
    torch.manual_seed(0)
    model = CompactV3Model(config).eval()
    tokens = torch.randint(64, (4000,))
    device = torch.device("cpu")

    runs = [evaluate_tokens(model, tokens, 2, 16, device, max_batches=5) for _ in range(4)]
    assert all(r["main_loss"] == runs[0]["main_loss"] for r in runs), (
        f"deterministic evaluation disagreed across runs: {[r['main_loss'] for r in runs]}"
    )
    assert runs[0]["batches"] == 5

    # The sampling-based path is what motivated this; confirm it does drift.
    provider = PackedTokenProvider(tokens, 2, 16, seed=7)
    sampled = [evaluate_provider(model, provider, 5, device)["main_loss"] for _ in range(4)]
    assert len(set(sampled)) > 1, "expected the random-window evaluator to vary between calls"


def test_evaluate_tokens_covers_whole_split_when_unbounded() -> None:
    from compact_v3.config import CompactV3Config
    from compact_v3.model import CompactV3Model
    from compact_v3.data import evaluate_tokens

    config = CompactV3Config(
        vocab_size=64, context_length=24, n_layer=1, d_model=16, n_heads=2,
        q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4,
        v_head_dim=4, n_routed_experts=2, n_shared_experts=1, top_k=1,
        expert_hidden_dim=8, mtp_depth=0,
    )
    model = CompactV3Model(config).eval()
    tokens = torch.randint(64, (17 * 17,))
    result = evaluate_tokens(model, tokens, 4, 16, torch.device("cpu"))
    assert result["batches"] == 5  # ceil(17 windows / batch 4)
    assert result["windows"] == 17
