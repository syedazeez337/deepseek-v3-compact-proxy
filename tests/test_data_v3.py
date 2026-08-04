import json

import torch

from data_v3 import DataConfig, PackedTokenProvider, _encode_documents, _pack, _train_tokenizer


def test_tokenizer_and_packing_contract(tmp_path) -> None:
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    path = tmp_path / "tokenizer.json"
    tokenizer = _train_tokenizer(texts, DataConfig(vocab_size=32), path)
    tokens = _encode_documents(tokenizer, texts)
    packed = _pack(tokens, context_length=4)
    assert path.exists()
    assert packed.numel() % 5 == 0
    assert tokenizer.token_to_id("[EOS]") in tokens.tolist()


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
