"""A transformers-shaped face for CompactV3Model.

This is the keystone piece. `src/compact_v3/` stays a plain PyTorch library; this
wraps it so the ecosystem tooling that assumes `PreTrainedModel` works against
it: lm-evaluation-harness, TRL, `save_pretrained` with native safetensors and
correct tied-weight handling, and `push_to_hub`.

Nothing in `src/compact_v3/` is modified or imported by anything here in reverse.
The wrapper is a consumer of the library, not a layer inside it.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from compact_v3.config import CompactV3Config, config_from_checkpoint
from compact_v3.model import CompactV3Model

_FIELDS = tuple(f.name for f in fields(CompactV3Config))


class CompactV3HFConfig(PretrainedConfig):
    model_type = "compact_v3"

    def __init__(self, **kwargs):
        defaults = asdict(CompactV3Config())
        for name in _FIELDS:
            setattr(self, name, kwargs.pop(name, defaults[name]))
        # HFLM in lm-eval discovers the usable window through these names, not
        # through our `context_length`. Without them it falls back to 2048 and
        # every rolling-loglikelihood window overruns the model.
        kwargs.setdefault("max_position_embeddings", self.context_length)
        # Load-bearing, and silently so. `get_expanded_tied_weights_keys` reads
        # `getattr(config, "tie_word_embeddings", False)` and returns an empty
        # map when it is absent, discarding `_tied_weights_keys` entirely. But
        # save_pretrained drops the aliases through a different path that does
        # not consult the config. Omit this and the model saves smaller, loads
        # without error, and is wrong.
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)

    def to_compact(self) -> CompactV3Config:
        return CompactV3Config(**{name: getattr(self, name) for name in _FIELDS})


class CompactV3ForCausalLM(PreTrainedModel):
    config_class = CompactV3HFConfig
    base_model_prefix = "compact"
    # `output.weight` is `token_embedding.weight`, and MTPObjective holds both by
    # reference, so three names alias one storage. Declaring them lets
    # save_pretrained drop the aliases and from_pretrained restore them, which is
    # the thing my hand-written export had to solve manually.
    #
    # transformers 5.x expects a dict of alias -> source. The 4.x docs still
    # describe a list, and passing a list fails with an AttributeError inside
    # save_pretrained rather than anything that names the real problem.
    _tied_weights_keys = {
        "compact.output.weight": "compact.token_embedding.weight",
        "compact.mtp.token_embedding.weight": "compact.token_embedding.weight",
        "compact.mtp.output_head.weight": "compact.token_embedding.weight",
    }

    def __init__(self, config: CompactV3HFConfig) -> None:
        super().__init__(config)
        self.compact = CompactV3Model(config.to_compact())
        # post_init is what expands `_tied_weights_keys` into the
        # `all_tied_weights_keys` map that from_pretrained reads. Without it,
        # saving succeeds and loading fails on a missing attribute.
        self.post_init()

    def _init_weights(self, module) -> None:
        # CompactV3Model already initialises itself in its own constructor.
        # transformers calls this during post_init; leaving it as a no-op keeps
        # the library's initialisation authoritative.
        return

    def get_input_embeddings(self):
        return self.compact.token_embedding

    def set_input_embeddings(self, value):
        self.compact.token_embedding = value

    def get_output_embeddings(self):
        return self.compact.output

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        logits, loss, _ = self.compact(input_ids, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def rebuild_non_persistent_buffers(self) -> int:
        """Recompute the RoPE tables after a from_pretrained load.

        `rope.py` registers `cos` and `sin` with `persistent=False`, so they are
        absent from every state_dict and from the safetensors file. transformers
        builds the module on the meta device and materialises it, and because
        nothing in the checkpoint claims those names they are never filled. They
        come back as uninitialised memory, not zeros, and from_pretrained reports
        no missing key and no error.

        Every parameter loads correctly and the model is silently wrong.
        """
        from compact_v3.rope import RotaryEmbedding

        config = self.config.to_compact()
        rebuilt = 0
        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                fresh = RotaryEmbedding(config.qk_rope_head_dim, config.context_length, config.rope_base)
                module.cos = fresh.cos.to(module.cos.device, module.cos.dtype)
                module.sin = fresh.sin.to(module.sin.device, module.sin.dtype)
                rebuilt += 1
        return rebuilt

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.rebuild_non_persistent_buffers()
        return model

    @classmethod
    def from_compact_checkpoint(cls, path: str | Path) -> "CompactV3ForCausalLM":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        compact_config = config_from_checkpoint(payload["model_config"])
        model = cls(CompactV3HFConfig(**asdict(compact_config)))
        model.compact.load_state_dict(payload["model"])
        model.eval()
        return model


def tokenizer_from_json(path: str | Path, context_length: int) -> PreTrainedTokenizerFast:
    """Wrap the raw `tokenizers` JSON as a transformers tokenizer.

    The corpus tokenizer is a bare `tokenizers.Tokenizer`, which the ecosystem
    cannot consume. The ByteLevel decoder must be reattached here for the same
    reason Gate S needed it: without it, decode returns byte fragments.
    """
    backend = Tokenizer.from_file(str(path))
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    backend.decoder = ByteLevelDecoder()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[EOS]",
        model_max_length=context_length,
    )


CompactV3HFConfig.register_for_auto_class()
CompactV3ForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = ["CompactV3ForCausalLM", "CompactV3HFConfig", "tokenizer_from_json"]
