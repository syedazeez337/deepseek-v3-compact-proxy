from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from compact_v3.norms import RMSNorm
from compact_v3.rope import RotaryEmbedding
from compact_v3.config import CompactV3Config


@dataclass
class MLACache:
    compressed_kv: Tensor
    rope_keys: Tensor

    @property
    def sequence_length(self) -> int:
        return self.compressed_kv.size(1)

    @property
    def values_per_token(self) -> int:
        return self.compressed_kv.size(-1) + self.rope_keys.size(-1)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: CompactV3Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.q_down = nn.Linear(config.d_model, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_content_up = nn.Linear(config.q_lora_rank, config.q_content_dim, bias=False)
        self.q_rope = nn.Linear(config.q_lora_rank, config.q_rope_dim, bias=False)
        self.kv_down = nn.Linear(config.d_model, config.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.k_content_up = nn.Linear(config.kv_lora_rank, config.n_heads * config.qk_nope_head_dim, bias=False)
        self.v_up = nn.Linear(config.kv_lora_rank, config.value_dim, bias=False)
        self.k_rope = nn.Linear(config.d_model, config.qk_rope_head_dim, bias=False)
        self.output = nn.Linear(config.value_dim, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.qk_rope_head_dim, config.context_length, config.rope_base)

    def _query(self, x: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        batch, sequence_length, _ = x.shape
        compressed = self.q_norm(self.q_down(x))
        content = self.q_content_up(compressed).view(
            batch, sequence_length, self.config.n_heads, self.config.qk_nope_head_dim
        ).transpose(1, 2)
        positional = self.q_rope(compressed).view(
            batch, sequence_length, self.config.n_heads, self.config.qk_rope_head_dim
        ).transpose(1, 2)
        positional = self.rope(positional, positions)
        return content, positional

    def _kv(self, x: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, sequence_length, _ = x.shape
        compressed = self.kv_norm(self.kv_down(x))
        content = self.k_content_up(compressed).view(
            batch, sequence_length, self.config.n_heads, self.config.qk_nope_head_dim
        ).transpose(1, 2)
        values = self.v_up(compressed).view(
            batch, sequence_length, self.config.n_heads, self.config.v_head_dim
        ).transpose(1, 2)
        positional = self.k_rope(x).view(
            batch, sequence_length, 1, self.config.qk_rope_head_dim
        ).transpose(1, 2)
        positional = self.rope(positional, positions)
        return compressed, content, values, positional

    def reference(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        batch, sequence_length, _ = x.shape
        positions = positions if positions is not None else torch.arange(sequence_length, device=x.device)
        query_content, query_rope = self._query(x, positions)
        _, key_content, values, key_rope = self._kv(x, positions)
        query_rope = query_rope.expand(-1, -1, -1, -1)
        key_rope = key_rope.expand(batch, self.config.n_heads, -1, -1)
        scores = torch.matmul(query_content, key_content.transpose(-2, -1))
        scores = scores + torch.matmul(query_rope, key_rope.transpose(-2, -1))
        scores = scores / (self.config.qk_head_dim**0.5)
        causal = torch.triu(torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.view(1, 1, sequence_length, sequence_length), float("-inf"))
        attended = torch.matmul(F.softmax(scores.float(), dim=-1).to(scores.dtype), values)
        return self.output(attended.transpose(1, 2).contiguous().view(batch, sequence_length, -1))

    def prefill(self, x: Tensor, positions: Tensor | None = None) -> tuple[Tensor, MLACache]:
        sequence_length = x.size(1)
        positions = positions if positions is not None else torch.arange(sequence_length, device=x.device)
        query_content, query_rope = self._query(x, positions)
        compressed, key_content, values, key_rope = self._kv(x, positions)
        cache = MLACache(compressed, key_rope.squeeze(1))
        key_rope_full = key_rope.expand(x.size(0), self.config.n_heads, -1, -1)
        scores = torch.matmul(query_content, key_content.transpose(-2, -1))
        scores = scores + torch.matmul(query_rope, key_rope_full.transpose(-2, -1))
        scores = scores / (self.config.qk_head_dim**0.5)
        causal = torch.triu(torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.view(1, 1, sequence_length, sequence_length), float("-inf"))
        attended = torch.matmul(F.softmax(scores.float(), dim=-1).to(scores.dtype), values)
        return self.output(attended.transpose(1, 2).contiguous().view(x.size(0), sequence_length, -1)), cache

    def decode_naive(self, x: Tensor, cache: MLACache) -> tuple[Tensor, MLACache]:
        if x.size(1) != 1:
            raise ValueError("decode expects exactly one token")
        position = cache.sequence_length
        positions = torch.tensor([position], device=x.device)
        query_content, query_rope = self._query(x, positions)
        compressed, key_content, values, key_rope = self._kv(x, positions)
        compressed_cache = torch.cat((cache.compressed_kv, compressed), dim=1)
        rope_cache = torch.cat((cache.rope_keys, key_rope.squeeze(1)), dim=1)
        batch = x.size(0)
        all_key_content = self.k_content_up(compressed_cache).view(
            batch, -1, self.config.n_heads, self.config.qk_nope_head_dim
        ).transpose(1, 2)
        all_values = self.v_up(compressed_cache).view(
            batch, -1, self.config.n_heads, self.config.v_head_dim
        ).transpose(1, 2)
        all_rope = rope_cache.unsqueeze(1).expand(-1, self.config.n_heads, -1, -1)
        scores = torch.matmul(query_content, all_key_content.transpose(-2, -1))
        scores = scores + torch.matmul(query_rope, all_rope.transpose(-2, -1))
        scores = scores / (self.config.qk_head_dim**0.5)
        attended = torch.matmul(F.softmax(scores.float(), dim=-1).to(scores.dtype), all_values)
        output = self.output(attended.transpose(1, 2).contiguous().view(batch, 1, -1))
        return output, MLACache(compressed_cache, rope_cache)

    def decode(self, x: Tensor, cache: MLACache) -> tuple[Tensor, MLACache]:
        if x.size(1) != 1:
            raise ValueError("decode expects exactly one token")
        position = cache.sequence_length
        positions = torch.tensor([position], device=x.device)
        query_content, query_rope = self._query(x, positions)
        compressed, _, _, key_rope = self._kv(x, positions)
        compressed_cache = torch.cat((cache.compressed_kv, compressed), dim=1)
        rope_cache = torch.cat((cache.rope_keys, key_rope.squeeze(1)), dim=1)
        batch = x.size(0)
        n_heads = self.config.n_heads
        k_up_weight = self.k_content_up.weight.view(n_heads, self.config.qk_nope_head_dim, self.config.kv_lora_rank)
        v_up_weight = self.v_up.weight.view(n_heads, self.config.v_head_dim, self.config.kv_lora_rank)
        absorbed_query = torch.einsum("bhqd,hdc->bhqc", query_content, k_up_weight.to(query_content.dtype))
        content_scores = torch.einsum("bhqc,btc->bhqt", absorbed_query, compressed_cache.to(absorbed_query.dtype))
        all_rope = rope_cache.unsqueeze(1).expand(-1, n_heads, -1, -1)
        rope_scores = torch.matmul(query_rope, all_rope.transpose(-2, -1))
        scores = (content_scores + rope_scores) / (self.config.qk_head_dim**0.5)
        attention = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weighted = torch.einsum("bhqt,btc->bhqc", attention, compressed_cache.to(attention.dtype))
        attended = torch.einsum("bhqc,hdc->bhqd", weighted, v_up_weight.to(weighted.dtype))
        output = self.output(attended.transpose(1, 2).contiguous().view(batch, 1, -1))
        return output, MLACache(compressed_cache, rope_cache)

    def cache_report(self, batch_size: int, sequence_length: int) -> dict[str, int]:
        return {
            "compressed_kv_values": batch_size * sequence_length * self.config.kv_lora_rank,
            "rope_key_values": batch_size * sequence_length * self.config.qk_rope_head_dim,
            "mla_values": batch_size * sequence_length * self.config.cache_values_per_token(),
            "full_kv_values": batch_size * sequence_length * self.config.full_kv_values_per_token(),
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.reference(x)
