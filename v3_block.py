from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from experts import SwiGLUExpert
from mla import MLACache, MultiHeadLatentAttention
from moe_v3 import DeepSeekMoE
from norms import RMSNorm
from routing import RoutingResult
from v3_config import CompactV3Config


@dataclass
class BlockCache:
    mla: MLACache


class CompactV3Block(nn.Module):
    def __init__(self, config: CompactV3Config, use_moe: bool | None = None, dense_hidden_dim: int | None = None) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        use_moe = config.use_moe if use_moe is None else use_moe
        self.moe = DeepSeekMoE(config) if use_moe else None
        dense_hidden_dim = config.expert_hidden_dim if dense_hidden_dim is None else dense_hidden_dim
        self.dense_ffn = None if use_moe else SwiGLUExpert(config.d_model, dense_hidden_dim)

    def forward(self, x: Tensor, update_balancer: bool = False) -> tuple[Tensor, RoutingResult, Tensor]:
        attention_output = self.attention(self.attn_norm(x))
        x = x + attention_output
        if self.moe is not None:
            moe_output, routing, balance_loss = self.moe(self.ffn_norm(x), update_balancer=update_balancer)
        else:
            moe_output = self.dense_ffn(self.ffn_norm(x))
            routing = None
            balance_loss = x.new_zeros(())
        return x + moe_output, routing, balance_loss

    def prefill(self, x: Tensor, update_balancer: bool = False) -> tuple[Tensor, BlockCache, RoutingResult, Tensor]:
        attention_output, cache = self.attention.prefill(self.attn_norm(x))
        x = x + attention_output
        if self.moe is not None:
            moe_output, routing, balance_loss = self.moe(self.ffn_norm(x), update_balancer=update_balancer)
        else:
            moe_output = self.dense_ffn(self.ffn_norm(x))
            routing = None
            balance_loss = x.new_zeros(())
        return x + moe_output, BlockCache(cache), routing, balance_loss

    def decode(self, x: Tensor, cache: BlockCache, update_balancer: bool = False) -> tuple[Tensor, BlockCache, RoutingResult, Tensor]:
        attention_output, next_cache = self.attention.decode(self.attn_norm(x), cache.mla)
        x = x + attention_output
        if self.moe is not None:
            moe_output, routing, balance_loss = self.moe(self.ffn_norm(x), update_balancer=update_balancer)
        else:
            moe_output = self.dense_ffn(self.ffn_norm(x))
            routing = None
            balance_loss = x.new_zeros(())
        return x + moe_output, BlockCache(next_cache), routing, balance_loss

    def parameter_report(self) -> dict[str, int | float]:
        attention = sum(parameter.numel() for parameter in self.attention.parameters())
        norms = sum(parameter.numel() for parameter in self.attn_norm.parameters()) + sum(
            parameter.numel() for parameter in self.ffn_norm.parameters()
        )
        if self.moe is not None:
            ffn_report = self.moe.parameter_report()
            ffn_total = int(ffn_report["total_parameters"])
            active_ffn = float(ffn_report["active_parameter_estimate"])
        else:
            ffn_total = sum(parameter.numel() for parameter in self.dense_ffn.parameters())
            active_ffn = float(ffn_total)
        return {
            "attention_parameters": attention,
            "moe_parameters": ffn_total,
            "normalization_parameters": norms,
            "total_parameters": attention + ffn_total + norms,
            "active_moe_parameters": active_ffn,
        }
