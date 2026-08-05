from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from compact_v3.norms import RMSNorm
from compact_v3.config import CompactV3Config


@dataclass
class MTPResult:
    loss: Tensor
    logits: Tensor
    target_offset: int


class MTPObjective(nn.Module):
    def __init__(self, config: CompactV3Config, token_embedding: nn.Embedding, output_head: nn.Linear) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = token_embedding
        self.output_head = output_head
        self.hidden_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.token_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.merge = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor, future_targets: Tensor | None = None) -> MTPResult | None:
        if self.config.mtp_depth == 0 or future_targets is None:
            return None
        if hidden_states.ndim != 3 or future_targets.ndim != 2:
            raise ValueError("hidden_states must be (batch, sequence, d_model) and targets must be (batch, sequence)")
        if hidden_states.shape[:2] != future_targets.shape:
            raise ValueError("future target shape must match hidden state batch and sequence")
        future_embeddings = self.token_embedding(future_targets)
        merged = self.merge(torch.cat((self.hidden_norm(hidden_states), self.token_norm(future_embeddings)), dim=-1))
        sequence_length = merged.size(1)
        causal = torch.triu(torch.ones(sequence_length, sequence_length, device=merged.device, dtype=torch.bool), diagonal=1)
        refined = self.transformer(merged, mask=causal)
        logits = self.output_head(self.final_norm(refined))
        return MTPResult(
            loss=nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), future_targets.reshape(-1)),
            logits=logits,
            target_offset=2,
        )

    def combined_loss(self, main_loss: Tensor, mtp_result: MTPResult | None, weight: float | None = None) -> Tensor:
        if mtp_result is None:
            return main_loss
        weight = self.config.mtp_weight if weight is None else weight
        return main_loss + weight * mtp_result.loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def make_future_targets(token_ids: Tensor, horizon: int = 2) -> Tensor:
    if token_ids.ndim != 2 or token_ids.size(1) <= horizon:
        raise ValueError("token_ids must have more positions than the requested horizon")
    return token_ids[:, horizon:]


def align_hidden_states(hidden_states: Tensor, horizon: int = 2) -> Tensor:
    if hidden_states.ndim != 3 or hidden_states.size(1) <= horizon:
        raise ValueError("hidden_states must have more positions than the requested horizon")
    return hidden_states[:, :-horizon]


def mtp_weight_schedule(
    step: int, total_steps: int, initial_weight: float, final_weight: float, decay_step_fraction: float
) -> float:
    if step / max(total_steps, 1) < decay_step_fraction:
        return initial_weight
    return final_weight


__all__ = ["MTPObjective", "MTPResult", "align_hidden_states", "make_future_targets", "mtp_weight_schedule"]
