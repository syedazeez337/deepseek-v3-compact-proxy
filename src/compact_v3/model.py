from __future__ import annotations

from dataclasses import asdict

import torch
from torch import Tensor, nn

from compact_v3.mtp import MTPObjective, align_hidden_states, make_future_targets
from compact_v3.norms import RMSNorm
from compact_v3.block import BlockCache, CompactV3Block
from compact_v3.config import CompactV3Config


class CompactV3Model(nn.Module):
    def __init__(self, config: CompactV3Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        dense_prefix_hidden_dim = (config.n_shared_experts + config.top_k) * config.expert_hidden_dim
        self.blocks = nn.ModuleList(
            [
                CompactV3Block(
                    config,
                    use_moe=config.use_moe and layer_index >= config.n_dense_layers,
                    dense_hidden_dim=dense_prefix_hidden_dim if layer_index < config.n_dense_layers else None,
                )
                for layer_index in range(config.n_layer)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.mtp = MTPObjective(config, self.token_embedding, self.output)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        update_balancer: bool = False,
    ) -> tuple[Tensor, Tensor | None, dict]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.size(1) > self.config.context_length:
            raise ValueError("sequence exceeds configured context_length")
        x = self.token_embedding(token_ids)
        routing = []
        balance_losses = []
        for block in self.blocks:
            x, block_routing, balance_loss = block(x, update_balancer=update_balancer)
            routing.append(block_routing)
            balance_losses.append(balance_loss)
        hidden_states = self.final_norm(x)
        logits = self.output(hidden_states)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        diagnostics = {
            "routing": routing,
            "balance_loss": torch.stack(balance_losses).mean(),
            "hidden_states": hidden_states,
        }
        if targets is not None and self.config.mtp_depth:
            mtp_hidden = align_hidden_states(hidden_states, horizon=2)
            mtp_targets = make_future_targets(token_ids, horizon=2)
            diagnostics["mtp"] = self.mtp(mtp_hidden, mtp_targets)
        else:
            diagnostics["mtp"] = None
        return logits, loss, diagnostics

    def prefill(self, token_ids: Tensor) -> tuple[Tensor, list[BlockCache]]:
        if token_ids.ndim != 2 or token_ids.size(1) > self.config.context_length:
            raise ValueError("prefill tokens exceed the configured shape/context")
        x = self.token_embedding(token_ids)
        caches = []
        for block in self.blocks:
            x, cache, _, _ = block.prefill(x)
            caches.append(cache)
        return self.output(self.final_norm(x)), caches

    def decode(self, token_ids: Tensor, caches: list[BlockCache]) -> tuple[Tensor, list[BlockCache]]:
        if token_ids.ndim != 2 or token_ids.size(1) != 1:
            raise ValueError("decode expects token_ids with shape (batch, 1)")
        if len(caches) != self.config.n_layer:
            raise ValueError("cache count does not match model depth")
        x = self.token_embedding(token_ids)
        next_caches = []
        for block, cache in zip(self.blocks, caches):
            x, next_cache, _, _ = block.decode(x, cache)
            next_caches.append(next_cache)
        return self.output(self.final_norm(x)), next_caches

    def parameter_report(self) -> dict[str, int | float]:
        unique_parameters = sum(parameter.numel() for parameter in self.parameters())
        block_reports = [block.parameter_report() for block in self.blocks]
        active_blocks = sum(float(report["active_moe_parameters"]) for report in block_reports)
        return {
            "unique_parameters": unique_parameters,
            "embedding_parameters": self.token_embedding.weight.numel(),
            "block_parameters": sum(int(report["total_parameters"]) for report in block_reports),
            "active_moe_parameters_per_layer_sum": active_blocks,
            "config": asdict(self.config),
        }

    def tied_output_embedding(self) -> bool:
        return self.output.weight.data_ptr() == self.token_embedding.weight.data_ptr()
