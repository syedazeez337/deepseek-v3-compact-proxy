from __future__ import annotations

import torch
from torch import Tensor, nn

from experts import SwiGLUExpert
from routing import LoadBalancer, RoutingResult, TopKRouter
from v3_config import CompactV3Config


class DeepSeekMoE(nn.Module):
    def __init__(self, config: CompactV3Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.router = TopKRouter(config.d_model, config.n_routed_experts, config.top_k, config.route_scale)
        self.load_balancer = LoadBalancer(self.router, config.router_bias_update_rate)
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(config.d_model, config.expert_hidden_dim) for _ in range(config.n_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [SwiGLUExpert(config.d_model, config.expert_hidden_dim) for _ in range(config.n_routed_experts)]
        )
        self.last_routing: RoutingResult | None = None

    def forward(self, x: Tensor, update_balancer: bool = False) -> tuple[Tensor, RoutingResult, Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        routing = self.router(flat)
        output = torch.zeros_like(flat)
        for expert in self.shared_experts:
            output = output + expert(flat)
        for expert_id, expert in enumerate(self.routed_experts):
            token_slots = routing.selected_indices == expert_id
            token_mask = token_slots.any(dim=-1)
            if not token_mask.any():
                continue
            token_indices = token_mask.nonzero(as_tuple=False).squeeze(-1)
            slot_indices = token_slots[token_mask].nonzero(as_tuple=False)[:, 1]
            expert_output = expert(flat[token_indices])
            weights = routing.selected_weights[token_indices, slot_indices].unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output * weights)
        if update_balancer and self.training:
            self.load_balancer.update(routing.expert_load)
        self.last_routing = routing
        balance_loss = routing.sequence_balance_loss() * self.config.sequence_balance_coefficient
        return output.reshape(original_shape), routing, balance_loss

    def parameter_report(self) -> dict[str, int | float]:
        shared = sum(parameter.numel() for parameter in self.shared_experts.parameters())
        routed = sum(parameter.numel() for parameter in self.routed_experts.parameters())
        router = sum(parameter.numel() for parameter in self.router.parameters())
        active_routed = routed * self.config.top_k / self.config.n_routed_experts
        return {
            "shared_parameters": shared,
            "routed_parameters": routed,
            "router_parameters": router,
            "total_parameters": shared + routed + router,
            "active_routed_parameter_estimate": active_routed,
            "active_parameter_estimate": shared + router + active_routed,
        }

    def routing_summary(self) -> dict | None:
        return self.last_routing.summary() if self.last_routing is not None else None
