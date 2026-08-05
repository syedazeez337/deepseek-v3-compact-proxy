from __future__ import annotations

import torch
from torch import Tensor, nn

from compact_v3.experts import SwiGLUExpert
from compact_v3.routing import LoadBalancer, RoutingResult, TopKRouter
from compact_v3.config import CompactV3Config


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
        self._dispatch_routed(flat, routing, output)
        if update_balancer and self.training:
            self.load_balancer.update(routing.expert_load)
        self.last_routing = routing
        balance_loss = routing.sequence_balance_loss() * self.config.sequence_balance_coefficient
        return output.reshape(original_shape), routing, balance_loss

    def _dispatch_routed(self, flat: Tensor, routing: RoutingResult, output: Tensor) -> None:
        """Group the (token, slot) assignments by expert in one sorted pass.

        The router already produced `expert_load` as a bincount over the same
        assignments, so the sorted groups line up with it exactly and a single
        `.tolist()` gives every group boundary. That one transfer is the only
        device-to-host sync here; the previous implementation ran `.any()` and
        `.nonzero()` per expert, which forced one sync per expert per layer.
        Matches the structure of DeepSeek-V3's own `inference/model.py` MoE
        loop, which likewise takes counts to the host once and skips empty
        experts from that list.
        """
        assignments = routing.selected_indices.reshape(-1)
        weights = routing.selected_weights.reshape(-1)
        token_ids = torch.arange(flat.size(0), device=flat.device)
        token_ids = token_ids.unsqueeze(1).expand_as(routing.selected_indices).reshape(-1)

        order = torch.argsort(assignments, stable=True)
        token_ids = token_ids[order]
        weights = weights[order]

        start = 0
        for expert_id, count in enumerate(routing.expert_load.tolist()):
            if count == 0:
                continue
            token_indices = token_ids[start:start + count]
            expert_output = self.routed_experts[expert_id](flat[token_indices])
            output.index_add_(0, token_indices, expert_output * weights[start:start + count].unsqueeze(-1))
            start += count

    def forward_reference(self, x: Tensor, update_balancer: bool = False) -> tuple[Tensor, RoutingResult, Tensor]:
        """Pre-Gate-T dispatch, kept as the equivalence reference for `forward`.

        Masks and re-derives indices per expert. Numerically identical to
        `forward` but syncs once per expert, so it is only used by tests.
        """
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
