from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RoutingResult:
    selected_indices: Tensor
    selected_weights: Tensor
    affinities: Tensor
    selection_scores: Tensor
    expert_load: Tensor

    @property
    def assignments(self) -> int:
        return self.selected_indices.numel()

    def sequence_balance_loss(self) -> Tensor:
        tokens = self.selected_indices.size(0)
        experts = self.affinities.size(-1)
        frequency = torch.bincount(self.selected_indices.flatten(), minlength=experts).float()
        frequency = frequency / max(tokens * self.selected_indices.size(1), 1) * experts
        mean_affinity = self.affinities.float().mean(dim=0)
        return (frequency * mean_affinity).sum()

    def load_entropy(self) -> float:
        n_experts = self.affinities.size(-1)
        total = self.expert_load.sum().clamp_min(1)
        probabilities = self.expert_load.float() / total
        nonzero = probabilities[probabilities > 0]
        entropy = -(nonzero * nonzero.log()).sum()
        if n_experts <= 1:
            return 1.0
        return float(entropy / torch.log(torch.tensor(float(n_experts))))

    def summary(self) -> dict:
        total = self.expert_load.sum().clamp_min(1)
        return {
            "selected_indices": self.selected_indices.detach().cpu().tolist(),
            "selected_weights": self.selected_weights.detach().float().cpu().tolist(),
            "expert_load": self.expert_load.detach().cpu().tolist(),
            "expert_fraction": (self.expert_load.float() / total).detach().cpu().tolist(),
            "assignments": self.assignments,
            "load_entropy": self.load_entropy(),
        }


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, route_scale: float = 1.0) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.n_experts = n_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self.projection = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))

    def forward(self, x: Tensor) -> RoutingResult:
        logits = self.projection(x)
        affinities = torch.sigmoid(logits.float())
        selection_scores = affinities + self.expert_bias.to(device=x.device, dtype=affinities.dtype)
        _, selected_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        selected_affinities = affinities.gather(-1, selected_indices)
        selected_weights = selected_affinities / selected_affinities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        selected_weights = selected_weights * self.route_scale
        expert_load = torch.bincount(selected_indices.flatten(), minlength=self.n_experts)
        return RoutingResult(selected_indices, selected_weights.to(dtype=x.dtype), affinities, selection_scores, expert_load)


class LoadBalancer:
    def __init__(self, router: TopKRouter, update_rate: float = 1e-3) -> None:
        self.router = router
        self.update_rate = update_rate

    @torch.no_grad()
    def update(self, expert_load: Tensor) -> None:
        target = expert_load.float().mean()
        delta = torch.where(expert_load.float() < target, self.update_rate, -self.update_rate)
        self.router.expert_bias.add_(delta.to(self.router.expert_bias.device))

    def bias(self) -> Tensor:
        return self.router.expert_bias.detach().clone()
