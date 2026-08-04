import pytest
import torch

from routing import LoadBalancer, TopKRouter


def test_top_k_selection_and_weight_normalization() -> None:
    torch.manual_seed(1)
    router = TopKRouter(d_model=8, n_experts=4, top_k=2, route_scale=2.5)
    result = router(torch.randn(6, 8))
    assert result.selected_indices.shape == (6, 2)
    assert result.selected_weights.shape == (6, 2)
    assert torch.allclose(result.selected_weights.sum(dim=-1), torch.full((6,), 2.5), atol=1e-6)
    assert int(result.expert_load.sum()) == 12


def test_bias_changes_selection_not_mixing_affinity() -> None:
    torch.manual_seed(2)
    router = TopKRouter(d_model=4, n_experts=3, top_k=1)
    x = torch.randn(5, 4)
    first = router(x)
    router.expert_bias[0] += 100.0
    second = router(x)
    assert not torch.equal(first.selected_indices, second.selected_indices)
    assert torch.allclose(first.affinities, second.affinities)


def test_load_balancer_pushes_underused_experts_up() -> None:
    router = TopKRouter(d_model=4, n_experts=4, top_k=1)
    balancer = LoadBalancer(router, update_rate=0.1)
    balancer.update(torch.tensor([8, 2, 2, 2]))
    assert router.expert_bias[0] < 0
    assert torch.all(router.expert_bias[1:] > 0)


def test_load_entropy_uniform_and_concentrated() -> None:
    router = TopKRouter(d_model=4, n_experts=4, top_k=1)
    x = torch.randn(4, 4)
    result = router(x)
    result.expert_load = torch.tensor([1, 1, 1, 1])
    assert result.load_entropy() == pytest.approx(1.0, abs=1e-6)
    result.expert_load = torch.tensor([4, 0, 0, 0])
    assert result.load_entropy() == pytest.approx(0.0, abs=1e-6)
    result.expert_load = torch.tensor([2, 2, 0, 0])
    assert 0.0 < result.load_entropy() < 1.0
    assert "load_entropy" in result.summary()


def test_top_k_validation() -> None:
    try:
        TopKRouter(d_model=4, n_experts=2, top_k=3)
    except ValueError:
        return
    raise AssertionError("invalid top_k was accepted")
