import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.value = nn.Linear(d_model, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(F.silu(self.gate(x)) * self.value(x))
