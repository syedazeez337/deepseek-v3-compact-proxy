import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("rotary dimension must be even")
        inverse_frequency = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("sin", angles.sin().repeat_interleave(2, dim=-1), persistent=False)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        if positions.numel() == 0:
            return x
        if positions.min() < 0 or positions.max() >= self.cos.size(0):
            raise ValueError("RoPE positions exceed the configured context")
        cos = self.cos[positions].to(device=x.device, dtype=x.dtype).view(1, 1, -1, x.size(-1))
        sin = self.sin[positions].to(device=x.device, dtype=x.dtype).view(1, 1, -1, x.size(-1))
        return x * cos + rotate_half(x) * sin

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        return self.rotate(x, positions)
