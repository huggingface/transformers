import numbers

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype

        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)

        return self.weight * input.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"
