# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch import nn


class RotaryBoundaryEmbedding(nn.Module):
    """Rotate endpoint channels so their dot product depends on distance.

    Args:
        dim: Even feature width to rotate.
        base: Geometric period of the rotary frequencies.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary mixing at integer boundary positions.

        Args:
            states: Endpoint states `[..., dim]`.
            positions: Integer positions broadcastable to the leading axes.

        Returns:
            Rotated states in the input dtype.
        """
        angle = positions.unsqueeze(-1).float() * self.inv_freq
        cos, sin = torch.cos(angle), torch.sin(angle)
        even = states[..., 0::2].float()
        odd = states[..., 1::2].float()
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).to(states.dtype)
