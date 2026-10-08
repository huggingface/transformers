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


def gather_states(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather `[B, N, D]` states with `[B, Q, C]` indices.

    Args:
        states: Source states.
        indices: Per-query candidate positions.

    Returns:
        Gathered states `[B, Q, C, D]`.
    """
    batch, length, dim = states.shape
    queries, count = indices.shape[1:3]
    flat = indices.clamp(0, length - 1).reshape(batch, queries * count, 1)
    flat = flat.expand(batch, queries * count, dim)
    return states.gather(1, flat).view(batch, queries, count, dim)


def gather_rows(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather `[B, N, D]` states with `[B, K]` indices.

    Args:
        states: Source states.
        indices: Row positions.

    Returns:
        Gathered states `[B, K, D]`.
    """
    length = states.shape[1]
    dim = states.shape[-1]
    index = indices.clamp(0, length - 1).unsqueeze(-1).expand(-1, -1, dim)
    return states.gather(1, index)


def gather_prefix(prefix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather prefix states for document-level or per-query indices.

    Args:
        prefix: Inclusive prefix tensor `[B, N, D]`.
        indices: `[B, K]` or `[B, Q, C]` positions.

    Returns:
        Prefix rows aligned with `indices`.
    """
    if indices.dim() == 2:
        return gather_rows(prefix, indices)
    return gather_states(prefix, indices)


class SpanContentPooler(nn.Module):
    """Pool token states over half-open spans with prefix sums.

    Args:
        hidden_size: Token state width.
        content_dim: Projected content width.
        dropout: Dropout probability.
        use_soft_max_pool: Also pool a smooth maximum of token values.
    """

    def __init__(
        self,
        hidden_size: int,
        content_dim: int,
        dropout: float = 0.1,
        use_soft_max_pool: bool = False,
    ) -> None:
        super().__init__()
        self.content_dim = content_dim
        self.use_soft_max_pool = use_soft_max_pool
        self.output_dim = content_dim * (2 if use_soft_max_pool else 1)
        self.value_projection = nn.Linear(hidden_size, content_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def build_prefix(
        self,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build mean and optional log-sum-exp prefixes.

        Args:
            text_states: Token states `[B, L, H]`.
            text_mask: Valid-token mask `[B, L]`.

        Returns:
            Mean prefix `[B, L + 1, C]` and an optional log-sum-exp prefix.
        """
        values = self.value_projection(text_states)
        values = values * text_mask.unsqueeze(-1).to(values.dtype)
        values32 = values.float()
        zeros = values32.new_zeros(values32.shape[0], 1, self.content_dim)
        mean_prefix = torch.cat((zeros, values32.cumsum(1)), dim=1)

        lse_prefix = None
        if self.use_soft_max_pool:
            floor = torch.finfo(torch.float32).min / 4.0
            masked = values32.masked_fill(~text_mask.unsqueeze(-1), floor)
            lse_prefix = torch.cat(
                (
                    zeros.new_full((zeros.shape[0], 1, self.content_dim), floor),
                    torch.logcumsumexp(masked, dim=1),
                ),
                dim=1,
            )
        return mean_prefix, lse_prefix

    def pool(
        self,
        mean_prefix: torch.Tensor,
        lse_prefix: torch.Tensor | None,
        starts: torch.Tensor,
        ends: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Pool `[start, end)` spans from precomputed prefixes.

        Args:
            mean_prefix: Cumulative value prefix.
            lse_prefix: Optional log-sum-exp prefix.
            starts: Inclusive start positions.
            ends: Exclusive end positions.
            out_dtype: Dtype of the normalized pool.

        Returns:
            Normalized span content.
        """
        length = (ends - starts).clamp_min(1).unsqueeze(-1).float()
        span_sum = gather_prefix(mean_prefix, ends) - gather_prefix(mean_prefix, starts)
        pooled = span_sum / length
        if lse_prefix is not None:
            lse_end = gather_prefix(lse_prefix, ends)
            lse_start = gather_prefix(lse_prefix, starts)
            delta = (lse_start - lse_end).clamp(max=-1e-6)
            soft_max = lse_end + torch.log1p(-torch.exp(delta))
            soft_max = torch.nan_to_num(soft_max, neginf=0.0, posinf=0.0)
            pooled = torch.cat((pooled, soft_max), dim=-1)
        pooled = self.layer_norm(pooled.to(out_dtype))
        return self.dropout(pooled)
