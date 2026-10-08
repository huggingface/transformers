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

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


MASK_LOGIT = -1.0e4


@dataclass
class BoundaryEncoding:
    """Encoded boundary states and their validity mask.

    Attributes:
        states: Boundary states `[B, L + 1, D]`.
        mask: True where boundary `i` satisfies `i <= n`.
    """

    states: torch.Tensor
    mask: torch.Tensor


@dataclass
class BoundaryMarginals:
    """Query-conditioned start, end, and inside scores.

    Attributes:
        start_logits: Start scores `[B, Q, L + 1]`.
        end_logits: End scores `[B, Q, L + 1]`.
        inside_logits: Token-inside scores `[B, Q, L]`.
        inside_prefix: Centered cumulative inside scores `[B, Q, L + 1]`.
        inside_prefix_mean: Per-query mean restored by interval scoring.
    """

    start_logits: torch.Tensor
    end_logits: torch.Tensor
    inside_logits: torch.Tensor
    inside_prefix: torch.Tensor
    inside_prefix_mean: torch.Tensor


def build_boundary_mask(text_lengths: torch.Tensor, max_text_length: int) -> torch.Tensor:
    """Return a mask that is true for boundaries `i <= n`.

    Args:
        text_lengths: Token counts `[B]`.
        max_text_length: Padded token length `L`.

    Returns:
        Boolean mask `[B, L + 1]`.
    """
    index = torch.arange(max_text_length + 1, device=text_lengths.device).unsqueeze(0)
    return index <= text_lengths.unsqueeze(1)


def shift_left_with_bos(text_states: torch.Tensor, bos_state: torch.Tensor) -> torch.Tensor:
    """Place each token to the left of the following boundary.

    Args:
        text_states: Token states `[B, L, H]`.
        bos_state: Learned left state for boundary 0.

    Returns:
        Left states `[B, L + 1, H]`.
    """
    batch, length, hidden = text_states.shape
    out = torch.empty(batch, length + 1, hidden, dtype=text_states.dtype, device=text_states.device)
    out[:, 0] = bos_state.to(text_states.dtype)
    out[:, 1:] = text_states
    return out


def shift_right_with_eos(
    text_states: torch.Tensor,
    text_lengths: torch.Tensor,
    eos_state: torch.Tensor,
) -> torch.Tensor:
    """Place each token to the right of the preceding boundary.

    The learned EOS state is written at each sample's final boundary.

    Args:
        text_states: Token states `[B, L, H]`.
        text_lengths: Token counts `[B]`.
        eos_state: Learned right state for boundary `n`.

    Returns:
        Right states `[B, L + 1, H]`.
    """
    batch, length, hidden = text_states.shape
    out = torch.empty(batch, length + 1, hidden, dtype=text_states.dtype, device=text_states.device)
    out[:, :length] = text_states
    out[:, length] = eos_state.to(text_states.dtype)
    eos = eos_state.to(text_states.dtype).view(1, hidden).expand(batch, hidden)
    out[torch.arange(batch, device=text_states.device), text_lengths.clamp(max=length)] = eos
    return out


class ResidualSwiGLU(nn.Module):
    """Pre-norm residual SwiGLU block.

    Args:
        dim: Feature width.
        multiplier: Hidden-size multiplier.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, multiplier: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(1, int(dim * multiplier))
        self.norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Apply the residual SwiGLU update.

        Args:
            states: Boundary states.

        Returns:
            Updated states of the same shape.
        """
        value, gate = self.input_projection(self.norm(states)).chunk(2, dim=-1)
        update = value * F.silu(gate)
        update = self.dropout(update)
        update = self.output_projection(update)
        return states + self.dropout(update)


class BoundaryAttentionBlock(nn.Module):
    """Pre-norm self-attention over valid boundary positions.

    Args:
        dim: Feature width.
        num_heads: Attention heads.
        window: Local window, or 0 for full attention.
        dropout: Attention and residual dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.norm = nn.LayerNorm(dim)
        self.qkv_projection = nn.Linear(dim, 3 * dim)
        self.output_projection = nn.Linear(dim, dim)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attend over boundaries that `mask` marks as valid.

        Args:
            states: Boundary states `[B, N, D]`.
            mask: Valid-boundary mask `[B, N]`.

        Returns:
            Residual states with padding positions cleared.
        """
        batch, length, dim = states.shape
        qkv = self.qkv_projection(self.norm(states)).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        allowed = mask.view(batch, 1, 1, length).expand(batch, 1, length, length)
        if self.window > 0:
            positions = torch.arange(length, device=states.device)
            local = (positions.view(length, 1) - positions.view(1, length)).abs() <= self.window
            allowed = allowed & local.view(1, 1, length, length)
        # Padding rows keep a diagonal key so attention stays finite.
        diagonal = torch.eye(length, dtype=torch.bool, device=states.device)
        allowed = allowed | diagonal.view(1, 1, length, length)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        update = self.dropout(self.output_projection(attended))
        return (states + update) * mask.unsqueeze(-1).to(states.dtype)


class BoundaryEncoder(nn.Module):
    """Project left and right token states into one vector per boundary.

    Boundary `i` sits between token `i - 1` and token `i`. Boundary 0 uses a
    learned BOS left state and boundary `n` uses a learned EOS right state.

    Args:
        hidden_size: Token hidden size.
        boundary_dim: Boundary state width.
        dropout: Dropout probability.
        refinement_layers: Number of residual SwiGLU blocks.
        ffn_multiplier: SwiGLU hidden multiplier.
        attention_layers: Number of boundary self-attention blocks.
        attention_heads: Heads in each attention block.
        attention_window: Local attention window, or 0 for full attention.
    """

    def __init__(
        self,
        hidden_size: int,
        boundary_dim: int,
        dropout: float = 0.1,
        refinement_layers: int = 1,
        ffn_multiplier: float = 2.0,
        attention_layers: int = 0,
        attention_heads: int = 4,
        attention_window: int = 0,
    ):
        super().__init__()
        if refinement_layers < 0:
            raise ValueError(f"refinement_layers must be >= 0, got {refinement_layers}")
        if ffn_multiplier <= 0:
            raise ValueError(f"ffn_multiplier must be > 0, got {ffn_multiplier}")
        self.hidden_size = hidden_size
        self.boundary_dim = boundary_dim
        self.left_projection = nn.Linear(hidden_size, boundary_dim)
        self.right_projection = nn.Linear(hidden_size, boundary_dim)
        self.output_projection = nn.Linear(2 * boundary_dim, boundary_dim)
        self.layer_norm = nn.LayerNorm(boundary_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList(
            BoundaryAttentionBlock(boundary_dim, attention_heads, attention_window, dropout)
            for _ in range(attention_layers)
        )
        self.refinement_blocks = nn.ModuleList(
            ResidualSwiGLU(boundary_dim, ffn_multiplier, dropout) for _ in range(refinement_layers)
        )
        self.bos_state = nn.Parameter(torch.zeros(hidden_size))
        self.eos_state = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.bos_state, std=0.02)
        nn.init.normal_(self.eos_state, std=0.02)

    def forward(self, text_states: torch.Tensor, text_mask: torch.Tensor) -> BoundaryEncoding:
        """Encode token states as masked boundary states.

        Args:
            text_states: Token states `[B, L, H]`.
            text_mask: Valid-token mask `[B, L]`.

        Returns:
            Boundary states and mask.
        """
        _batch, length, _hidden = text_states.shape
        text_lengths = text_mask.sum(dim=1).long()
        left = shift_left_with_bos(text_states, self.bos_state)
        right = shift_right_with_eos(text_states, text_lengths, self.eos_state)
        left_p = self.left_projection(left)
        right_p = self.right_projection(right)
        states = self.output_projection(torch.cat([left_p, right_p], dim=-1))
        states = self.layer_norm(states)
        states = self.dropout(states)
        mask = build_boundary_mask(text_lengths, length)
        for block in self.attention_blocks:
            states = block(states, mask)
        for block in self.refinement_blocks:
            states = block(states)
        states = states * mask.unsqueeze(-1).to(states.dtype)
        return BoundaryEncoding(states=states, mask=mask)


def _masked_fill_min(logits: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Replace rejected logits with a finite sentinel."""
    return logits.masked_fill(~keep_mask, MASK_LOGIT)


class BoundaryQueryHead(nn.Module):
    """Score start, end, and inside positions for every query.

    Args:
        hidden_size: Token hidden size.
        boundary_dim: Boundary and score width.
        query_dim: Query state width. Defaults to `hidden_size`.
        dropout: Dropout on projected boundary and token states.
    """

    def __init__(
        self,
        hidden_size: int,
        boundary_dim: int,
        query_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.boundary_dim = boundary_dim
        q_dim = query_dim if query_dim is not None else hidden_size
        self.start_boundary_projection = nn.Linear(boundary_dim, boundary_dim)
        self.start_query_projection = nn.Linear(q_dim, boundary_dim)
        self.end_boundary_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_query_projection = nn.Linear(q_dim, boundary_dim)
        self.inside_text_projection = nn.Linear(hidden_size, boundary_dim)
        self.inside_query_projection = nn.Linear(q_dim, boundary_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        boundary_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> BoundaryMarginals:
        """Dot-product start, end, and inside logits.

        Args:
            boundary_states: Encoded boundaries `[B, L + 1, D]`.
            boundary_mask: Valid-boundary mask `[B, L + 1]`.
            text_states: Token states `[B, L, H]`.
            text_mask: Valid-token mask `[B, L]`.
            query_states: Query states `[B, Q, H]`.
            query_mask: Valid-query mask `[B, Q]`.

        Returns:
            Masked marginals and a centered inside prefix.
        """
        scale = 1.0 / math.sqrt(self.boundary_dim)
        start_b = self.dropout(self.start_boundary_projection(boundary_states))
        start_q = self.start_query_projection(query_states)
        start_logits = torch.einsum("bld,bqd->bql", start_b, start_q) * scale

        end_b = self.dropout(self.end_boundary_projection(boundary_states))
        end_q = self.end_query_projection(query_states)
        end_logits = torch.einsum("bld,bqd->bql", end_b, end_q) * scale

        inside_t = self.dropout(self.inside_text_projection(text_states))
        inside_q = self.inside_query_projection(query_states)
        inside_logits = torch.einsum("bld,bqd->bql", inside_t, inside_q) * scale

        boundary_keep = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
        token_keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
        start_logits = _masked_fill_min(start_logits, boundary_keep)
        end_logits = _masked_fill_min(end_logits, boundary_keep)
        inside_logits = _masked_fill_min(inside_logits, token_keep)

        inside_for_prefix = inside_logits.masked_fill(~token_keep, 0.0).float()
        valid_count = token_keep.sum(-1, keepdim=True).clamp_min(1)
        inside_mean = (inside_for_prefix.sum(-1, keepdim=True) / valid_count).detach()
        centered = (inside_for_prefix - inside_mean) * token_keep.to(inside_for_prefix.dtype)
        zeros = torch.zeros(
            centered.shape[0],
            centered.shape[1],
            1,
            dtype=torch.float32,
            device=inside_for_prefix.device,
        )
        inside_prefix = torch.cat([zeros, centered.cumsum(dim=-1)], dim=-1)
        return BoundaryMarginals(
            start_logits=start_logits,
            end_logits=end_logits,
            inside_logits=inside_logits,
            inside_prefix=inside_prefix,
            inside_prefix_mean=inside_mean,
        )
