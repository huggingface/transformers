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

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from . import loss_gliner2
from .configuration_gliner2 import Gliner2Config


_CLASSIFICATION_TASK = 4


def _mlp(input_dim, intermediate_dims, output_dim, dropout=0.0, activation="relu"):
    """Build the sequential MLP whose indices match published checkpoints."""
    layers = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers.append(nn.Linear(in_dim, dim))
        layers.append(ACT2FN[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)  # trf-ignore: TRF036


def _projection(hidden_size, dropout, out_dim=None):
    """Expand by 4, apply relu, then project back."""
    if out_dim is None:
        out_dim = hidden_size
    return nn.Sequential(  # trf-ignore: TRF036
        nn.Linear(hidden_size, out_dim * 4),
        ACT2FN["relu"],
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim),
    )


def _extract_elements(sequence, indices):
    """Gather `[B, K, D]` rows from `[B, L, D]`."""
    hidden = sequence.size(-1)
    expanded = indices.unsqueeze(2).expand(-1, -1, hidden)
    return torch.gather(sequence, 1, expanded)


class CompileSafeGRU(nn.Module):
    """Single-layer GRU with `nn.GRU` parameter names."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size**0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, inputs, hidden):
        """Run the GRU. `inputs` is `(seq, batch, input)` and `hidden` is `(batch, hidden)`."""
        seq_len = inputs.shape[0]
        if seq_len == 0:
            return inputs.new_empty(0, hidden.shape[0], self.hidden_size)
        gi_all = F.linear(inputs, self.weight_ih_l0, self.bias_ih_l0)
        outputs = []
        for step in range(seq_len):
            gi = gi_all[step]
            gh = F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0)
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            reset = torch.sigmoid(i_r + h_r)
            update = torch.sigmoid(i_z + h_z)
            new = torch.tanh(i_n + reset * h_n)
            hidden = (1 - update) * new + update * hidden
            outputs.append(hidden)
        return torch.stack(outputs, dim=0)


class DownscaledTransformer(nn.Module):
    """Project into a small encoder, then back to the input width."""

    def __init__(self, input_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_projector = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_projector = _mlp(
            hidden_size + input_size,
            [input_size, input_size],
            input_size,
            dropout=0.0,
        )

    def forward(self, inputs):
        projected = self.in_projector(inputs)
        transformed = self.transformer(projected)
        return self.out_projector(torch.cat([transformed, inputs], dim=-1))


class CountLSTM(nn.Module):
    """Count-step embeddings from a learned position and a GRU."""

    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)
        self.projector = _mlp(hidden_size * 2, [hidden_size * 4], hidden_size, dropout=0.0)

    def forward(self, field_emb, count):
        """Return `(count, fields, hidden)` embeddings."""
        fields, hidden = field_emb.shape
        count = min(count, self.max_count)
        indices = torch.arange(count, device=field_emb.device)
        positions = self.pos_embedding(indices).unsqueeze(1).expand(count, fields, hidden)
        output = self.gru(positions, field_emb)
        broadcast = field_emb.unsqueeze(0).expand_as(output)
        return self.projector(torch.cat([output, broadcast], dim=-1))


class CountLSTMv2(nn.Module):
    """Count-step embeddings with a GRU followed by a downscaled transformer."""

    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)
        self.transformer = DownscaledTransformer(hidden_size, hidden_size=128, num_heads=4, num_layers=2, dropout=0.1)

    def forward(self, field_emb, count):
        """Return `(count, fields, hidden)` embeddings."""
        fields, _ = field_emb.size()
        count = min(count, self.max_count)
        indices = torch.arange(self.max_count, device=field_emb.device)[:count]
        positions = self.pos_embedding(indices).unsqueeze(1).expand(-1, fields, -1)
        output = self.gru(positions, field_emb)
        broadcast = field_emb.unsqueeze(0).expand_as(output)
        return self.transformer(output + broadcast)


class SpanMarkerV0(nn.Module):
    """Span states from projected start and end markers."""

    def __init__(self, hidden_size, max_width, dropout=0.1):
        super().__init__()
        self.max_width = max_width
        self.project_start = _projection(hidden_size, dropout)
        self.project_end = _projection(hidden_size, dropout)
        self.out_project = _projection(hidden_size * 2, dropout, hidden_size)

    def forward(self, hidden, span_idx):
        """Return `[B, L, max_width, D]` span states."""
        batch, length, _ = hidden.size()
        start = _extract_elements(self.project_start(hidden), span_idx[:, :, 0])
        end = _extract_elements(self.project_end(hidden), span_idx[:, :, 1])
        return self.out_project(torch.cat([start, end], dim=-1).relu()).view(batch, length, self.max_width, -1)


class SpanRepLayer(nn.Module):  # trf-ignore: TRF026
    """`markerV0` span representation. The submodule name is `span_rep_layer`."""

    def __init__(self, hidden_size, max_width, span_mode="markerV0", dropout=0.1):
        super().__init__()
        if span_mode != "markerV0":
            raise ValueError(f"Unknown span mode {span_mode}")
        self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, dropout=dropout)

    def forward(self, hidden, span_idx):
        return self.span_rep_layer(hidden, span_idx)


def _gather(token_embeddings, indices, mask):
    """Gather rows, clamping pads, then zero them with the mask."""
    hidden = token_embeddings.shape[-1]
    safe = indices.clamp(0, token_embeddings.shape[1] - 1)
    states = token_embeddings.gather(1, safe.unsqueeze(-1).expand(-1, -1, hidden))
    return states * mask.unsqueeze(-1).to(states.dtype)


def _span_indices(length, max_width, device):
    """Build safe `(1, length * max_width, 2)` start/end indices."""
    starts = torch.arange(length, device=device).unsqueeze(1).expand(-1, max_width)
    offsets = torch.arange(max_width, device=device).unsqueeze(0)
    ends = starts + offsets
    valid = ends < length
    starts_flat = starts.reshape(-1)
    ends_flat = ends.reshape(-1)
    invalid = ~valid.reshape(-1)
    starts_flat = torch.where(invalid, torch.zeros_like(starts_flat), starts_flat)
    ends_flat = torch.where(invalid, torch.zeros_like(ends_flat), ends_flat)
    return torch.stack([starts_flat, ends_flat], dim=-1).unsqueeze(0)


def _rows(states, mask, groups, group):
    """Return the valid rows of one sample whose group id matches."""
    keep = mask & (groups == group)
    return states[keep]


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
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # trf-ignore: TRF058

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
        nn.init.normal_(self.bos_state, std=0.02)  # trf-ignore: TRF049
        nn.init.normal_(self.eos_state, std=0.02)  # trf-ignore: TRF049

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


MASK_LOGIT = -1.0e4


@dataclass(frozen=True)
class ProposalStats:
    """Element counts and optional gold-recall diagnostics."""

    boundary_score_elements: int
    conditional_pair_score_elements: int
    max_materialized_pair_elements: int
    retained_candidate_count: torch.Tensor
    gold_hit_without_injection: torch.Tensor | None = None
    gold_total: torch.Tensor | None = None
    start_hit: torch.Tensor | None = None
    end_hit: torch.Tensor | None = None
    boundary_total: torch.Tensor | None = None
    unique_candidates: torch.Tensor | None = None


@dataclass
class BoundaryProposals:
    """Padded start/end candidates for one query axis.

    Attributes:
        indices: Half-open `[start, end)` pairs `[B, Q, C, 2]`.
        logits: Full prior, or None when proposal logits were not requested.
        valid_mask: True for real candidates `[B, Q, C]`.
        gold_mask: True where a training candidate was injected.
        stats: Optional proposal diagnostics.
        compat_logits: Marginal-free endpoint compatibility `[B, Q, C]`.
        score_start_states: Gathered reranker start states.
        score_end_states: Gathered reranker end states.
    """

    indices: torch.Tensor
    logits: torch.Tensor | None
    valid_mask: torch.Tensor
    gold_mask: torch.Tensor | None = None
    stats: ProposalStats | None = None
    compat_logits: torch.Tensor | None = None
    score_start_states: torch.Tensor | None = None
    score_end_states: torch.Tensor | None = None


@dataclass
class CandidateTensorBatch:
    """Per-query candidates consumed by decoding.

    Attributes:
        indices: Half-open spans `[B, Q, C, 2]`.
        proposal_logits: Proposal prior `[B, Q, C]`, if retained.
        pair_logits: Reranked pair logits `[B, Q, C]`.
        valid_mask: Real-candidate mask `[B, Q, C]`.
        query_mask: Real-query mask `[B, Q]`.
        candidate_states: Optional endpoint states `[B, Q, C, H]`.
    """

    indices: torch.Tensor
    proposal_logits: torch.Tensor | None
    pair_logits: torch.Tensor
    valid_mask: torch.Tensor
    query_mask: torch.Tensor
    candidate_states: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.indices.dim() != 4 or self.indices.shape[-1] != 2:
            raise ValueError(f"indices must be [B, Q, C, 2], got {tuple(self.indices.shape)}")
        batch, queries, count, _ = self.indices.shape
        if self.proposal_logits is not None and tuple(self.proposal_logits.shape) != (
            batch,
            queries,
            count,
        ):
            raise ValueError("proposal_logits shape must be [B, Q, C]")
        if tuple(self.pair_logits.shape) != (batch, queries, count):
            raise ValueError("pair_logits shape must be [B, Q, C]")
        if tuple(self.valid_mask.shape) != (batch, queries, count):
            raise ValueError("valid_mask shape must be [B, Q, C]")
        if tuple(self.query_mask.shape) != (batch, queries):
            raise ValueError("query_mask shape must be [B, Q]")


def select_top_boundaries(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the top-k boundaries with a stable index tie-break.

    Args:
        logits: Boundary scores `[B, Q, N]`.
        valid_mask: Positions that may be selected.
        k: Maximum boundaries to keep.

    Returns:
        Scores, indices, and a validity mask, each `[B, Q, k]`.
    """
    length = logits.shape[-1]
    k = min(k, length)
    masked = logits.masked_fill(~valid_mask, MASK_LOGIT)
    scores, idx = torch.sort(masked, dim=-1, descending=True, stable=True)
    scores = scores[..., :k]
    idx = idx[..., :k]
    valid = torch.gather(valid_mask, -1, idx)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    return scores, idx, valid


def resolve_boundary_budget(
    n_boundaries: int,
    *,
    base_k: int,
    alpha: float,
    k_max: int,
    bucket: int,
) -> int:
    """Resolve a length-adaptive boundary top-k from the sequence shape.

    Args:
        n_boundaries: Number of boundary positions.
        base_k: Budget used when `alpha` is zero.
        alpha: Fraction of the token length requested as extra budget.
        k_max: Hard cap.
        bucket: Rounding multiple.

    Returns:
        The resolved top-k.
    """
    if alpha <= 0.0:
        return base_k
    requested = int(math.ceil(alpha * max(n_boundaries - 1, 0)))
    requested = max(base_k, min(requested, k_max))
    return min(k_max, int(math.ceil(requested / bucket) * bucket))


def merge_running_topk(
    current_scores: torch.Tensor,
    current_indices: torch.Tensor,
    block_scores: torch.Tensor,
    block_indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a running top-k with the next block, breaking ties stably.

    Args:
        current_scores: Scores retained so far.
        current_indices: Indices retained so far.
        block_scores: Scores from the new block.
        block_indices: Indices from the new block.
        k: Maximum entries to keep.

    Returns:
        Merged scores and indices.
    """
    scores = torch.cat([current_scores, block_scores], dim=-1)
    indices = torch.cat([current_indices, block_indices], dim=-1)
    top_scores, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    take = min(k, scores.shape[-1])
    top_scores = top_scores[..., :take]
    order = order[..., :take]
    top_indices = torch.gather(indices, -1, order)
    return top_scores, top_indices


def _score_ends_blockwise(
    sq: torch.Tensor,
    end_proj_all: torch.Tensor,
    start_indices: torch.Tensor,
    start_scores: torch.Tensor,
    start_valid: torch.Tensor,
    boundary_mask: torch.Tensor,
    query_mask: torch.Tensor,
    end_marginals: torch.Tensor,
    block_size: int,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Stream end blocks and keep the top ends after each selected start.

    Args:
        sq: Gated start states `[B, Q, K, D]`.
        end_proj_all: Projected end states `[B, N, D]`.
        start_indices: Selected start positions `[B, Q, K]`.
        start_scores: Selected start scores `[B, Q, K]`.
        start_valid: Real start slots `[B, Q, K]`.
        boundary_mask: Valid boundaries `[B, N]`.
        query_mask: Valid queries `[B, Q]`.
        end_marginals: End marginals `[B, Q, N]`.
        block_size: End positions scored together.
        top_k: Ends retained per start.
        scale: Compatibility scale.

    Returns:
        Top scores, end indices, scored elements, and the largest block.
    """
    batch, queries, _k, _dim = sq.shape
    n_boundaries = end_proj_all.shape[1]
    device = sq.device
    top_scores = torch.full((batch, queries, _k, top_k), MASK_LOGIT, device=device, dtype=sq.dtype)
    top_idx = torch.zeros((batch, queries, _k, top_k), device=device, dtype=torch.long)
    conditional_elems = 0
    max_block_elems = 0
    for start in range(0, n_boundaries, block_size):
        stop = min(start + block_size, n_boundaries)
        width = stop - start
        block_compat = torch.einsum("bqkd,bed->bqke", sq, end_proj_all[:, start:stop]) * scale
        block = block_compat + end_marginals[:, :, start:stop].unsqueeze(2) + start_scores.unsqueeze(-1)
        end_index = torch.arange(start, stop, device=device)
        keep = (
            boundary_mask[:, start:stop].view(batch, 1, 1, width)
            & query_mask.view(batch, queries, 1, 1)
            & (end_index.view(1, 1, 1, width) > start_indices.unsqueeze(-1))
            & start_valid.unsqueeze(-1)
        )
        block = block.masked_fill(~keep, MASK_LOGIT)
        block_idx = end_index.view(1, 1, 1, width).expand(batch, queries, _k, width)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)
        conditional_elems += batch * queries * _k * width
        max_block_elems = max(max_block_elems, batch * queries * _k * width)
    return top_scores, top_idx, conditional_elems, max_block_elems


def _score_starts_blockwise(
    eq: torch.Tensor,
    start_proj_all: torch.Tensor,
    end_indices: torch.Tensor,
    end_scores: torch.Tensor,
    end_valid: torch.Tensor,
    boundary_mask: torch.Tensor,
    query_mask: torch.Tensor,
    start_marginals: torch.Tensor,
    block_size: int,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Stream start blocks and keep the top starts before each selected end.

    Args:
        eq: Gated end states `[B, Q, K, D]`.
        start_proj_all: Projected start states `[B, N, D]`.
        end_indices: Selected end positions `[B, Q, K]`.
        end_scores: Selected end scores `[B, Q, K]`.
        end_valid: Real end slots `[B, Q, K]`.
        boundary_mask: Valid boundaries `[B, N]`.
        query_mask: Valid queries `[B, Q]`.
        start_marginals: Start marginals `[B, Q, N]`.
        block_size: Start positions scored together.
        top_k: Starts retained per end.
        scale: Compatibility scale.

    Returns:
        Top scores, start indices, scored elements, and the largest block.
    """
    batch, queries, _k, _dim = eq.shape
    n_boundaries = start_proj_all.shape[1]
    device = eq.device
    top_scores = torch.full((batch, queries, _k, top_k), MASK_LOGIT, device=device, dtype=eq.dtype)
    top_idx = torch.zeros((batch, queries, _k, top_k), device=device, dtype=torch.long)
    conditional_elems = 0
    max_block_elems = 0
    for start in range(0, n_boundaries, block_size):
        stop = min(start + block_size, n_boundaries)
        width = stop - start
        block_compat = torch.einsum("bqkd,bed->bqke", eq, start_proj_all[:, start:stop]) * scale
        block = block_compat + start_marginals[:, :, start:stop].unsqueeze(2) + end_scores.unsqueeze(-1)
        start_index = torch.arange(start, stop, device=device)
        keep = (
            boundary_mask[:, start:stop].view(batch, 1, 1, width)
            & query_mask.view(batch, queries, 1, 1)
            & (start_index.view(1, 1, 1, width) < end_indices.unsqueeze(-1))
            & end_valid.unsqueeze(-1)
        )
        block = block.masked_fill(~keep, MASK_LOGIT)
        block_idx = start_index.view(1, 1, 1, width).expand(batch, queries, _k, width)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)
        conditional_elems += batch * queries * _k * width
        max_block_elems = max(max_block_elems, batch * queries * _k * width)
    return top_scores, top_idx, conditional_elems, max_block_elems


def assemble_candidates(
    pair_starts: torch.Tensor,
    pair_ends: torch.Tensor,
    pair_scores: torch.Tensor,
    pair_valid: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    capacity: int,
    n_boundaries: int,
    gold_pairs: torch.Tensor | None = None,
    gold_mask: torch.Tensor | None = None,
    gold_injection_prob: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate pairs, optionally inject gold, and pad to capacity.

    Args:
        pair_starts: Proposed starts `[B, Q, P]`.
        pair_ends: Proposed ends `[B, Q, P]`.
        pair_scores: Selection scores `[B, Q, P]`.
        pair_valid: Real proposals `[B, Q, P]`.
        query_mask: Real queries `[B, Q]`.
        capacity: Output candidate count.
        n_boundaries: Boundary axis length used to pack pair keys.
        gold_pairs: Optional gold spans `[B, Q, G, 2]`.
        gold_mask: Optional gold mask `[B, Q, G]`.
        gold_injection_prob: Probability of keeping each gold span.
        generator: Generator for partial gold injection.

    Returns:
        Indices, validity, gold mask, pre-injection keys, and pre-injection mask.
    """
    dtype = pair_scores.dtype
    floor = MASK_LOGIT
    ceiling = -floor
    invalid_key = n_boundaries * n_boundaries
    pre_valid = pair_valid & query_mask.unsqueeze(-1)
    pre_keys = pair_starts * n_boundaries + pair_ends
    pre_keys = torch.where(pre_valid, pre_keys, torch.full_like(pre_keys, invalid_key))
    keys = pre_keys
    scores = torch.where(pre_valid, pair_scores, torch.full_like(pair_scores, floor))
    valid = pre_valid
    is_gold = torch.zeros_like(valid)

    if gold_pairs is not None and gold_mask is not None:
        gvalid = gold_mask & query_mask.unsqueeze(-1)
        oob = (gold_pairs[..., 0] >= n_boundaries) | (gold_pairs[..., 1] >= n_boundaries) | (gold_pairs < 0).any(-1)
        gvalid = gvalid & ~oob
        if gold_injection_prob <= 0.0:
            gvalid = torch.zeros_like(gvalid)
        elif gold_injection_prob < 1.0:
            sampled = torch.rand(gvalid.shape, device=gvalid.device, generator=generator)
            gvalid = gvalid & (sampled < gold_injection_prob)
        safe_gold = gold_pairs.clamp(0, n_boundaries - 1)
        gkeys = safe_gold[..., 0] * n_boundaries + safe_gold[..., 1]
        gkeys = torch.where(gvalid, gkeys, torch.full_like(gkeys, invalid_key))
        gscores = torch.where(
            gvalid,
            torch.full(gvalid.shape, ceiling, dtype=dtype, device=pair_scores.device),
            torch.full(gvalid.shape, floor, dtype=dtype, device=pair_scores.device),
        )
        keys = torch.cat((keys, gkeys), dim=-1)
        scores = torch.cat((scores, gscores), dim=-1)
        valid = torch.cat((valid, gvalid), dim=-1)
        is_gold = torch.cat((is_gold, gvalid), dim=-1)

    by_score = torch.argsort(scores, dim=-1, descending=True, stable=True)
    keys = torch.gather(keys, -1, by_score)
    scores = torch.gather(scores, -1, by_score)
    valid = torch.gather(valid, -1, by_score)
    is_gold = torch.gather(is_gold, -1, by_score)
    by_key = torch.argsort(keys, dim=-1, stable=True)
    keys = torch.gather(keys, -1, by_key)
    scores = torch.gather(scores, -1, by_key)
    valid = torch.gather(valid, -1, by_key)
    is_gold = torch.gather(is_gold, -1, by_key)

    first = torch.ones_like(valid)
    first[..., 1:] = keys[..., 1:] != keys[..., :-1]
    keep = valid & first
    scores = torch.where(keep, scores, torch.full_like(scores, floor))
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    take = min(capacity, order.shape[-1])
    order = order[..., :take]
    selected_keys = torch.gather(keys, -1, order)
    selected_valid = torch.gather(keep, -1, order)
    selected_gold = torch.gather(is_gold, -1, order) & selected_valid

    starts = torch.div(selected_keys, n_boundaries, rounding_mode="floor")
    ends = selected_keys - starts * n_boundaries
    indices = torch.stack((starts, ends), dim=-1)
    indices = torch.where(selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices))
    if take < capacity:
        pad = capacity - take
        indices = F.pad(indices, (0, 0, 0, pad))
        selected_valid = F.pad(selected_valid, (0, pad), value=False)
        selected_gold = F.pad(selected_gold, (0, pad), value=False)
    return indices, selected_valid, selected_gold, pre_keys, pre_valid


class SparseBoundaryProposer(nn.Module):
    """Select a capped set of start/end pairs for each query."""

    def __init__(self, boundary_dim: int, query_dim: int, settings):
        """Read proposal budgets from `boundary_config`."""
        super().__init__()
        self.boundary_dim = boundary_dim
        self.settings = settings
        self.start_pair_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_key_projection = nn.Linear(boundary_dim, boundary_dim)
        self.start_query_projection = nn.Linear(
            query_dim,
            boundary_dim // 2 if settings.enable_rotary_endpoints else boundary_dim,
        )
        self.rotary = (
            RotaryBoundaryEmbedding(boundary_dim, settings.rotary_base) if settings.enable_rotary_endpoints else None
        )

    def _project(
        self, boundary_states: torch.Tensor, query_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project endpoints and the query gate.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.

        Returns:
            Start states, end states, and the query gate.
        """
        start_all = self.start_pair_projection(boundary_states)
        end_all = self.end_key_projection(boundary_states)
        if self.rotary is not None:
            positions = torch.arange(boundary_states.shape[1], device=boundary_states.device).view(1, -1)
            start_all = self.rotary(start_all, positions)
            end_all = self.rotary(end_all, positions)
        gate = torch.sigmoid(self.start_query_projection(query_states))
        if self.rotary is not None:
            gate = gate.repeat_interleave(2, dim=-1)
        return start_all, end_all, gate

    def score_explicit_pairs(
        self,
        boundary_states: torch.Tensor,
        query_states: torch.Tensor,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score the marginal-free prior at caller-provided spans.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.
            indices: Half-open spans `[B, Q, C, 2]`.
            valid_mask: Spans that receive a finite prior `[B, Q, C]`.

        Returns:
            Compatibility logits `[B, Q, C]`, zero where invalid.
        """
        if indices.dim() != 4 or indices.shape[-1] != 2:
            raise ValueError(f"indices must be [B, Q, C, 2], got {tuple(indices.shape)}")
        if valid_mask.shape != indices.shape[:-1]:
            raise ValueError(
                f"valid_mask must match indices [B, Q, C], got {tuple(valid_mask.shape)} and {tuple(indices.shape)}"
            )
        if (
            indices.shape[0] != boundary_states.shape[0]
            or indices.shape[0] != query_states.shape[0]
            or indices.shape[1] != query_states.shape[1]
        ):
            raise ValueError("explicit pair batch/query dimensions do not match states")

        start_all, end_all, gate = self._project(boundary_states, query_states)
        limit = boundary_states.shape[1] - 1
        starts = indices[..., 0].clamp(0, limit)
        ends = indices[..., 1].clamp(0, limit)
        start_states = gather_states(start_all, starts) * gate.unsqueeze(2)
        end_states = gather_states(end_all, ends)
        compatibility = (start_states * end_states).sum(-1) / math.sqrt(self.boundary_dim)
        return torch.where(valid_mask, compatibility, torch.zeros_like(compatibility))

    def forward(
        self,
        boundary_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        *,
        gold_pairs: torch.Tensor | None = None,
        gold_mask: torch.Tensor | None = None,
        return_stats: bool = False,
        return_proposal_logits: bool = True,
        gold_injection_prob: float = 1.0,
        generator: torch.Generator | None = None,
        scorer_start_states: torch.Tensor | None = None,
        scorer_end_states: torch.Tensor | None = None,
    ) -> BoundaryProposals:
        """Propose capped start/end pairs and their compatibility prior."""
        settings = self.settings
        batch, n_boundaries, _dim = boundary_states.shape
        queries = query_states.shape[1]
        scale = 1.0 / math.sqrt(self.boundary_dim)
        training = self.training and gold_pairs is not None
        capacity = settings.training_candidate_budget if training else settings.candidate_budget
        start_k = resolve_boundary_budget(
            n_boundaries,
            base_k=settings.start_top_k,
            alpha=settings.boundary_top_k_alpha,
            k_max=settings.boundary_top_k_max,
            bucket=settings.boundary_top_k_bucket,
        )
        end_k = resolve_boundary_budget(
            n_boundaries,
            base_k=settings.end_top_k,
            alpha=settings.boundary_top_k_alpha,
            k_max=settings.boundary_top_k_max,
            bucket=settings.boundary_top_k_bucket,
        )
        pair_elements = batch * queries * start_k * n_boundaries
        use_vectorized = settings.export_mode == "vectorized" or (
            settings.export_mode == "auto" and pair_elements <= settings.vectorized_pair_elements
        )
        end_block = n_boundaries if use_vectorized else settings.end_block_size
        b_valid = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)

        start_proj_all, end_proj_all, gate = self._project(boundary_states, query_states)
        select_start = start_proj_all.detach()
        select_end = end_proj_all.detach()
        select_gate = gate.detach()
        select_start_logits = start_logits.detach()
        select_end_logits = end_logits.detach()

        with torch.no_grad():
            st_scores, st_idx, st_valid = select_top_boundaries(select_start_logits, b_valid, start_k)
            sq = gather_states(select_start, st_idx) * select_gate.unsqueeze(2)
            fwd_scores, fwd_end_idx, cond_e1, maxe1 = _score_ends_blockwise(
                sq,
                select_end,
                st_idx,
                st_scores,
                st_valid,
                boundary_mask,
                query_mask,
                select_end_logits,
                end_block,
                settings.ends_per_start,
                scale,
            )
        fwd_start = st_idx.unsqueeze(-1).expand(-1, -1, -1, settings.ends_per_start)
        fwd_pairs_s = fwd_start.reshape(batch, queries, -1)
        fwd_pairs_e = fwd_end_idx.reshape(batch, queries, -1)
        fwd_pairs_sc = fwd_scores.reshape(batch, queries, -1)
        fwd_pairs_valid = (
            st_valid.unsqueeze(-1)
            & query_mask.view(batch, queries, 1, 1)
            & boundary_mask.gather(1, fwd_end_idx.reshape(batch, -1).clamp(0, boundary_mask.shape[1] - 1)).view_as(
                fwd_end_idx
            )
            & (fwd_end_idx > fwd_start)
        ).reshape(batch, queries, -1)

        pair_starts = [fwd_pairs_s]
        pair_ends = [fwd_pairs_e]
        pair_scores = [fwd_pairs_sc]
        pair_valid = [fwd_pairs_valid]
        cond_e2 = 0
        maxe2 = 0
        en_idx = en_valid = None
        if settings.bidirectional_proposals:
            with torch.no_grad():
                en_scores, en_idx, en_valid = select_top_boundaries(select_end_logits, b_valid, end_k)
                eq = gather_states(select_end, en_idx) * select_gate.unsqueeze(2)
                bwd_scores, bwd_start_idx, cond_e2, maxe2 = _score_starts_blockwise(
                    eq,
                    select_start,
                    en_idx,
                    en_scores,
                    en_valid,
                    boundary_mask,
                    query_mask,
                    select_start_logits,
                    end_block,
                    settings.starts_per_end,
                    scale,
                )
            bwd_end = en_idx.unsqueeze(-1).expand(-1, -1, -1, settings.starts_per_end)
            pair_starts.append(bwd_start_idx.reshape(batch, queries, -1))
            pair_ends.append(bwd_end.reshape(batch, queries, -1))
            pair_scores.append(bwd_scores.reshape(batch, queries, -1))
            pair_valid.append(
                (
                    en_valid.unsqueeze(-1)
                    & query_mask.view(batch, queries, 1, 1)
                    & boundary_mask.gather(
                        1,
                        bwd_start_idx.reshape(batch, -1).clamp(0, boundary_mask.shape[1] - 1),
                    ).view_as(bwd_start_idx)
                    & (bwd_end > bwd_start_idx)
                ).reshape(batch, queries, -1)
            )

        all_s = torch.cat(pair_starts, dim=-1)
        all_e = torch.cat(pair_ends, dim=-1)
        all_sc = torch.cat(pair_scores, dim=-1).detach()
        all_valid = torch.cat(pair_valid, dim=-1)
        with torch.no_grad():
            out_idx, out_valid, out_gold, pre_keys, pre_valid = assemble_candidates(
                all_s,
                all_e,
                all_sc,
                all_valid,
                query_mask,
                capacity=capacity,
                n_boundaries=n_boundaries,
                gold_pairs=gold_pairs if training else None,
                gold_mask=gold_mask if training else None,
                gold_injection_prob=gold_injection_prob,
                generator=generator,
            )

        si = out_idx[..., 0]
        ej = out_idx[..., 1]
        score_start_selected = score_end_selected = None
        if scorer_start_states is not None and scorer_end_states is not None:
            prop_dim = start_proj_all.shape[-1]
            score_dim = scorer_start_states.shape[-1]
            start_all = torch.cat((start_proj_all, scorer_start_states), dim=-1)
            end_all = torch.cat((end_proj_all, scorer_end_states), dim=-1)
            g_s, score_start_selected = gather_states(start_all, si).split((prop_dim, score_dim), dim=-1)
            g_e, score_end_selected = gather_states(end_all, ej).split((prop_dim, score_dim), dim=-1)
        else:
            g_s = gather_states(start_proj_all, si)
            g_e = gather_states(end_proj_all, ej)
        g_s = g_s * gate.unsqueeze(2)
        compat = (g_s * g_e).sum(-1) * scale
        out_logits = None
        if self.training or return_proposal_logits:
            sm = torch.gather(start_logits, 2, si.clamp(0, start_logits.shape[2] - 1))
            em = torch.gather(end_logits, 2, ej.clamp(0, end_logits.shape[2] - 1))
            logits_diff = compat + sm + em
            out_logits = torch.where(out_valid, logits_diff, torch.full_like(logits_diff, MASK_LOGIT))
        out_compat = torch.where(out_valid, compat, torch.zeros_like(compat))

        stats = None
        if return_stats:
            retained = out_valid.sum()
            gold_hit = gold_total = start_hit = end_hit = boundary_total = None
            if gold_pairs is not None and gold_mask is not None:
                diagnostic_gold = gold_mask & query_mask.unsqueeze(-1)
                gold_keys = gold_pairs[..., 0] * n_boundaries + gold_pairs[..., 1]
                pair_hit = ((gold_keys.unsqueeze(-1) == pre_keys.unsqueeze(-2)) & pre_valid.unsqueeze(-2)).any(
                    -1
                ) & diagnostic_gold
                gold_hit = pair_hit.sum()
                gold_total = diagnostic_gold.sum()
                selected_starts = st_idx
                selected_starts_valid = st_valid
                if settings.bidirectional_proposals:
                    selected_ends = en_idx
                    selected_ends_valid = en_valid
                else:
                    selected_ends = fwd_end_idx.reshape(batch, queries, -1)
                    selected_ends_valid = fwd_pairs_valid
                start_hit = (
                    (gold_pairs[..., 0].unsqueeze(-1) == selected_starts.unsqueeze(-2))
                    & selected_starts_valid.unsqueeze(-2)
                ).any(-1) & diagnostic_gold
                end_hit = (
                    (gold_pairs[..., 1].unsqueeze(-1) == selected_ends.unsqueeze(-2))
                    & selected_ends_valid.unsqueeze(-2)
                ).any(-1) & diagnostic_gold
                start_hit = start_hit.sum()
                end_hit = end_hit.sum()
                boundary_total = diagnostic_gold.sum()
            stats = ProposalStats(
                boundary_score_elements=batch * queries * n_boundaries * 2,
                conditional_pair_score_elements=cond_e1 + cond_e2,
                max_materialized_pair_elements=max(maxe1, maxe2),
                retained_candidate_count=retained,
                gold_hit_without_injection=gold_hit,
                gold_total=gold_total,
                start_hit=start_hit,
                end_hit=end_hit,
                boundary_total=boundary_total,
                unique_candidates=retained,
            )
        return BoundaryProposals(
            indices=out_idx,
            logits=out_logits,
            valid_mask=out_valid,
            gold_mask=out_gold if training else None,
            stats=stats,
            compat_logits=out_compat,
            score_start_states=score_start_selected,
            score_end_states=score_end_selected,
        )


MASK_LOGIT = -1.0e4


def gather_boundary_states(boundary_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather boundary states at per-query candidate indices.

    Args:
        boundary_states: States `[B, N, D]`.
        indices: Positions `[B, Q, C]`.

    Returns:
        Gathered states `[B, Q, C, D]`.
    """
    return gather_states(boundary_states, indices)


def interval_prefix_score(
    prefix: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score `[start, end)` from a length-`L + 1` prefix.

    Args:
        prefix: Cumulative scores `[B, Q, L + 1]`.
        starts: Start indices `[B, Q, C]`.
        ends: End indices `[B, Q, C]`.
        mean: Optional per-query offset restored onto the interval.

    Returns:
        Interval scores `[B, Q, C]`.
    """
    max_idx = prefix.shape[2] - 1
    starts = starts.clamp(0, max_idx)
    ends = ends.clamp(0, max_idx)
    interval = torch.gather(prefix, 2, ends) - torch.gather(prefix, 2, starts)
    if mean is not None:
        interval = interval + mean * (ends - starts).to(interval.dtype)
    return interval


def continuous_length_features(
    starts: torch.Tensor,
    ends: torch.Tensor,
    text_lengths: torch.Tensor,
) -> torch.Tensor:
    """Build length features that do not depend on a maximum span width.

    Args:
        starts: Start indices `[B, Q, C]`.
        ends: End indices `[B, Q, C]`.
        text_lengths: Token counts `[B]`.

    Returns:
        Features `[B, Q, C, 3]`.
    """
    length = (ends - starts).clamp(min=1).float()
    batch = starts.shape[0]
    text_length = text_lengths.view(batch, 1, 1).float().clamp(min=1)
    return torch.stack(
        (torch.log1p(length), length / text_length, torch.rsqrt(length)),
        dim=-1,
    )


def mask_invalid_candidate_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Replace invalid candidate logits with a finite sentinel.

    Args:
        logits: Candidate scores.
        valid_mask: True for real candidates.

    Returns:
        Masked logits.
    """
    return logits.masked_fill(~valid_mask, MASK_LOGIT)


class SparseBoundaryPairScorer(nn.Module):
    """Rerank each proposed span with marginals, content, and length.

    The added prior is marginal-free endpoint compatibility, so start and end
    marginals enter the logit once.

    Args:
        boundary_dim: Boundary state width.
        query_dim: Query state width.
        pair_dim: Endpoint compatibility width.
        use_inside_evidence: Add normalized inside-prefix evidence.
        dropout: Dropout probability.
        enable_span_content: Pool token content into the score.
        content_dim: Content projection width.
        content_soft_max_pool: Also pool a smooth maximum.
        enable_rotary_endpoints: Rotate reranker endpoints.
        rotary_base: Rotary frequency base.
        query_conditioned_inside_weight: Predict the inside coefficient.
        endpoint_difference_features: Add a learned endpoint-difference term.
        reranker_endpoint_compat: Add multi-head endpoint compatibility.
        multihead_pair_compat_heads: Heads mixed into one compatibility logit.
        content_hidden_size: Token width used by the content pooler.
    """

    def __init__(
        self,
        boundary_dim: int,
        query_dim: int,
        pair_dim: int,
        use_inside_evidence: bool = True,
        dropout: float = 0.1,
        enable_span_content: bool = False,
        content_dim: int = 64,
        content_soft_max_pool: bool = False,
        enable_rotary_endpoints: bool = False,
        rotary_base: float = 10000.0,
        query_conditioned_inside_weight: bool = False,
        endpoint_difference_features: bool = False,
        reranker_endpoint_compat: bool = True,
        multihead_pair_compat_heads: int = 1,
        content_hidden_size: int | None = None,
    ):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.pair_dim = pair_dim
        self.use_inside_evidence = use_inside_evidence
        self.enable_span_content = enable_span_content
        self.enable_rotary_endpoints = enable_rotary_endpoints
        self.query_conditioned_inside_weight = query_conditioned_inside_weight
        self.endpoint_difference_features = endpoint_difference_features
        self.reranker_endpoint_compat = reranker_endpoint_compat
        if multihead_pair_compat_heads <= 0 or pair_dim % multihead_pair_compat_heads:
            raise ValueError(
                "pair_dim must be divisible by multihead_pair_compat_heads, got "
                f"{pair_dim} and {multihead_pair_compat_heads}"
            )
        self.multihead_pair_compat_heads = multihead_pair_compat_heads
        self.start_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.end_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.query_gate = nn.Linear(query_dim, pair_dim // 2 if enable_rotary_endpoints else pair_dim)
        self.length_query_projection = nn.Linear(query_dim, 3)
        self.inside_weight = (
            nn.Linear(query_dim, 1) if query_conditioned_inside_weight else nn.Parameter(torch.tensor(1.0))
        )
        self.endpoint_difference_projection = nn.Linear(2 * pair_dim, 1) if endpoint_difference_features else None
        self.rotary = RotaryBoundaryEmbedding(pair_dim, rotary_base) if enable_rotary_endpoints else None
        self.content_pooler = (
            SpanContentPooler(
                content_hidden_size if content_hidden_size is not None else query_dim,
                content_dim,
                dropout=dropout,
                use_soft_max_pool=content_soft_max_pool,
            )
            if enable_span_content
            else None
        )
        content_output_dim = self.content_pooler.output_dim if self.content_pooler is not None else 0
        self.content_query_projection = nn.Linear(query_dim, content_output_dim) if enable_span_content else None
        self.content_bias = nn.Linear(content_output_dim, 1) if enable_span_content else None
        self.dropout = nn.Dropout(dropout)
        # Keep this layer last so its init does not shift the caller RNG stream.
        rng_state = torch.random.get_rng_state()
        self.compat_mix = nn.Linear(multihead_pair_compat_heads, 1)
        nn.init.constant_(self.compat_mix.weight, 1.0 / multihead_pair_compat_heads)  # trf-ignore: TRF049
        nn.init.zeros_(self.compat_mix.bias)  # trf-ignore: TRF049
        torch.random.set_rng_state(rng_state)

    def project_endpoints(self, boundary_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project every boundary before candidate gathering.

        Args:
            boundary_states: Boundary states `[B, N, D]`.

        Returns:
            Start and end projections `[B, N, P]`.
        """
        start = self.start_endpoint_projection(boundary_states)
        end = self.end_endpoint_projection(boundary_states)
        if self.rotary is not None:
            n_boundaries = boundary_states.shape[1]
            positions = torch.arange(n_boundaries, device=boundary_states.device).view(1, n_boundaries)
            start = self.rotary(start, positions)
            end = self.rotary(end, positions)
        return start, end

    def forward(
        self,
        boundary_states: torch.Tensor,
        query_states: torch.Tensor,
        proposals: BoundaryProposals,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        inside_prefix: torch.Tensor | None,
        text_lengths: torch.Tensor,
        text_states: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        inside_prefix_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one logit per proposed candidate.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.
            proposals: Sparse candidates, including a compatibility prior.
            start_logits: Start marginals `[B, Q, N]`.
            end_logits: End marginals `[B, Q, N]`.
            inside_prefix: Centered inside prefix, or None to skip it.
            text_lengths: Token counts `[B]`.
            text_states: Token states required when content pooling is enabled.
            text_mask: Token mask required when content pooling is enabled.
            inside_prefix_mean: Mean restored onto inside intervals.

        Returns:
            Masked pair logits `[B, Q, C]`.
        """
        starts = proposals.indices[..., 0]
        ends = proposals.indices[..., 1]
        valid = proposals.valid_mask
        scale = 1.0 / math.sqrt(self.pair_dim)
        if proposals.score_start_states is not None and proposals.score_end_states is not None:
            s_proj = self.dropout(proposals.score_start_states)
            e_proj = self.dropout(proposals.score_end_states)
        else:
            start_all, end_all = self.project_endpoints(boundary_states)
            s_proj = self.dropout(gather_boundary_states(start_all, starts))
            e_proj = self.dropout(gather_boundary_states(end_all, ends))
        gate = torch.sigmoid(self.query_gate(query_states))
        if self.enable_rotary_endpoints:
            gate = gate.repeat_interleave(2, dim=-1)
        gate = gate.unsqueeze(2)
        if self.reranker_endpoint_compat:
            per_head = (
                (s_proj * gate * e_proj).reshape(*s_proj.shape[:-1], self.multihead_pair_compat_heads, -1).sum(-1)
            )
            compat = self.compat_mix(per_head).squeeze(-1) * scale
        else:
            compat = torch.zeros_like(starts, dtype=s_proj.dtype)
        if self.endpoint_difference_projection is not None:
            difference = torch.cat((s_proj - e_proj, (s_proj - e_proj).abs()), dim=-1)
            compat = compat + self.endpoint_difference_projection(difference).squeeze(-1)

        a = torch.gather(start_logits, 2, starts.clamp(0, start_logits.shape[2] - 1))
        bmarg = torch.gather(end_logits, 2, ends.clamp(0, end_logits.shape[2] - 1))
        prior_source = proposals.compat_logits if proposals.compat_logits is not None else proposals.logits
        if prior_source is None:
            raise ValueError("boundary proposals must provide compat_logits or logits")
        prior = torch.where(valid, prior_source, torch.zeros_like(prior_source))
        score = compat + a + bmarg + prior

        if self.content_pooler is not None:
            if text_states is None or text_mask is None:
                raise ValueError("span content scoring requires text_states and text_mask")
            mean_prefix, lse_prefix = self.content_pooler.build_prefix(text_states, text_mask)
            span_content = self.content_pooler.pool(mean_prefix, lse_prefix, starts, ends, score.dtype)
            coefficient = self.content_query_projection(query_states).unsqueeze(2)
            content_scale = 1.0 / math.sqrt(span_content.shape[-1])
            score = score + (span_content * coefficient).sum(-1) * content_scale
            score = score + self.content_bias(span_content).squeeze(-1)

        if self.use_inside_evidence and inside_prefix is not None:
            interval = interval_prefix_score(inside_prefix, starts, ends, inside_prefix_mean).to(score.dtype)
            denom = torch.sqrt((ends - starts).clamp(min=1).to(score.dtype))
            inside_weight = (
                self.inside_weight(query_states).squeeze(-1).unsqueeze(-1)
                if self.query_conditioned_inside_weight
                else self.inside_weight
            )
            score = score + inside_weight * (interval / denom)

        feats = continuous_length_features(starts, ends, text_lengths)
        length_coeff = self.length_query_projection(query_states).unsqueeze(2)
        score = score + (feats.to(length_coeff.dtype) * length_coeff).sum(-1)
        return mask_invalid_candidate_logits(score, valid)


MASK_LOGIT = -1.0e4

OVERLAP_IDENTICAL = 0
OVERLAP_NESTED_INSIDE = 1
OVERLAP_NESTED_OUTSIDE = 2
OVERLAP_CROSSING = 3
OVERLAP_SAME_START = 4
OVERLAP_SAME_END = 5
OVERLAP_DISJOINT_LEFT = 6
OVERLAP_DISJOINT_RIGHT = 7
NUM_OVERLAP_BUCKETS = 8


@dataclass
class PooledCandidates:
    """One document-level span pool.

    Attributes:
        indices: Half-open spans `[B, C, 2]`.
        mask: Real pool entries `[B, C]`.
        proposal_logits: Query-agnostic proposal scores `[B, C]`.
        gold_mask: Gold membership `[B, C, Q]`.
        compat_logits: Marginal-free compatibility `[B, C]`.
        stats: Optional pool diagnostics.
    """

    indices: torch.Tensor
    mask: torch.Tensor
    proposal_logits: torch.Tensor | None
    gold_mask: torch.Tensor | None
    compat_logits: torch.Tensor | None = None
    stats: ProposalStats | None = None

    def to_candidate_batch(
        self,
        pair_logits: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_states: torch.Tensor | None = None,
    ) -> CandidateTensorBatch:
        """Broadcast the document pool onto the per-query candidate layout.

        Args:
            pair_logits: Query scores in candidate-major order `[B, C, Q]`.
            query_mask: Valid queries `[B, Q]`.
            candidate_states: Optional pool states `[B, C, H]`.

        Returns:
            Candidates with query-major pair logits `[B, Q, C]`.
        """
        batch, count, _ = self.indices.shape
        queries = query_mask.shape[1]
        indices = self.indices.unsqueeze(1).expand(batch, queries, count, 2)
        valid = self.mask.unsqueeze(1).expand(batch, queries, count)
        proposal = (
            self.proposal_logits.unsqueeze(1).expand(batch, queries, count)
            if self.proposal_logits is not None
            else None
        )
        states = (
            candidate_states.unsqueeze(1).expand(batch, queries, count, candidate_states.shape[-1])
            if candidate_states is not None
            else None
        )
        return CandidateTensorBatch(
            indices=indices,
            proposal_logits=proposal,
            pair_logits=pair_logits.transpose(1, 2),
            valid_mask=valid,
            query_mask=query_mask,
            candidate_states=states,
        )


def _deduplicate_pool(
    keys: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    capacity: int,
    n_boundaries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the highest-priority copy of each document span key.

    Args:
        keys: Packed start/end keys `[B, P]`.
        scores: Priority scores `[B, P]`.
        valid: Real keys `[B, P]`.
        capacity: Output width.
        n_boundaries: Boundary count used to mark invalid keys.

    Returns:
        Selected keys and their validity mask, padded to `capacity`.
    """
    invalid_key = n_boundaries * n_boundaries
    keys = torch.where(valid, keys, torch.full_like(keys, invalid_key))
    scores = torch.where(valid, scores, torch.full_like(scores, MASK_LOGIT))
    by_score = torch.argsort(scores, dim=-1, descending=True, stable=True)
    keys = keys.gather(-1, by_score)
    scores = scores.gather(-1, by_score)
    valid = valid.gather(-1, by_score)
    by_key = torch.argsort(keys, dim=-1, stable=True)
    keys = keys.gather(-1, by_key)
    scores = scores.gather(-1, by_key)
    valid = valid.gather(-1, by_key)
    first = torch.ones_like(valid)
    first[..., 1:] = keys[..., 1:] != keys[..., :-1]
    keep = valid & first
    order = torch.argsort(
        torch.where(keep, scores, torch.full_like(scores, MASK_LOGIT)),
        dim=-1,
        descending=True,
        stable=True,
    )[..., :capacity]
    selected_keys = keys.gather(-1, order)
    selected_valid = keep.gather(-1, order)
    if selected_keys.shape[-1] < capacity:
        pad = capacity - selected_keys.shape[-1]
        selected_keys = F.pad(selected_keys, (0, pad))
        selected_valid = F.pad(selected_valid, (0, pad), value=False)
    return selected_keys, selected_valid


class DocumentCandidatePool(nn.Module):
    """Build one deduplicated span pool per document.

    Args:
        boundary_dim: Boundary state width.
        pool_boundary_top_k: Endpoints kept from the query-union marginals.
        pool_size: Maximum spans retained per document.
        min_pool_per_query: Spans reserved for each active query.
    """

    def __init__(
        self,
        boundary_dim: int,
        *,
        pool_boundary_top_k: int,
        pool_size: int,
        min_pool_per_query: int,
    ) -> None:
        super().__init__()
        self.pool_boundary_top_k = pool_boundary_top_k
        self.pool_size = pool_size
        self.min_pool_per_query = min_pool_per_query
        self.start_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_projection = nn.Linear(boundary_dim, boundary_dim)

    def forward(
        self,
        boundary_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        query_mask: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        *,
        gold_pairs: torch.Tensor | None = None,
        gold_mask: torch.Tensor | None = None,
        gold_injection_prob: float = 1.0,
        return_stats: bool = False,
        generator: torch.Generator | None = None,
    ) -> PooledCandidates:
        """Pair union endpoints once and keep a capped document pool.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            boundary_mask: Valid boundaries `[B, N]`.
            query_mask: Valid queries `[B, Q]`.
            start_logits: Start marginals `[B, Q, N]`.
            end_logits: End marginals `[B, Q, N]`.
            gold_pairs: Optional gold spans `[B, Q, G, 2]`.
            gold_mask: Mask for `gold_pairs`.
            gold_injection_prob: Fraction of gold spans forced into the pool.
            return_stats: Populate recall diagnostics.
            generator: Generator for partial gold injection.

        Returns:
            A padded document pool.
        """
        batch, n_boundaries, dim = boundary_states.shape
        queries = query_mask.shape[1]
        floor = torch.full_like(start_logits, MASK_LOGIT)
        q_boundary = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
        union_start = torch.where(q_boundary, start_logits, floor).amax(1)
        union_end = torch.where(q_boundary, end_logits, floor).amax(1)
        union_valid = boundary_mask & query_mask.any(-1, keepdim=True)

        _, starts, starts_valid = select_top_boundaries(
            union_start.unsqueeze(1),
            union_valid.unsqueeze(1),
            self.pool_boundary_top_k,
        )
        _, ends, ends_valid = select_top_boundaries(
            union_end.unsqueeze(1),
            union_valid.unsqueeze(1),
            self.pool_boundary_top_k,
        )
        starts = starts[:, 0]
        ends = ends[:, 0]
        starts_valid = starts_valid[:, 0]
        ends_valid = ends_valid[:, 0]
        ks, ke = starts.shape[1], ends.shape[1]
        pair_s = starts.unsqueeze(-1).expand(batch, ks, ke).reshape(batch, -1)
        pair_e = ends.unsqueeze(1).expand(batch, ks, ke).reshape(batch, -1)
        pair_valid = (
            starts_valid.unsqueeze(-1) & ends_valid.unsqueeze(1) & (ends.unsqueeze(1) > starts.unsqueeze(-1))
        ).reshape(batch, -1)

        start_all = self.start_projection(boundary_states)
        end_all = self.end_projection(boundary_states)
        selected_start = gather_rows(start_all, pair_s)
        selected_end = gather_rows(end_all, pair_e)
        compat = (selected_start * selected_end).sum(-1) / math.sqrt(dim)
        union_pair_score = (
            compat
            + union_start.gather(1, pair_s.clamp(0, n_boundaries - 1))
            + union_end.gather(1, pair_e.clamp(0, n_boundaries - 1))
        )

        quota = min(self.min_pool_per_query, pair_s.shape[-1])
        quota_keys = pair_s.new_zeros((batch, 0))
        quota_scores = union_pair_score.new_zeros((batch, 0))
        quota_valid = pair_valid.new_zeros((batch, 0))
        if quota:
            s_idx = pair_s.clamp(0, start_logits.shape[2] - 1).unsqueeze(1).expand(batch, queries, -1)
            e_idx = pair_e.clamp(0, end_logits.shape[2] - 1).unsqueeze(1).expand(batch, queries, -1)
            per_query = start_logits.gather(2, s_idx) + end_logits.gather(2, e_idx) + compat.unsqueeze(1)
            per_query_valid = pair_valid.unsqueeze(1) & query_mask.unsqueeze(-1)
            ranked = torch.argsort(
                per_query.masked_fill(~per_query_valid, MASK_LOGIT),
                dim=-1,
                descending=True,
                stable=True,
            )[..., :quota]
            quota_s = s_idx.gather(-1, ranked)
            quota_e = e_idx.gather(-1, ranked)
            quota_valid = per_query_valid.gather(-1, ranked).reshape(batch, -1)
            quota_keys = (quota_s * n_boundaries + quota_e).reshape(batch, -1)
            rank_bonus = torch.arange(
                quota,
                0,
                -1,
                device=boundary_states.device,
                dtype=union_pair_score.dtype,
            )
            quota_scores = (
                union_pair_score.new_full((batch, queries, quota), -MASK_LOGIT * 0.5) + rank_bonus.view(1, 1, quota)
            ).reshape(batch, -1)

        global_keys = pair_s * n_boundaries + pair_e
        all_keys = torch.cat((quota_keys, global_keys), -1)
        all_scores = torch.cat((quota_scores, union_pair_score.detach()), -1)
        all_valid = torch.cat((quota_valid, pair_valid), -1)
        diagnostic_keys = diagnostic_valid = None
        if return_stats:
            with torch.no_grad():
                diagnostic_keys, diagnostic_valid = _deduplicate_pool(
                    all_keys, all_scores, all_valid, self.pool_size, n_boundaries
                )

        if gold_pairs is not None and gold_mask is not None:
            gvalid = gold_mask & query_mask.unsqueeze(-1)
            oob = (
                (gold_pairs[..., 0] >= n_boundaries) | (gold_pairs[..., 1] >= n_boundaries) | (gold_pairs < 0).any(-1)
            )
            gvalid = gvalid & ~oob
            if gold_injection_prob <= 0.0:
                gvalid = torch.zeros_like(gvalid)
            elif gold_injection_prob < 1.0:
                sampled = torch.rand(gvalid.shape, device=gvalid.device, generator=generator)
                gvalid = gvalid & (sampled < gold_injection_prob)
            safe_gold = gold_pairs.clamp(0, n_boundaries - 1)
            gkeys = safe_gold[..., 0] * n_boundaries + safe_gold[..., 1]
            all_keys = torch.cat((all_keys, gkeys.reshape(batch, -1)), -1)
            all_valid = torch.cat((all_valid, gvalid.reshape(batch, -1)), -1)
            gold_priority = union_pair_score.new_full((batch, gkeys.shape[1] * gkeys.shape[2]), -MASK_LOGIT)
            all_scores = torch.cat((all_scores, gold_priority), -1)

        with torch.no_grad():
            selected_keys, selected_valid = _deduplicate_pool(
                all_keys, all_scores, all_valid, self.pool_size, n_boundaries
            )
        selected_keys = torch.where(selected_valid, selected_keys, torch.zeros_like(selected_keys))
        selected_s = torch.div(selected_keys, n_boundaries, rounding_mode="floor")
        selected_e = selected_keys - selected_s * n_boundaries
        indices = torch.stack((selected_s, selected_e), -1)
        indices = torch.where(selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices))

        gs = gather_rows(start_all, selected_s)
        ge = gather_rows(end_all, selected_e)
        selected_compat = (gs * ge).sum(-1) / math.sqrt(dim)
        selected_score = (
            selected_compat
            + union_start.gather(1, selected_s.clamp(0, n_boundaries - 1))
            + union_end.gather(1, selected_e.clamp(0, n_boundaries - 1))
        )
        selected_score = selected_score.masked_fill(~selected_valid, MASK_LOGIT)
        selected_compat = torch.where(selected_valid, selected_compat, torch.zeros_like(selected_compat))

        selected_gold = None
        if gold_pairs is not None and gold_mask is not None:
            selected_gold = (indices.unsqueeze(2).unsqueeze(3) == gold_pairs.unsqueeze(1)).all(-1)
            selected_gold = (selected_gold & gold_mask.unsqueeze(1) & selected_valid.unsqueeze(-1).unsqueeze(-1)).any(
                -1
            )

        stats = None
        if return_stats:
            gold_hit = gold_total = start_hit = end_hit = boundary_total = None
            if gold_pairs is not None and gold_mask is not None:
                diagnostic_gold = gold_mask & query_mask.unsqueeze(-1)
                gold_keys = gold_pairs[..., 0] * n_boundaries + gold_pairs[..., 1]
                pair_hit = (
                    (gold_keys.unsqueeze(-1) == diagnostic_keys.unsqueeze(1).unsqueeze(1))
                    & diagnostic_valid.unsqueeze(1).unsqueeze(1)
                ).any(-1) & diagnostic_gold
                gold_hit = pair_hit.sum()
                gold_total = diagnostic_gold.sum()
                start_hit = (
                    (gold_pairs[..., 0].unsqueeze(-1) == starts.unsqueeze(1).unsqueeze(1))
                    & starts_valid.unsqueeze(1).unsqueeze(1)
                ).any(-1) & diagnostic_gold
                end_hit = (
                    (gold_pairs[..., 1].unsqueeze(-1) == ends.unsqueeze(1).unsqueeze(1))
                    & ends_valid.unsqueeze(1).unsqueeze(1)
                ).any(-1) & diagnostic_gold
                start_hit, end_hit = start_hit.sum(), end_hit.sum()
                boundary_total = diagnostic_gold.sum()
            stats = ProposalStats(
                boundary_score_elements=batch * queries * n_boundaries * 2,
                conditional_pair_score_elements=batch * ks * ke,
                max_materialized_pair_elements=batch * ks * ke,
                retained_candidate_count=selected_valid.sum(),
                gold_hit_without_injection=gold_hit,
                gold_total=gold_total,
                start_hit=start_hit,
                end_hit=end_hit,
                boundary_total=boundary_total,
                unique_candidates=selected_valid.sum(),
            )
        return PooledCandidates(
            indices=indices,
            mask=selected_valid,
            proposal_logits=selected_score,
            gold_mask=selected_gold,
            compat_logits=selected_compat,
            stats=stats,
        )


def classify_overlap_buckets(indices: torch.Tensor) -> torch.Tensor:
    """Classify every ordered span pair into one of eight geometry buckets.

    Args:
        indices: Half-open spans `[B, C, 2]`.

    Returns:
        Bucket ids `[B, C, C]`.
    """
    s1 = indices[..., :, None, 0]
    e1 = indices[..., :, None, 1]
    s2 = indices[..., None, :, 0]
    e2 = indices[..., None, :, 1]
    out = torch.full_like(s1 + s2, OVERLAP_CROSSING)
    disjoint_left = e1 <= s2
    disjoint_right = e2 <= s1
    identical = (s1 == s2) & (e1 == e2)
    same_start = (s1 == s2) & (e1 != e2)
    same_end = (e1 == e2) & (s1 != s2)
    nested_inside = (s1 > s2) & (e1 < e2)
    nested_outside = (s1 < s2) & (e1 > e2)
    out = torch.where(disjoint_left, OVERLAP_DISJOINT_LEFT, out)
    out = torch.where(disjoint_right, OVERLAP_DISJOINT_RIGHT, out)
    out = torch.where(nested_inside, OVERLAP_NESTED_INSIDE, out)
    out = torch.where(nested_outside, OVERLAP_NESTED_OUTSIDE, out)
    out = torch.where(same_start, OVERLAP_SAME_START, out)
    out = torch.where(same_end, OVERLAP_SAME_END, out)
    return torch.where(identical, OVERLAP_IDENTICAL, out)


class OverlapBiasedCandidateAttention(nn.Module):
    """Candidate self-attention biased by span overlap geometry.

    Args:
        dim: Candidate feature width.
        heads: Attention heads.
        dropout: Residual dropout probability.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(NUM_OVERLAP_BUCKETS, heads))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(  # trf-ignore: TRF036
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attend across pool spans and clear padding rows.

        Args:
            states: Candidate states `[B, C, D]`.
            indices: Half-open spans `[B, C, 2]`.
            mask: Real candidates `[B, C]`.

        Returns:
            Updated candidate states.
        """
        batch, count, dim = states.shape
        qkv = self.qkv(self.norm1(states)).reshape(batch, count, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        buckets = classify_overlap_buckets(indices)
        bias = self.relative_bias[buckets].permute(0, 3, 1, 2)
        logits = logits + bias
        logits = logits.masked_fill(~mask[:, None, None, :], MASK_LOGIT)
        weights = torch.softmax(logits, -1)
        weights = weights * mask[:, None, :, None].to(weights.dtype)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, count, dim)
        states = states + self.dropout(self.output(attended))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states * mask.unsqueeze(-1).to(states.dtype)


class EvidenceConditionedQueryAttention(nn.Module):
    """Update queries after adding a pool-evidence residual.

    Args:
        query_dim: Incoming query width.
        model_dim: Attention width.
        heads: Attention heads.
        dropout: Feed-forward and attention dropout.
    """

    def __init__(self, query_dim: int, model_dim: int, heads: int, dropout: float):
        super().__init__()
        self.query_in = nn.Linear(query_dim, model_dim)
        self.evidence_in = nn.Linear(model_dim, model_dim)
        self.attention = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(  # trf-ignore: TRF036
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(
        self,
        query_states: torch.Tensor,
        evidence: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run query self-attention conditioned on pooled evidence.

        Args:
            query_states: Projected queries `[B, Q, D]`.
            evidence: Evidence summarized from the pool `[B, Q, D]`.
            query_mask: Valid queries `[B, Q]`.

        Returns:
            Updated queries, with padding rows cleared.
        """
        states = self.query_in(query_states) + self.evidence_in(evidence)
        safe_query_mask = query_mask.clone()
        safe_query_mask[:, 0] |= ~query_mask.any(-1)
        attended, _ = self.attention(
            self.norm1(states),
            self.norm1(states),
            self.norm1(states),
            key_padding_mask=~safe_query_mask,
            need_weights=False,
        )
        states = states + attended
        states = states + self.ffn(self.norm2(states))
        return states * query_mask.unsqueeze(-1).to(states.dtype)


class SharedPoolScorer(nn.Module):
    """Score every query against one shared candidate pool.

    Args:
        boundary_dim: Boundary state width.
        query_dim: Query state width.
        pair_dim: Candidate feature width.
        dropout: Dropout probability.
        candidate_attention_layers: Overlap-biased candidate blocks.
        candidate_attention_heads: Heads in candidate and query attention.
        query_attention_layers: Evidence-conditioned query blocks.
        enable_span_content: Add pooled token content.
        content_dim: Content projection width.
        content_soft_max_pool: Also pool a smooth maximum.
        text_hidden_size: Token hidden size read by the content pooler.
    """

    def __init__(
        self,
        boundary_dim: int,
        query_dim: int,
        pair_dim: int,
        *,
        dropout: float,
        candidate_attention_layers: int,
        candidate_attention_heads: int,
        query_attention_layers: int,
        enable_span_content: bool,
        content_dim: int,
        content_soft_max_pool: bool,
        text_hidden_size: int,
    ) -> None:
        super().__init__()
        self.start_projection = nn.Linear(boundary_dim, pair_dim)
        self.end_projection = nn.Linear(boundary_dim, pair_dim)
        self.length_projection = nn.Linear(3, pair_dim)
        self.prior_projection = nn.Linear(1, pair_dim)
        self.content_pooler = (
            SpanContentPooler(text_hidden_size, content_dim, dropout, content_soft_max_pool)
            if enable_span_content
            else None
        )
        content_output = self.content_pooler.output_dim if self.content_pooler else 0
        self.content_projection = nn.Linear(content_output, pair_dim) if content_output else None
        self.candidate_norm = nn.LayerNorm(pair_dim)
        self.candidate_layers = nn.ModuleList(
            [
                OverlapBiasedCandidateAttention(pair_dim, candidate_attention_heads, dropout)
                for _ in range(candidate_attention_layers)
            ]
        )
        self.query_projection = nn.Linear(query_dim, pair_dim)
        self.query_layers = nn.ModuleList(
            [
                EvidenceConditionedQueryAttention(
                    pair_dim,
                    pair_dim,
                    candidate_attention_heads,
                    dropout,
                )
                for _ in range(query_attention_layers)
            ]
        )
        self.film = nn.Linear(pair_dim, 2 * pair_dim)
        self.film_output = nn.Sequential(  # trf-ignore: TRF036
            nn.Linear(pair_dim, 64),  # trf-ignore: TRF024
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # trf-ignore: TRF024
        )

    def forward(
        self,
        boundary_states: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        pooled: PooledCandidates,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        inside_prefix: torch.Tensor | None,
        text_lengths: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        inside_prefix_mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query scores in candidate-major order plus candidate states.

        Args:
            boundary_states: Boundary states `[B, N, D]`.
            query_states: Query states `[B, Q, H]`.
            query_mask: Valid queries `[B, Q]`.
            pooled: Document pool.
            start_logits: Start marginals `[B, Q, N]`.
            end_logits: End marginals `[B, Q, N]`.
            inside_prefix: Centered inside prefix `[B, Q, L + 1]`, or None.
            text_lengths: Token counts `[B]`.
            text_states: Token states `[B, L, H]`.
            text_mask: Valid tokens `[B, L]`.
            inside_prefix_mean: Mean restored onto inside intervals.

        Returns:
            Scores `[B, C, Q]` and contextual candidate states `[B, C, P]`.
        """
        starts, ends = pooled.indices[..., 0], pooled.indices[..., 1]
        start_rep = gather_rows(self.start_projection(boundary_states), starts)
        end_rep = gather_rows(self.end_projection(boundary_states), ends)
        feature_dtype = start_rep.dtype
        length = (ends - starts).clamp_min(1).to(feature_dtype)
        text_length = text_lengths[:, None].to(feature_dtype).clamp_min(1)
        length_features = torch.stack((torch.log1p(length), length / text_length, torch.rsqrt(length)), -1)
        prior = pooled.compat_logits if pooled.compat_logits is not None else pooled.proposal_logits
        if prior is None:
            prior = start_rep.new_zeros(starts.shape)
        candidate = (
            start_rep
            + end_rep
            + self.length_projection(length_features)
            + self.prior_projection(prior.unsqueeze(-1).to(feature_dtype))
        )
        if self.content_pooler is not None:
            mean_prefix, lse_prefix = self.content_pooler.build_prefix(text_states, text_mask)
            content = self.content_pooler.pool(mean_prefix, lse_prefix, starts, ends, candidate.dtype)
            candidate = candidate + self.content_projection(content)
        candidate = self.candidate_norm(candidate)
        candidate = candidate * pooled.mask.unsqueeze(-1).to(candidate.dtype)
        for layer in self.candidate_layers:
            candidate = layer(candidate, pooled.indices, pooled.mask)

        query = self.query_projection(query_states)
        if self.query_layers:
            for layer in self.query_layers:
                preliminary = torch.einsum("bcd,bqd->bcq", candidate, query)
                evidence_weights = torch.softmax(
                    preliminary.masked_fill(~pooled.mask.unsqueeze(-1), MASK_LOGIT),
                    dim=1,
                )
                evidence = torch.einsum("bcq,bcd->bqd", evidence_weights, candidate)
                query = layer(query, evidence, query_mask)

        score = torch.einsum("bcd,bqd->bcq", candidate, query) / math.sqrt(candidate.shape[-1])
        gamma, beta = self.film(query).chunk(2, -1)
        conditioned = candidate.unsqueeze(2) * (1.0 + gamma.unsqueeze(1))
        conditioned = conditioned + beta.unsqueeze(1)
        score = score + self.film_output(conditioned).squeeze(-1)

        count = starts.shape[1]
        s_idx = starts.clamp(0, start_logits.shape[2] - 1).unsqueeze(1).expand(-1, query_states.shape[1], count)
        e_idx = ends.clamp(0, end_logits.shape[2] - 1).unsqueeze(1).expand_as(s_idx)
        score = score + start_logits.gather(2, s_idx).transpose(1, 2)
        score = score + end_logits.gather(2, e_idx).transpose(1, 2)
        if inside_prefix is not None:
            interval = inside_prefix.gather(2, e_idx.clamp(max=inside_prefix.shape[2] - 1)) - inside_prefix.gather(
                2, s_idx.clamp(max=inside_prefix.shape[2] - 1)
            )
            if inside_prefix_mean is not None:
                interval = interval + inside_prefix_mean * (e_idx - s_idx).to(interval.dtype)
            score = score + (interval / torch.sqrt((e_idx - s_idx).clamp_min(1).float())).transpose(1, 2).to(
                score.dtype
            )
        score = score.masked_fill(~pooled.mask.unsqueeze(-1), MASK_LOGIT)
        score = score.masked_fill(~query_mask.unsqueeze(1), MASK_LOGIT)
        return score, candidate


@dataclass(frozen=True)
class RelationTypeSpec:
    """One relation type and the entity queries allowed at each argument."""

    relation_type: str
    head_query_ids: tuple[int, ...]
    tail_query_ids: tuple[int, ...]
    allow_self: bool = False


@dataclass
class RelationPairBatch:
    """Flattened typed relation pairs.

    Index tensors have shape `[P]`. Ends are exclusive.

    Attributes:
        batch_index: Sample index of each pair.
        relation_index: Relation index of each pair.
        head_start: Inclusive head start.
        head_end: Exclusive head end.
        tail_start: Inclusive tail start.
        tail_end: Exclusive tail end.
        head_prob: Head mention probability.
        tail_prob: Tail mention probability.
        pair_mask: True for pairs that should be scored.
        head_keys: Optional decode metadata `(type, start, end)`.
        tail_keys: Optional decode metadata `(type, start, end)`.
        relation_types: Optional relation name per pair.
    """

    batch_index: torch.Tensor
    relation_index: torch.Tensor
    head_start: torch.Tensor
    head_end: torch.Tensor
    tail_start: torch.Tensor
    tail_end: torch.Tensor
    head_prob: torch.Tensor
    tail_prob: torch.Tensor
    pair_mask: torch.Tensor | None = None
    head_keys: list[tuple[str, int, int]] = field(default_factory=list)
    tail_keys: list[tuple[str, int, int]] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.batch_index.shape[0])


def _query_type(layout, query_id: int) -> str:
    """Return a query role name, or the id when the layout has no such query."""
    try:
        return layout.query(query_id).role_name
    except KeyError:
        return str(query_id)


def safe_relation_indices(relation_indices: torch.Tensor, relation_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp relation indices and report which originals were in range.

    Args:
        relation_indices: Proposed relation ids.
        relation_count: Number of relation queries.

    Returns:
        Gather-safe indices and a boolean validity mask.
    """
    if relation_count <= 0:
        return (
            torch.zeros_like(relation_indices),
            torch.zeros_like(relation_indices, dtype=torch.bool),
        )
    valid = (relation_indices >= 0) & (relation_indices < relation_count)
    return relation_indices.clamp(min=0, max=relation_count - 1), valid


class TypedRelationPairGenerator:
    """Propose a capped cross product of typed head and tail mentions."""

    def __init__(
        self,
        heads_per_relation: int = 32,
        tails_per_relation: int = 32,
        pair_cap: int = 128,
        argument_threshold: float = 0.0,
    ) -> None:
        """Store the retention caps used while proposing pairs."""
        self.heads_per_relation = heads_per_relation
        self.tails_per_relation = tails_per_relation
        self.pair_cap = pair_cap
        self.argument_threshold = argument_threshold

    def generate(
        self,
        candidates,
        query_layouts,
        relation_schema,
        *,
        compact: bool = True,
    ) -> RelationPairBatch:
        """Propose pairs for one schema shared by every sample.

        Args:
            candidates: Per-query mention candidates.
            query_layouts: One query layout per sample, used for decode names.
            relation_schema: Relation types applied to every sample.
            compact: Drop padded pairs and attach Python metadata.

        Returns:
            Flattened relation pairs.
        """
        schemas = [relation_schema for _ in range(candidates.indices.shape[0])]
        return self.generate_batched(candidates, query_layouts, schemas, compact=compact)

    def generate_batched(
        self,
        candidates,
        query_layouts,
        relation_schemas,
        *,
        compact: bool = False,
        routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> RelationPairBatch:
        """Propose `[B, R, pair_cap]` typed pairs."""
        settings = self
        device = candidates.indices.device
        batch_size, queries, cand_count = candidates.valid_mask.shape
        rel_count = (
            routing[0].shape[1] if routing is not None else max((len(item) for item in relation_schemas), default=0)
        )
        if rel_count == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return RelationPairBatch(
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                candidates.pair_logits.new_zeros(0),
                candidates.pair_logits.new_zeros(0),
                pair_mask=torch.zeros(0, dtype=torch.bool, device=device),
            )

        if routing is not None:
            head_member, tail_member, relation_valid, allow_self = routing
            head_member = head_member.to(device=device)
            tail_member = tail_member.to(device=device)
            relation_valid = relation_valid.to(device=device)
            allow_self = allow_self.to(device=device)
        else:
            head_member = torch.zeros(batch_size, rel_count, queries, dtype=torch.bool, device=device)
            tail_member = torch.zeros_like(head_member)
            relation_valid = torch.zeros(batch_size, rel_count, dtype=torch.bool, device=device)
            allow_self = torch.zeros_like(relation_valid)
            for batch_index, schemas in enumerate(relation_schemas):
                for relation_index, spec in enumerate(schemas):
                    relation_valid[batch_index, relation_index] = True
                    allow_self[batch_index, relation_index] = spec.allow_self
                    valid_h = [q for q in spec.head_query_ids if 0 <= q < queries]
                    valid_t = [q for q in spec.tail_query_ids if 0 <= q < queries]
                    if valid_h:
                        head_member[batch_index, relation_index, valid_h] = True
                    if valid_t:
                        tail_member[batch_index, relation_index, valid_t] = True

        probs = torch.sigmoid(candidates.pair_logits)
        base_valid = candidates.valid_mask & candidates.query_mask.unsqueeze(-1)
        flat_prob = probs.reshape(batch_size, 1, queries * cand_count).expand(-1, rel_count, -1)
        flat_valid = base_valid.reshape(batch_size, 1, -1).expand(-1, rel_count, -1)
        head_valid = flat_valid & head_member.unsqueeze(-1).expand(-1, -1, -1, cand_count).reshape(
            batch_size, rel_count, -1
        )
        tail_valid = flat_valid & tail_member.unsqueeze(-1).expand(-1, -1, -1, cand_count).reshape(
            batch_size, rel_count, -1
        )
        threshold = flat_prob >= settings.argument_threshold
        head_valid = head_valid & threshold
        tail_valid = tail_valid & threshold
        floor = torch.finfo(flat_prob.dtype).min
        flat_spans = candidates.indices.reshape(batch_size, queries * cand_count, 2)

        def select(valid: torch.Tensor, requested: int):
            take = min(requested, queries * cand_count)
            secondary = (
                torch.arange(queries * cand_count, device=device).view(1, 1, -1).expand(batch_size, rel_count, -1)
            )
            end_key = flat_spans[..., 1].unsqueeze(1).expand(-1, rel_count, -1)
            end_order = torch.argsort(end_key.gather(-1, secondary), dim=-1, stable=True)
            secondary = secondary.gather(-1, end_order)
            start_key = flat_spans[..., 0].unsqueeze(1).expand(-1, rel_count, -1)
            start_order = torch.argsort(start_key.gather(-1, secondary), dim=-1, stable=True)
            secondary = secondary.gather(-1, start_order)
            ordered_score = flat_prob.gather(-1, secondary)
            ordered_valid = valid.gather(-1, secondary)
            rank_in_secondary = torch.argsort(
                ordered_score.masked_fill(~ordered_valid, floor),
                dim=-1,
                descending=True,
                stable=True,
            )[..., :take]
            ranked = secondary.gather(-1, rank_in_secondary)
            selected_valid = valid.gather(-1, ranked)
            selected_prob = flat_prob.gather(-1, ranked)
            if take < requested:
                pad = requested - take
                ranked = F.pad(ranked, (0, pad))
                selected_valid = F.pad(selected_valid, (0, pad), value=False)
                selected_prob = F.pad(selected_prob, (0, pad))
            qslot = torch.div(ranked, cand_count, rounding_mode="floor")
            cslot = ranked - qslot * cand_count
            qslot = qslot.clamp(0, queries - 1)
            cslot = cslot.clamp(0, cand_count - 1)
            batch = torch.arange(batch_size, device=device)[:, None, None]
            spans = candidates.indices[batch, qslot, cslot]
            return selected_prob, qslot, spans, selected_valid

        hp, hq, hspan, hvalid = select(head_valid, settings.heads_per_relation)
        tp, tq, tspan, tvalid = select(tail_valid, settings.tails_per_relation)
        pair_score = hp.unsqueeze(-1) * tp.unsqueeze(-2)
        pair_valid = hvalid.unsqueeze(-1) & tvalid.unsqueeze(-2)
        same_span = (hspan.unsqueeze(-2) == tspan.unsqueeze(-3)).all(-1)
        pair_valid = pair_valid & (allow_self[..., None, None] | ~same_span)
        pair_valid = pair_valid & relation_valid[..., None, None]
        flat_pair_score = pair_score.flatten(2)
        flat_pair_valid = pair_valid.flatten(2)
        take = min(settings.pair_cap, flat_pair_score.shape[-1])
        keep = torch.argsort(
            flat_pair_score.masked_fill(~flat_pair_valid, floor),
            dim=-1,
            descending=True,
            stable=True,
        )[..., :take]
        kept_valid = flat_pair_valid.gather(-1, keep)
        if take < settings.pair_cap:
            keep = F.pad(keep, (0, settings.pair_cap - take))
            kept_valid = F.pad(kept_valid, (0, settings.pair_cap - take), value=False)
        hi = torch.div(keep, settings.tails_per_relation, rounding_mode="floor")
        ti = keep - hi * settings.tails_per_relation
        hi = hi.clamp(0, settings.heads_per_relation - 1)
        ti = ti.clamp(0, settings.tails_per_relation - 1)

        def gather_selected(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return values.gather(
                2,
                index.clamp(0, values.shape[2] - 1).unsqueeze(-1).expand(*index.shape, values.shape[-1]),
            )

        hs = gather_selected(hspan, hi)
        ts = gather_selected(tspan, ti)
        hp_out = hp.gather(2, hi.clamp(0, hp.shape[2] - 1))
        tp_out = tp.gather(2, ti.clamp(0, tp.shape[2] - 1))
        hq_out = hq.gather(2, hi.clamp(0, hq.shape[2] - 1))
        tq_out = tq.gather(2, ti.clamp(0, tq.shape[2] - 1))
        bi = torch.arange(batch_size, device=device)[:, None, None].expand_as(keep)
        ri = torch.arange(rel_count, device=device)[None, :, None].expand_as(keep)
        flat_mask = kept_valid.reshape(-1)
        tensors = [
            bi.reshape(-1),
            ri.reshape(-1),
            hs[..., 0].reshape(-1),
            hs[..., 1].reshape(-1),
            ts[..., 0].reshape(-1),
            ts[..., 1].reshape(-1),
            hp_out.reshape(-1),
            tp_out.reshape(-1),
            hq_out.reshape(-1),
            tq_out.reshape(-1),
        ]
        if compact:
            tensors = [value[flat_mask] for value in tensors]
            flat_mask = torch.ones_like(tensors[0], dtype=torch.bool)
        out = RelationPairBatch(
            batch_index=tensors[0],
            relation_index=tensors[1],
            head_start=tensors[2],
            head_end=tensors[3],
            tail_start=tensors[4],
            tail_end=tensors[5],
            head_prob=tensors[6],
            tail_prob=tensors[7],
            pair_mask=flat_mask,
        )
        if compact:
            for index in range(len(out)):
                batch_index = int(out.batch_index[index])
                relation_index = int(out.relation_index[index])
                spec = relation_schemas[batch_index][relation_index]
                layout = query_layouts[batch_index] if batch_index < len(query_layouts) else None
                out.relation_types.append(spec.relation_type)
                hquery = int(tensors[8][index])
                tquery = int(tensors[9][index])
                out.head_keys.append(
                    (
                        _query_type(layout, hquery) if layout is not None else str(hquery),
                        int(out.head_start[index]),
                        int(out.head_end[index]),
                    )
                )
                out.tail_keys.append(
                    (
                        _query_type(layout, tquery) if layout is not None else str(tquery),
                        int(out.tail_start[index]),
                        int(out.tail_end[index]),
                    )
                )
        return out


class SparseRelationScorer(nn.Module):
    """Score proposed relation pairs from the four endpoint states.

    Args:
        hidden_size: Boundary-state width.
        dropout: MLP dropout probability.
        relation_query_dim: Relation-query width. Defaults to `hidden_size`.
        use_biaffine_content: Add pooled head/tail content features.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        relation_query_dim: int | None = None,
        use_biaffine_content: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.relation_query_dim = relation_query_dim or hidden_size
        self.use_biaffine_content = use_biaffine_content
        in_dim = 4 * hidden_size + self.relation_query_dim + 2
        self.mlp = nn.Sequential(  # trf-ignore: TRF036
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        if use_biaffine_content:
            self.head_content_projection = nn.Linear(hidden_size, hidden_size)
            self.tail_content_projection = nn.Linear(hidden_size, hidden_size)
            self.relation_content_gate = nn.Linear(self.relation_query_dim, hidden_size)
            self.content_linear = nn.Linear(2 * hidden_size + self.relation_query_dim, 1)

    def forward(
        self,
        boundary_states: torch.Tensor,
        relation_query_states: torch.Tensor,
        entity_candidates,
        relation_pairs: RelationPairBatch,
    ) -> torch.Tensor:
        """Score each proposed pair.

        Args:
            boundary_states: Boundary or token states `[B, L, H]`.
            relation_query_states: Relation queries `[B, R, H]`.
            entity_candidates: Candidate carrier kept for call compatibility.
            relation_pairs: Pairs from `TypedRelationPairGenerator`.

        Returns:
            Pair logits `[P]`, zero where the pair is invalid.
        """
        del entity_candidates  # spans are read from relation_pairs
        if len(relation_pairs) == 0:
            return boundary_states.new_zeros(0)

        batch_count = min(boundary_states.shape[0], relation_query_states.shape[0])
        if batch_count <= 0 or relation_query_states.shape[1] <= 0:
            return boundary_states.new_zeros(len(relation_pairs))
        batch_valid = (relation_pairs.batch_index >= 0) & (relation_pairs.batch_index < batch_count)
        batch = relation_pairs.batch_index.clamp(0, batch_count - 1)
        relation_index, relation_valid = safe_relation_indices(
            relation_pairs.relation_index, relation_query_states.shape[1]
        )
        pair_valid = batch_valid & relation_valid
        if relation_pairs.pair_mask is not None:
            pair_valid = pair_valid & relation_pairs.pair_mask
        length = boundary_states.shape[1]

        def gather(pos: torch.Tensor) -> torch.Tensor:
            pos = pos.clamp(0, max(length - 1, 0))
            return boundary_states[batch, pos]

        h_start = gather(relation_pairs.head_start)
        h_end = gather(relation_pairs.head_end - 1)
        t_start = gather(relation_pairs.tail_start)
        t_end = gather(relation_pairs.tail_end - 1)
        rel = relation_query_states[batch, relation_index]
        delta = (relation_pairs.tail_start - relation_pairs.head_start).to(boundary_states.dtype)
        order = torch.sign(delta).unsqueeze(-1)
        dist = (delta.abs() / float(max(length, 1))).unsqueeze(-1)
        feats = torch.cat([h_start, h_end, t_start, t_end, rel, order, dist], dim=-1)
        score = self.mlp(feats).squeeze(-1)
        if self.use_biaffine_content:
            prefix = torch.cat(
                (
                    boundary_states.new_zeros(boundary_states.shape[0], 1, self.hidden_size),
                    boundary_states.float().cumsum(1).to(boundary_states.dtype),
                ),
                dim=1,
            )

            def pool(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
                span_sum = prefix[batch, end.clamp(0, length)] - prefix[batch, start.clamp(0, length)]
                width = (end - start).clamp_min(1).unsqueeze(-1).to(span_sum.dtype)
                return span_sum / width

            head_content = self.head_content_projection(pool(relation_pairs.head_start, relation_pairs.head_end))
            tail_content = self.tail_content_projection(pool(relation_pairs.tail_start, relation_pairs.tail_end))
            gate = torch.sigmoid(self.relation_content_gate(rel))
            biaffine = (head_content * gate * tail_content).sum(-1) / (self.hidden_size**0.5)
            linear = self.content_linear(torch.cat((head_content, tail_content, rel), dim=-1)).squeeze(-1)
            score = score + biaffine + linear
        return score.masked_fill(~pair_valid, 0.0)


@dataclass
class RecordGroupOutput:
    """One record group's instance and field-assignment scores.

    Attributes:
        spec: Record schema that produced this group.
        object_logits: Instance object scores `[Ni]`.
        assign_logits: Per-field logits `[Ni, 1 + Cf]`, column 0 is absent.
        field_query_ids: Boundary query id of each field.
        field_specs: Field schema objects, in assignment order.
        field_spans: Half-open candidate spans per field.
        field_cand_mask: Real candidates per field.
        field_cand_logits: Pair logits of those candidates.
        instance_seed: `(field_index, candidate_index)` seed, if any.
        instance_spans: Span of each seeded instance.
    """

    spec: object
    object_logits: torch.Tensor
    assign_logits: list[torch.Tensor]
    field_query_ids: list[int]
    field_specs: list
    field_spans: list[torch.Tensor]
    field_cand_mask: list[torch.Tensor]
    field_cand_logits: list[torch.Tensor]
    instance_seed: list[tuple[int, int] | None]
    instance_spans: list[tuple[int, int] | None]

    @property
    def num_instances(self) -> int:
        """Number of instance rows in this group."""
        return int(self.object_logits.shape[0])


class RecordHead(nn.Module):
    """Form record instances and score field-to-instance assignment."""

    def __init__(self, hidden_size: int, record_dim: int, instance_queries: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.record_dim = record_dim
        self.instance_queries = instance_queries
        self.inst_proj = nn.Linear(hidden_size, record_dim)
        self.field_proj = nn.Linear(hidden_size, record_dim)
        self.cand_proj = nn.Linear(hidden_size, record_dim)
        self.null_embed = nn.Parameter(torch.randn(record_dim) * 0.02)
        self.object_head = nn.Linear(hidden_size, 1)
        self.latent_seed_head = nn.Linear(hidden_size, 1)
        self.instance_embed = nn.Parameter(torch.randn(instance_queries, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, record_dim)
        self.k_proj = nn.Linear(hidden_size, record_dim)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def _assign_logits(
        self,
        inst_states: torch.Tensor,
        field_query_states: torch.Tensor,
        field_cand_states: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Score absent-aware assignment of each field candidate.

        Args:
            inst_states: Instance states `[Ni, H]`.
            field_query_states: Field queries `[F, H]`.
            field_cand_states: Candidate states per field, each `[Cf, H]`.

        Returns:
            Logits per field, each `[Ni, 1 + Cf]`.
        """
        inst_q = self.inst_proj(inst_states)
        field_q = self.field_proj(field_query_states)
        out: list[torch.Tensor] = []
        for f_idx, cand in enumerate(field_cand_states):
            query = inst_q + field_q[f_idx].unsqueeze(0)
            null_col = query @ self.null_embed
            if cand.shape[0] == 0:
                out.append(null_col.unsqueeze(-1))
                continue
            cand_p = self.cand_proj(cand)
            cand_scores = query @ cand_p.t()
            out.append(torch.cat([null_col.unsqueeze(-1), cand_scores], dim=-1))
        return out

    def _anchorless_states(self, field_cand_states: list[torch.Tensor]) -> torch.Tensor:
        """Cross-attend learned instance queries over field candidates.

        Args:
            field_cand_states: Non-empty candidate states are concatenated.

        Returns:
            Conditioned instance states `[I, H]`.
        """
        inst = self.instance_embed
        ctx = [cand for cand in field_cand_states if cand.shape[0] > 0]
        if not ctx:
            return inst
        ctx = torch.cat(ctx, dim=0)
        q = self.q_proj(inst)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        attn = (q @ k.t()) / math.sqrt(self.record_dim)
        weights = torch.softmax(attn, dim=-1)
        return inst + weights @ v

    def forward_group(
        self,
        spec,
        query_states: torch.Tensor,
        candidates,
        sample_index: int,
    ) -> RecordGroupOutput:
        """Score one record schema for one sample.

        Args:
            spec: Schema with `mode`, `fields`, and optional `anchor_query_id`.
            query_states: This sample's query states `[Q, H]`.
            candidates: Batch candidates containing states, spans, and logits.
            sample_index: Batch row to read.

        Returns:
            Instance object logits and per-field assignment logits.
        """
        device = query_states.device
        field_specs = list(spec.fields)
        field_query_ids = [fspec.query_id for fspec in field_specs]
        query_count = min(
            query_states.shape[0],
            candidates.valid_mask.shape[1],
            candidates.pair_logits.shape[1],
            candidates.candidate_states.shape[1],
        )
        if query_count <= 0:
            raise ValueError("record routing requires at least one boundary query")
        valid_query_ids = [0 <= query_id < query_count for query_id in field_query_ids]
        safe_field_query_ids = [min(max(query_id, 0), query_count - 1) for query_id in field_query_ids]

        field_cand_states: list[torch.Tensor] = []
        field_spans: list[torch.Tensor] = []
        field_cand_mask: list[torch.Tensor] = []
        field_cand_logits: list[torch.Tensor] = []
        cand_states_all = candidates.candidate_states
        for qid, query_valid in zip(safe_field_query_ids, valid_query_ids):
            mask = candidates.valid_mask[sample_index, qid] & query_valid
            keep = torch.nonzero(mask, as_tuple=False).flatten()
            spans = candidates.indices[sample_index, qid][keep]
            logits = candidates.pair_logits[sample_index, qid][keep]
            states = cand_states_all[sample_index, qid][keep]
            field_cand_states.append(states)
            field_spans.append(spans.to(torch.long))
            field_cand_mask.append(torch.ones(keep.shape[0], dtype=torch.bool, device=device))
            field_cand_logits.append(logits)

        fq = query_states[safe_field_query_ids]
        instance_seed: list[tuple[int, int] | None] = []
        instance_spans: list[tuple[int, int] | None] = []

        if spec.mode == "natural":
            anchor_field_idx = field_query_ids.index(spec.anchor_query_id)
            anchor_states = field_cand_states[anchor_field_idx]
            anchor_spans = field_spans[anchor_field_idx]
            anchor_logits = field_cand_logits[anchor_field_idx]
            ni = anchor_states.shape[0]
            inst_states = anchor_states
            object_logits = anchor_logits
            for cand_idx in range(ni):
                instance_seed.append((anchor_field_idx, cand_idx))
                instance_spans.append((int(anchor_spans[cand_idx, 0]), int(anchor_spans[cand_idx, 1])))
        elif spec.mode == "latent":
            seed_states: list[torch.Tensor] = []
            seed_scores: list[torch.Tensor] = []
            for f_idx, states in enumerate(field_cand_states):
                if states.shape[0] == 0:
                    continue
                scores = self.latent_seed_head(states).squeeze(-1)
                for cand_idx in range(states.shape[0]):
                    seed_states.append(states[cand_idx])
                    seed_scores.append(scores[cand_idx])
                    instance_seed.append((f_idx, cand_idx))
                    span = field_spans[f_idx][cand_idx]
                    instance_spans.append((int(span[0]), int(span[1])))
            if seed_states:
                inst_states = torch.stack(seed_states, dim=0)
                object_logits = torch.stack(seed_scores, dim=0)
            else:
                inst_states = query_states.new_zeros((0, self.hidden_size))
                object_logits = query_states.new_zeros((0,))
        else:
            inst_states = self._anchorless_states(field_cand_states)
            object_logits = self.object_head(inst_states).squeeze(-1)
            for _ in range(inst_states.shape[0]):
                instance_seed.append(None)
                instance_spans.append(None)

        assign_logits = self._assign_logits(inst_states, fq, field_cand_states)
        return RecordGroupOutput(
            spec=spec,
            object_logits=object_logits,
            assign_logits=assign_logits,
            field_query_ids=field_query_ids,
            field_specs=field_specs,
            field_spans=field_spans,
            field_cand_mask=field_cand_mask,
            field_cand_logits=field_cand_logits,
            instance_seed=instance_seed,
            instance_spans=instance_spans,
        )


@dataclass
class BoundaryHeadOutput:  # trf-ignore: TRF031
    """Marginal logits and sparse candidates from `BoundaryHead`."""

    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None
    inside_logits: torch.Tensor | None = None
    candidates: CandidateTensorBatch | None = None
    null_logits: torch.Tensor | None = None
    count_log_rates: torch.Tensor | None = None
    batch_size: int = 0


class BoundaryHead(nn.Module):
    """Encode boundaries, propose spans, and rerank them."""

    def __init__(
        self,
        hidden_size: int,
        settings,
        query_dim: int | None = None,
        build_candidate_states: bool = False,
    ):
        """Build the encoder, proposer, and scorer from `boundary_config`."""
        super().__init__()
        self.hidden_size = hidden_size
        self.settings = settings
        self.query_dim = query_dim if query_dim is not None else hidden_size
        self.collect_diagnostics = False
        self._gold_injection_prob = 1.0
        self._consistency_scale = 1.0
        self._soft_iou_scale = 1.0

        dim = settings.boundary_dim
        self.candidate_encoder = nn.Linear(2 * dim, hidden_size) if build_candidate_states else None
        self.boundary_encoder = BoundaryEncoder(
            hidden_size,
            dim,
            settings.dropout,
            settings.boundary_refinement_layers,
            settings.boundary_ffn_multiplier,
            settings.boundary_attention_layers,
            settings.boundary_attention_heads,
            settings.boundary_attention_window,
        )
        self.boundary_query_head = BoundaryQueryHead(hidden_size, dim, self.query_dim, settings.dropout)
        self.boundary_proposer = SparseBoundaryProposer(dim, self.query_dim, settings)
        self.pair_scorer = SparseBoundaryPairScorer(
            dim,
            self.query_dim,
            settings.pair_dim,
            use_inside_evidence=settings.use_inside_evidence,
            dropout=settings.dropout,
            enable_span_content=settings.enable_span_content,
            content_dim=settings.content_dim,
            content_soft_max_pool=settings.content_soft_max_pool,
            enable_rotary_endpoints=settings.enable_rotary_endpoints,
            rotary_base=settings.rotary_base,
            query_conditioned_inside_weight=settings.query_conditioned_inside_weight,
            endpoint_difference_features=settings.endpoint_difference_features,
            reranker_endpoint_compat=settings.reranker_endpoint_compat,
            multihead_pair_compat_heads=settings.multihead_pair_compat_heads,
            content_hidden_size=hidden_size,
        )
        self.use_inside_evidence = settings.use_inside_evidence
        # Optional heads are zeroed without moving the caller RNG stream.
        auxiliary_rng_state = torch.random.get_rng_state()
        self.null_projection = nn.Linear(self.query_dim, 1) if settings.enable_abstention else None
        if self.null_projection is not None:
            nn.init.zeros_(self.null_projection.weight)  # trf-ignore: TRF049
            nn.init.zeros_(self.null_projection.bias)  # trf-ignore: TRF049
        self.count_head = nn.Linear(self.query_dim, 1) if settings.enable_count_head else None
        if self.count_head is not None:
            nn.init.zeros_(self.count_head.weight)  # trf-ignore: TRF049
            nn.init.zeros_(self.count_head.bias)  # trf-ignore: TRF049
        torch.random.set_rng_state(auxiliary_rng_state)
        shared_rng_state = torch.random.get_rng_state()
        self.shared_pool_builder = DocumentCandidatePool(
            dim,
            pool_boundary_top_k=settings.pool_boundary_top_k,
            pool_size=settings.pool_size,
            min_pool_per_query=settings.min_pool_per_query,
        )
        self.shared_pool_scorer = SharedPoolScorer(
            dim,
            self.query_dim,
            settings.pair_dim,
            dropout=settings.dropout,
            candidate_attention_layers=settings.candidate_attention_layers,
            candidate_attention_heads=settings.candidate_attention_heads,
            query_attention_layers=settings.query_attention_layers,
            enable_span_content=settings.enable_span_content,
            content_dim=settings.content_dim,
            content_soft_max_pool=settings.content_soft_max_pool,
            text_hidden_size=hidden_size,
        )
        torch.random.set_rng_state(shared_rng_state)

    def set_gold_injection_prob(self, value: float) -> None:  # trf-ignore: TRF033
        """Set the training gold-injection probability.

        Args:
            value: Probability in `[0, 1]`.
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"gold_injection_prob must be in [0, 1], got {value}")
        self._gold_injection_prob = float(value)

    def set_consistency_scale(self, value: float) -> None:  # trf-ignore: TRF033
        """Set the marginal-consistency loss multiplier.

        Args:
            value: Non-negative scale. The trainer anneals this during warmup.
        """
        if value < 0:
            raise ValueError(f"consistency_scale must be >= 0, got {value}")
        self._consistency_scale = float(value)

    def set_soft_iou_scale(self, value: float) -> None:  # trf-ignore: TRF033
        """Set the soft-IoU loss multiplier.

        Args:
            value: Non-negative scale. The trainer anneals this over training.
        """
        if value < 0:
            raise ValueError(f"soft_iou_scale must be >= 0, got {value}")
        self._soft_iou_scale = float(value)

    def score_explicit_spans(
        self,
        token_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score caller-provided half-open spans for every query.

        Args:
            token_states: Token states `[B, L, H]`.
            text_mask: Valid tokens `[B, L]`.
            query_states: Query states `[B, Q, H]`.
            query_mask: Valid queries `[B, Q]`.
            indices: Spans `[B, Q, C, 2]`.
            valid_mask: Optional extra mask `[B, Q, C]`.

        Returns:
            Pair logits `[B, Q, C]`.
        """
        if indices.dim() != 4 or indices.shape[-1] != 2:
            raise ValueError(f"indices must be [B, Q, C, 2], got {tuple(indices.shape)}")
        batch, queries, _, _ = indices.shape
        if batch != token_states.shape[0] or batch != query_states.shape[0] or queries != query_states.shape[1]:
            raise ValueError("explicit span batch/query dimensions do not match states")

        text_lengths = text_mask.sum(dim=1).long()
        starts = indices[..., 0]
        ends = indices[..., 1]
        legal = (starts >= 0) & (ends > starts) & (ends <= text_lengths.view(batch, 1, 1)) & query_mask.unsqueeze(-1)
        if valid_mask is not None:
            if valid_mask.shape != indices.shape[:-1]:
                raise ValueError(
                    "valid_mask must match indices [B, Q, C], got "
                    f"{tuple(valid_mask.shape)} and {tuple(indices.shape)}"
                )
            legal = legal & valid_mask

        encoding = self.boundary_encoder(token_states, text_mask)
        marginals = self.boundary_query_head(
            encoding.states,
            encoding.mask,
            token_states,
            text_mask,
            query_states,
            query_mask,
        )
        compatibility = self.boundary_proposer.score_explicit_pairs(encoding.states, query_states, indices, legal)
        proposals = BoundaryProposals(
            indices=indices,
            logits=None,
            valid_mask=legal,
            compat_logits=compatibility,
        )
        inside_prefix = marginals.inside_prefix if self.use_inside_evidence else None
        return self.pair_scorer(
            encoding.states,
            query_states,
            proposals,
            marginals.start_logits,
            marginals.end_logits,
            inside_prefix,
            text_lengths,
            token_states,
            text_mask,
            inside_prefix_mean=marginals.inside_prefix_mean,
        )

    def forward(
        self,
        token_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        targets=None,
        *,
        return_candidates: bool = True,
        gold_injection_prob: float | None = None,
        collect_diagnostics: bool | None = None,
    ) -> BoundaryHeadOutput:
        """Propose and rerank boundary spans."""
        batch = token_states.shape[0]
        text_lengths = text_mask.sum(dim=1).long()
        encoding = self.boundary_encoder(token_states, text_mask)
        marginals = self.boundary_query_head(
            encoding.states,
            encoding.mask,
            token_states,
            text_mask,
            query_states,
            query_mask,
        )
        gold_pairs = gold_mask = None
        if targets is not None:
            gold_pairs = targets.mention_pairs
            gold_mask = targets.mention_mask
        diagnostics = self.collect_diagnostics if collect_diagnostics is None else collect_diagnostics
        if targets is not None:
            diagnostics = True
        injection = self._gold_injection_prob if gold_injection_prob is None else gold_injection_prob
        inside_prefix = marginals.inside_prefix if self.use_inside_evidence else None
        pooled = None
        pooled_candidate_states = None
        pooled_logits = None
        if self.settings.candidate_pool == "shared":
            pooled = self.shared_pool_builder(
                encoding.states,
                encoding.mask,
                query_mask,
                marginals.start_logits,
                marginals.end_logits,
                gold_pairs=gold_pairs,
                gold_mask=gold_mask,
                gold_injection_prob=injection if self.training else 0.0,
                return_stats=diagnostics,
            )
            pooled_logits, _ = self.shared_pool_scorer(
                encoding.states,
                query_states,
                query_mask,
                pooled,
                marginals.start_logits,
                marginals.end_logits,
                inside_prefix,
                text_lengths,
                token_states,
                text_mask,
                inside_prefix_mean=marginals.inside_prefix_mean,
            )
            queries = query_states.shape[1]
            count = pooled.indices.shape[1]
            proposals = BoundaryProposals(
                indices=pooled.indices.unsqueeze(1).expand(batch, queries, count, 2),
                logits=(
                    pooled.proposal_logits.unsqueeze(1).expand(batch, queries, count)
                    if pooled.proposal_logits is not None
                    else None
                ),
                valid_mask=pooled.mask.unsqueeze(1).expand(batch, queries, count),
                gold_mask=(pooled.gold_mask.transpose(1, 2) if pooled.gold_mask is not None else None),
                stats=pooled.stats,
                compat_logits=(
                    pooled.compat_logits.unsqueeze(1).expand(batch, queries, count)
                    if pooled.compat_logits is not None
                    else None
                ),
            )
            pair_logits = pooled_logits.transpose(1, 2)
            if self.candidate_encoder is not None:
                start_states = gather_rows(encoding.states, pooled.indices[..., 0])
                end_states = gather_rows(encoding.states, pooled.indices[..., 1])
                pooled_candidate_states = self.candidate_encoder(
                    torch.cat((start_states, end_states), -1)
                ).masked_fill(~pooled.mask.unsqueeze(-1), 0.0)
        else:
            scorer_start_states, scorer_end_states = self.pair_scorer.project_endpoints(encoding.states)
            proposals = self.boundary_proposer(
                encoding.states,
                encoding.mask,
                query_states,
                query_mask,
                marginals.start_logits,
                marginals.end_logits,
                gold_pairs=gold_pairs,
                gold_mask=gold_mask,
                return_stats=diagnostics,
                return_proposal_logits=self.training,
                gold_injection_prob=injection,
                scorer_start_states=scorer_start_states,
                scorer_end_states=scorer_end_states,
            )
            pair_logits = self.pair_scorer(
                encoding.states,
                query_states,
                proposals,
                marginals.start_logits,
                marginals.end_logits,
                inside_prefix,
                text_lengths,
                token_states,
                text_mask,
                inside_prefix_mean=marginals.inside_prefix_mean,
            )
        null_logits = self.null_projection(query_states).squeeze(-1) if self.null_projection is not None else None
        count_log_rates = self.count_head(query_states).squeeze(-1) if self.count_head is not None else None

        candidates = None
        if return_candidates:
            candidate_states = pooled_candidate_states
            if self.candidate_encoder is not None and pooled is None:
                gathered_start = gather_boundary_states(encoding.states, proposals.indices[..., 0])
                gathered_end = gather_boundary_states(encoding.states, proposals.indices[..., 1])
                candidate_states = self.candidate_encoder(torch.cat([gathered_start, gathered_end], dim=-1))
                candidate_states = candidate_states.masked_fill(~proposals.valid_mask.unsqueeze(-1), 0.0)
            candidates = (
                pooled.to_candidate_batch(pooled_logits, query_mask, candidate_states)
                if pooled is not None
                else CandidateTensorBatch(
                    indices=proposals.indices,
                    proposal_logits=proposals.logits,
                    pair_logits=pair_logits,
                    valid_mask=proposals.valid_mask,
                    query_mask=query_mask,
                    candidate_states=candidate_states,
                )
            )
        return BoundaryHeadOutput(
            start_logits=marginals.start_logits,
            end_logits=marginals.end_logits,
            inside_logits=marginals.inside_logits,
            candidates=candidates,
            null_logits=null_logits,
            count_log_rates=count_log_rates,
            batch_size=batch,
        )


@auto_docstring
class Gliner2PreTrainedModel(PreTrainedModel):
    config: Gliner2Config
    base_model_prefix = "gliner2"
    input_modalities = ("text",)
    _no_split_modules = []
    # The text encoder is DeBERTa, which rejects SDPA and runs eager.
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, CompileSafeGRU):
            module.reset_parameters()
            return
        super()._init_weights(module)


@auto_docstring
@dataclass
class Gliner2ModelOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Encoder states.
    text_states (`torch.FloatTensor` of shape `(batch_size, num_words, hidden_size)`, *optional*):
        One state per word.
    query_states (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
        Field-marker states.
    cls_states (`torch.FloatTensor` of shape `(batch_size, num_labels, hidden_size)`, *optional*):
        Classification-label states.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    text_states: torch.FloatTensor | None = None
    query_states: torch.FloatTensor | None = None
    cls_states: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class Gliner2SchemaExtractionOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Encoder states.
    classification_logits (`list`, *optional*):
        One logit vector per classification group, per batch row. Temperature is not applied.
    span_logits (`list`, *optional*):
        One `(count, fields, words, width)` logit tensor per span group, per batch row.
        The count axis is the gold count during training and the predicted count during inference.
    counts (`list`, *optional*):
        Instance counts aligned with `span_logits`.
    boundary (`BoundaryHeadOutput`, *optional*):
        Boundary-head marginals and candidates.
    text_states (`torch.FloatTensor` of shape `(batch_size, num_words, hidden_size)`, *optional*):
        Gathered word states from this forward. Attribute rescoring reuses them.
    text_mask (`torch.BoolTensor` of shape `(batch_size, num_words)`, *optional*):
        Valid word positions for `text_states`.
    query_states (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
        Gathered field-marker states from this forward.
    query_mask (`torch.BoolTensor` of shape `(batch_size, num_queries)`, *optional*):
        Valid field markers for `query_states`.
    relation_pairs (`torch.LongTensor` of shape `(num_pairs, 6)`, *optional*):
        Valid relation edges as batch, relation, head start, head end, tail start, tail end.
    relation_logits (`torch.FloatTensor` of shape `(num_pairs,)`, *optional*):
        Raw scores for `relation_pairs`, from the relation head inside `forward`.
    relation_temperature (`float`, *optional*):
        Divisor applied to `relation_logits` before the sigmoid.
    record_logits (`list`, *optional*):
        Per-sample record object logits and field-assignment logits.
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Sum of the active training objectives. Omitted when labels are absent.
    losses (`dict`, *optional*):
        Named training terms. Omitted when labels are absent.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    classification_logits: list | None = None
    span_logits: list | None = None
    counts: list | None = None
    boundary: object | None = None
    text_states: torch.FloatTensor | None = None
    text_mask: torch.Tensor | None = None
    query_states: torch.FloatTensor | None = None
    query_mask: torch.Tensor | None = None
    relation_pairs: torch.LongTensor | None = None
    relation_logits: torch.FloatTensor | None = None
    relation_temperature: float | None = None
    record_logits: list | None = None
    loss: torch.FloatTensor | None = None
    losses: dict | None = None


@auto_docstring(
    custom_intro="""
    Encoder plus the shared gather of word, query, and classification-label states.
    """
)
class Gliner2Model(Gliner2PreTrainedModel):
    def __init__(self, config: Gliner2Config):
        super().__init__(config)
        self.encoder = _load_encoder(config.encoder_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_word_indices: torch.LongTensor | None = None,
        text_word_mask: torch.Tensor | None = None,
        query_marker_indices: torch.LongTensor | None = None,
        query_marker_mask: torch.Tensor | None = None,
        cls_marker_indices: torch.LongTensor | None = None,
        cls_marker_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gliner2ModelOutput:
        r"""
        text_word_indices (`torch.LongTensor` of shape `(batch_size, num_words)`, *optional*):
            First-subword index of each word.
        text_word_mask (`torch.Tensor` of shape `(batch_size, num_words)`, *optional*):
            Mask of valid words.
        query_marker_indices (`torch.LongTensor` of shape `(batch_size, num_queries)`, *optional*):
            Token index of each field marker.
        query_marker_mask (`torch.Tensor` of shape `(batch_size, num_queries)`, *optional*):
            Mask of valid field markers.
        cls_marker_indices (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Token index of each classification label.
        cls_marker_mask (`torch.Tensor` of shape `(batch_size, num_labels)`, *optional*):
            Mask of valid classification labels.
        """
        encoder_kwargs = {
            key: kwargs[key] for key in ("output_hidden_states", "output_attentions", "return_dict") if key in kwargs
        }
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **encoder_kwargs,
        )
        hidden = encoded.last_hidden_state
        if text_word_indices is None:
            return Gliner2ModelOutput(
                last_hidden_state=hidden,
                hidden_states=encoded.hidden_states,
                attentions=encoded.attentions,
            )
        return Gliner2ModelOutput(
            last_hidden_state=hidden,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            text_states=_gather(hidden, text_word_indices, text_word_mask),
            query_states=_gather(hidden, query_marker_indices, query_marker_mask),
            cls_states=_gather(hidden, cls_marker_indices, cls_marker_mask),
        )


@auto_docstring(
    custom_intro="""
    Schema extraction over a text encoder. The span head or the boundary head is selected by
    `config.architecture`. Every label in the schema is scored in one forward pass.
    """
)
class Gliner2ForSchemaExtraction(Gliner2PreTrainedModel):
    def __init__(self, config: Gliner2Config):
        super().__init__(config)
        self.encoder = _load_encoder(config.encoder_config)
        hidden = config.encoder_config.hidden_size
        self.max_width = config.max_width
        dropout = 0.0
        if config.architecture == "boundary":
            cfg = config.boundary_config
            dropout = cfg.dropout
            self.boundary_head = BoundaryHead(
                hidden,
                cfg,
                build_candidate_states=cfg.enable_records,
            )
            self.record_decoder = (
                RecordHead(hidden, cfg.record_dim, cfg.record_instance_queries) if cfg.enable_records else None
            )
            if cfg.enable_relations:
                query_dim = hidden * 2 if cfg.directional_relation_states else hidden
                self.relation_scorer = SparseRelationScorer(
                    hidden,
                    dropout=cfg.dropout,
                    relation_query_dim=query_dim,
                    use_biaffine_content=cfg.relation_biaffine_content,
                )
                self.relation_pair_generator = TypedRelationPairGenerator(
                    heads_per_relation=cfg.relation_heads_per_type,
                    tails_per_relation=cfg.relation_tails_per_type,
                    pair_cap=cfg.relation_pair_cap,
                    argument_threshold=cfg.relation_argument_proposal_threshold,
                )
            else:
                self.relation_scorer = None
                self.relation_pair_generator = None
        else:
            self.span_rep = SpanRepLayer(hidden, config.max_width, span_mode="markerV0", dropout=0.1)
            if config.counting_layer == "count_lstm_v2":
                self.count_embed = CountLSTMv2(hidden)
            else:
                self.count_embed = CountLSTM(hidden)
            self.count_pred = _mlp(hidden, [hidden * 2], 20, dropout=0.0)
            self.boundary_head = None
            self.record_decoder = None
            self.relation_scorer = None
        self.classifier = _mlp(hidden, [hidden * 2], 1, dropout=dropout)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def score_spans(
        self,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score explicit spans on cached word states.

        The text encoder is not run again. `text_states` are the gathered
        word states from the schema forward.

        Args:
            text_states: Word states `[batch, words, hidden]`.
            text_mask: Valid words `[batch, words]`.
            query_states: Field states `[batch, queries, hidden]`.
            query_mask: Valid fields `[batch, queries]`.
            indices: Half-open spans `[batch, queries, candidates, 2]`.
            valid_mask: Optional candidate mask `[batch, queries, candidates]`.

        Returns:
            Pair logits `[batch, queries, candidates]`.
        """
        if self.boundary_head is None:
            raise ValueError("score_spans is available for boundary models")
        return self.boundary_head.score_explicit_spans(
            text_states, text_mask, query_states, query_mask, indices, valid_mask
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_word_indices: torch.LongTensor | None = None,
        text_word_mask: torch.Tensor | None = None,
        query_marker_indices: torch.LongTensor | None = None,
        query_marker_mask: torch.Tensor | None = None,
        query_group_index: torch.LongTensor | None = None,
        cls_marker_indices: torch.LongTensor | None = None,
        cls_marker_mask: torch.Tensor | None = None,
        cls_group_index: torch.LongTensor | None = None,
        prompt_marker_indices: torch.LongTensor | None = None,
        prompt_marker_mask: torch.Tensor | None = None,
        prompt_group_index: torch.LongTensor | None = None,
        task_type_ids: torch.LongTensor | None = None,
        group_mask: torch.Tensor | None = None,
        relation_head_index: torch.LongTensor | None = None,
        relation_tail_index: torch.LongTensor | None = None,
        relation_mask: torch.Tensor | None = None,
        record_mode_ids: torch.LongTensor | None = None,
        record_anchor_query: torch.LongTensor | None = None,
        record_field_query: torch.LongTensor | None = None,
        record_field_cardinality: torch.LongTensor | None = None,
        record_field_mask: torch.Tensor | None = None,
        record_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: dict | None = None,
        targets: dict | None = None,
        soft_iou_scale: float = 1.0,
        consistency_scale: float = 1.0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gliner2SchemaExtractionOutput:
        r"""
        text_word_indices (`torch.LongTensor` of shape `(batch_size, num_words)`, *optional*):
            First-subword index of each word.
        text_word_mask (`torch.Tensor` of shape `(batch_size, num_words)`, *optional*):
            Mask of valid words.
        query_marker_indices (`torch.LongTensor` of shape `(batch_size, num_queries)`, *optional*):
            Token index of each field marker.
        query_marker_mask (`torch.Tensor` of shape `(batch_size, num_queries)`, *optional*):
            Mask of valid field markers.
        query_group_index (`torch.LongTensor` of shape `(batch_size, num_queries)`, *optional*):
            Schema group of each field marker.
        cls_marker_indices (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Token index of each classification label.
        cls_marker_mask (`torch.Tensor` of shape `(batch_size, num_labels)`, *optional*):
            Mask of valid classification labels.
        cls_group_index (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Schema group of each classification label.
        prompt_marker_indices (`torch.LongTensor` of shape `(batch_size, num_prompts)`, *optional*):
            Token index of each `[P]` marker.
        prompt_marker_mask (`torch.Tensor` of shape `(batch_size, num_prompts)`, *optional*):
            Mask of valid `[P]` markers.
        prompt_group_index (`torch.LongTensor` of shape `(batch_size, num_prompts)`, *optional*):
            Schema group of each `[P]` marker.
        task_type_ids (`torch.LongTensor` of shape `(batch_size, num_groups)`, *optional*):
            Task id of each schema group. Entities are `1` and classifications are `4`.
        group_mask (`torch.Tensor` of shape `(batch_size, num_groups)`, *optional*):
            Mask of valid schema groups.
        relation_head_index (`torch.LongTensor` of shape `(batch_size, num_relations)`, *optional*):
            Query index of each relation head.
        relation_tail_index (`torch.LongTensor` of shape `(batch_size, num_relations)`, *optional*):
            Query index of each relation tail.
        relation_mask (`torch.Tensor` of shape `(batch_size, num_relations)`, *optional*):
            Mask of valid relations.
        record_mode_ids (`torch.LongTensor` of shape `(batch_size, num_records)`, *optional*):
            Record mode id. Natural is `1`, latent is `2`, and anchorless is `3`.
        record_anchor_query (`torch.LongTensor` of shape `(batch_size, num_records)`, *optional*):
            Anchor query id, or `-1` when the record has no anchor.
        record_field_query (`torch.LongTensor` of shape `(batch_size, num_records, num_fields)`, *optional*):
            Boundary query id of each record field.
        record_field_cardinality (`torch.LongTensor` of shape `(batch_size, num_records, num_fields)`, *optional*):
            Cardinality id of each record field.
        record_field_mask (`torch.Tensor` of shape `(batch_size, num_records, num_fields)`, *optional*):
            Mask of valid record fields.
        record_mask (`torch.Tensor` of shape `(batch_size, num_records)`, *optional*):
            Mask of valid record groups.
        labels (`dict`, *optional*):
            Training targets. Span keys are `classification_targets` and `span_structures`.
            Boundary keys include `mention_pairs`, `mention_mask`, `record_groups`, and
            `relation_gold_pairs`. Absent labels keep this call inference-only.
        targets (`dict`, *optional*):
            Extra target tensors merged into `labels`.
        soft_iou_scale (`float`, *optional*, defaults to 1.0):
            Multiplier on the soft-IoU term. The trainer anneals this value.
        consistency_scale (`float`, *optional*, defaults to 1.0):
            Multiplier on the marginal-consistency term.
        """
        encoder_kwargs = {
            key: kwargs[key] for key in ("output_hidden_states", "output_attentions", "return_dict") if key in kwargs
        }
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **encoder_kwargs,
        )
        hidden = encoded.last_hidden_state
        supervision = _merged_labels(labels, targets)
        if text_word_indices is None:
            if supervision is not None:
                raise ValueError("labels require text_word_indices")
            return Gliner2SchemaExtractionOutput(
                last_hidden_state=hidden,
                hidden_states=encoded.hidden_states,
                attentions=encoded.attentions,
            )

        text_states = _gather(hidden, text_word_indices, text_word_mask)
        query_states = _gather(hidden, query_marker_indices, query_marker_mask)
        cls_states = _gather(hidden, cls_marker_indices, cls_marker_mask)
        if self.config.architecture == "boundary":
            head = self.boundary_head
            if head is not None and soft_iou_scale == 1.0:
                soft_iou_scale = head._soft_iou_scale
            if head is not None and consistency_scale == 1.0:
                consistency_scale = head._consistency_scale
            return self._boundary_forward(
                hidden,
                encoded,
                text_states,
                text_word_mask,
                query_states,
                query_marker_mask,
                cls_states,
                cls_marker_mask,
                cls_group_index,
                task_type_ids,
                group_mask,
                supervision,
                soft_iou_scale,
                consistency_scale,
                relation_head_index,
                relation_tail_index,
                relation_mask,
                record_mode_ids,
                record_anchor_query,
                record_field_query,
                record_field_cardinality,
                record_field_mask,
                record_mask,
            )

        return self._span_forward(
            hidden,
            encoded,
            text_states,
            text_word_mask,
            query_states,
            query_marker_mask,
            query_group_index,
            cls_states,
            cls_marker_mask,
            cls_group_index,
            prompt_marker_indices,
            prompt_marker_mask,
            prompt_group_index,
            task_type_ids,
            group_mask,
            supervision,
        )

    def _span_forward(
        self,
        hidden,
        encoded,
        text_states,
        text_word_mask,
        query_states,
        query_marker_mask,
        query_group_index,
        cls_states,
        cls_marker_mask,
        cls_group_index,
        prompt_marker_indices,
        prompt_marker_mask,
        prompt_group_index,
        task_type_ids,
        group_mask,
        supervision,
    ) -> Gliner2SchemaExtractionOutput:
        """Score span groups. Gold counts replace predicted counts when targets exist."""
        mention_mask = None if supervision is None else supervision.get("mention_mask")
        if (
            supervision is not None
            and supervision.get("mention_pairs") is not None
            and mention_mask is not None
            and bool(mention_mask.any())
        ):
            raise ValueError("mention targets require a boundary model")
        if supervision is not None and _has_boundary_targets(supervision):
            raise ValueError("record and relation targets require a boundary model")
        classification_logits = _classification_logits(
            cls_states, cls_marker_mask, cls_group_index, task_type_ids, group_mask, self.classifier
        )
        prompt_states = None
        if prompt_marker_indices is not None:
            prompt_states = _gather(hidden, prompt_marker_indices, prompt_marker_mask)
        loss = None
        losses = None
        structures = None if supervision is None else supervision.get("span_structures")
        if structures is None:
            span_logits, counts = _span_logits(
                text_states,
                text_word_mask,
                query_states,
                query_marker_mask,
                query_group_index,
                prompt_states,
                prompt_marker_mask,
                prompt_group_index,
                task_type_ids,
                group_mask,
                self.span_rep,
                self.count_pred,
                self.count_embed,
                self.max_width,
            )
        else:
            span_logits, counts, struct_loss, count_loss = self._gold_span_logits(
                text_states,
                text_word_mask,
                query_states,
                query_marker_mask,
                query_group_index,
                prompt_states,
                prompt_marker_mask,
                prompt_group_index,
                task_type_ids,
                group_mask,
                structures,
            )
        if supervision is not None:
            anchor = text_states.sum() * 0.0
            cls_loss = anchor
            if supervision.get("classification_targets") is not None:
                cls_loss = loss_gliner2.summed_classification_loss(
                    classification_logits, supervision["classification_targets"]
                )
            if structures is None:
                struct_loss = anchor
                count_loss = anchor
            loss = cls_loss + struct_loss + count_loss
            losses = {
                "classification_loss": cls_loss,
                "structure_loss": struct_loss,
                "count_loss": count_loss,
                "loss": loss,
            }
        return Gliner2SchemaExtractionOutput(
            last_hidden_state=hidden,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            classification_logits=classification_logits,
            span_logits=span_logits,
            counts=counts,
            text_states=text_states,
            text_mask=text_word_mask,
            query_states=query_states,
            query_mask=query_marker_mask,
            loss=loss,
            losses=losses,
        )

    def _gold_span_logits(
        self,
        text_states,
        text_mask,
        query_states,
        query_mask,
        query_groups,
        prompt_states,
        prompt_mask,
        prompt_groups,
        task_ids,
        group_mask,
        structures,
    ):
        """Span logits from `count_embed` at `min(count, 19)`, plus span and count losses."""
        if query_groups is None or task_ids is None or prompt_states is None or group_mask is None:
            raise ValueError("span targets require query, prompt, task, and group tensors")
        if len(structures) != text_states.shape[0]:
            raise ValueError("span_structures must have one entry per batch row")
        span_rows = []
        count_rows = []
        struct_loss = text_states.sum() * 0.0
        count_prompt = []
        count_target = []
        for index in range(text_states.shape[0]):
            words = text_states[index][text_mask[index].bool()]
            length = words.shape[0]
            rep = None
            if length:
                indices = _span_indices(length, self.max_width, words.device)
                rep = self.span_rep(words.unsqueeze(0), indices).squeeze(0)
            sample_spans = []
            sample_counts = []
            sample_structures = structures[index]
            slot = 0
            for group in range(task_ids.shape[1]):
                if not bool(group_mask[index, group]) or int(task_ids[index, group]) == _CLASSIFICATION_TASK:
                    continue
                if slot >= len(sample_structures):
                    raise ValueError("span_structures has fewer groups than the schema")
                structure = sample_structures[slot]
                slot += 1
                fields = _rows(query_states[index], query_mask[index], query_groups[index], group)
                prompt = _rows(prompt_states[index], prompt_mask[index], prompt_groups[index], group)
                gold = loss_gliner2.clamp_gold_count(
                    structure.shape[0] if torch.is_tensor(structure) else structure[0]
                )
                task_id = int(task_ids[index, group])
                scored = False
                if gold > 0 and prompt.numel() > 0 and fields.numel() > 0 and rep is not None:
                    if torch.is_tensor(structure):
                        structure = structure[:gold]
                    projected = self.count_embed(fields, gold)
                    scores = count_conditioned_scores(rep, projected)
                    sample_spans.append(scores)
                    sample_counts.append(gold)
                    span_mask = loss_gliner2.invalid_span_mask(length, self.max_width, scores.device)
                    struct_loss = struct_loss + loss_gliner2.span_structure_loss(
                        scores, structure, span_mask, training=self.training
                    )
                    scored = True
                else:
                    sample_spans.append(text_states.new_zeros(0, fields.shape[0], length, self.max_width))
                    sample_counts.append(0)
                if (
                    gold > 0
                    and prompt.numel() > 0
                    and loss_gliner2.supervises_count(task_id)
                    and (scored or rep is None or fields.numel() == 0)
                ):
                    count_prompt.append(prompt[:1])
                    count_target.append(gold)
            if slot != len(sample_structures):
                raise ValueError("span_structures has more groups than the schema")
            span_rows.append(sample_spans)
            count_rows.append(sample_counts)
        if count_prompt:
            count_logits = self.count_pred(torch.cat(count_prompt, dim=0))
            count_loss = loss_gliner2.span_count_loss(
                count_logits, torch.tensor(count_target, dtype=torch.long, device=count_logits.device)
            )
        else:
            count_loss = text_states.new_zeros(())
        return span_rows, count_rows, struct_loss, count_loss

    def _boundary_forward(
        self,
        hidden,
        encoded,
        text_states,
        text_word_mask,
        query_states,
        query_marker_mask,
        cls_states,
        cls_marker_mask,
        cls_group_index,
        task_type_ids,
        group_mask,
        supervision,
        soft_iou_scale,
        consistency_scale,
        relation_head_index,
        relation_tail_index,
        relation_mask,
        record_mode_ids,
        record_anchor_query,
        record_field_query,
        record_field_cardinality,
        record_field_mask,
        record_mask,
    ) -> Gliner2SchemaExtractionOutput:
        """Boundary forward. Gold spans are injected only while this module is training."""
        if supervision is not None and supervision.get("span_structures") and supervision.get("mention_pairs") is None:
            raise ValueError("span_structures require a span model")
        head_targets = None
        if supervision is not None and supervision.get("mention_pairs") is not None:
            if supervision.get("mention_mask") is None:
                raise ValueError("mention_pairs require mention_mask")
            head_targets = SimpleNamespace(
                mention_pairs=supervision["mention_pairs"].to(text_states.device),
                mention_mask=supervision["mention_mask"].to(text_states.device),
            )
        # Classification-only batches have no extractive queries to score.
        if query_states.shape[1] > 0:
            boundary = self.boundary_head(
                text_states,
                text_word_mask,
                query_states,
                query_marker_mask,
                head_targets,
                return_candidates=True,
            )
        else:
            boundary = None
        classification_logits = _classification_logits(
            cls_states, cls_marker_mask, cls_group_index, task_type_ids, group_mask, self.classifier
        )
        loss = None
        losses = None
        if supervision is not None:
            loss, losses = self._boundary_loss(
                text_states,
                text_word_mask,
                query_states,
                query_marker_mask,
                boundary,
                classification_logits,
                supervision,
                soft_iou_scale,
                consistency_scale,
            )
        relation_pairs = None
        relation_logits = None
        relation_temperature = None
        record_logits = None
        if not self.training:
            relation_pairs, relation_logits = self._inference_relations(
                text_states, query_states, boundary, relation_head_index, relation_tail_index, relation_mask
            )
            record_logits = self._inference_records(
                query_states,
                boundary,
                record_mode_ids,
                record_anchor_query,
                record_field_query,
                record_field_cardinality,
                record_field_mask,
                record_mask,
            )
            if relation_logits is not None:
                relation_temperature = float(self.config.boundary_config.relation_temperature)
        return Gliner2SchemaExtractionOutput(
            last_hidden_state=hidden,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
            classification_logits=classification_logits,
            boundary=boundary,
            text_states=text_states,
            text_mask=text_word_mask,
            query_states=query_states,
            query_mask=query_marker_mask,
            relation_pairs=relation_pairs,
            relation_logits=relation_logits,
            relation_temperature=relation_temperature,
            record_logits=record_logits,
            loss=loss,
            losses=losses,
        )

    def _inference_relations(self, text_states, query_states, boundary, head_index, tail_index, relation_mask):
        """Score typed relation pairs. Returns coordinate rows and raw logits."""
        if (
            self.relation_scorer is None
            or self.relation_pair_generator is None
            or boundary is None
            or boundary.candidates is None
            or head_index is None
            or tail_index is None
            or relation_mask is None
            or query_states.shape[1] == 0
            or not bool(relation_mask.any())
        ):
            return None, None
        device = text_states.device
        head_index = head_index.to(device)
        tail_index = tail_index.to(device)
        relation_mask = relation_mask.to(device).bool()
        batch, relations = head_index.shape
        queries = query_states.shape[1]
        hidden = query_states.shape[-1]
        head_slot = head_index.clamp(0, queries - 1)
        tail_slot = tail_index.clamp(0, queries - 1)
        head_states = query_states.gather(1, head_slot.unsqueeze(-1).expand(-1, -1, hidden))
        tail_states = query_states.gather(1, tail_slot.unsqueeze(-1).expand(-1, -1, hidden))
        directional = bool(self.config.boundary_config.directional_relation_states)
        if directional:
            relation_states = torch.cat((head_states, tail_states), dim=-1)
        else:
            relation_states = (head_states + tail_states) * 0.5
        head_member = torch.zeros(batch, relations, queries, dtype=torch.bool, device=device)
        tail_member = torch.zeros_like(head_member)
        batch_index = torch.arange(batch, device=device)[:, None].expand_as(head_index)
        relation_index = torch.arange(relations, device=device)[None, :].expand_as(head_index)
        valid = relation_mask
        head_member[batch_index[valid], relation_index[valid], head_slot[valid]] = True
        tail_member[batch_index[valid], relation_index[valid], tail_slot[valid]] = True
        allow_self = torch.zeros(batch, relations, dtype=torch.bool, device=device)
        pairs = self.relation_pair_generator.generate_batched(
            boundary.candidates,
            [None] * batch,
            [[] for _ in range(batch)],
            compact=False,
            routing=(head_member, tail_member, relation_mask, allow_self),
        )
        logits = self.relation_scorer(text_states, relation_states, boundary.candidates, pairs)
        keep = pairs.pair_mask
        if keep is None or not bool(keep.any()):
            return text_states.new_zeros(0, 6).long(), text_states.new_zeros(0)
        coords = torch.stack(
            (
                pairs.batch_index,
                pairs.relation_index,
                pairs.head_start,
                pairs.head_end,
                pairs.tail_start,
                pairs.tail_end,
            ),
            dim=-1,
        )
        return coords[keep].long(), logits[keep]

    def _inference_records(
        self,
        query_states,
        boundary,
        mode_ids,
        anchor_query,
        field_query,
        field_cardinality,
        field_mask,
        record_mask,
    ):
        """Score each record group. One `(object_logits, assign_logits)` pair per group."""
        if (
            self.record_decoder is None
            or boundary is None
            or boundary.candidates is None
            or getattr(boundary.candidates, "candidate_states", None) is None
            or record_mask is None
            or mode_ids is None
            or field_query is None
            or field_cardinality is None
            or field_mask is None
            or not bool(record_mask.any())
        ):
            return None
        device = query_states.device
        record_mask = record_mask.to(device).bool()
        rows = []
        for sample_index in range(query_states.shape[0]):
            sample = []
            for record_index in range(record_mask.shape[1]):
                if not bool(record_mask[sample_index, record_index]):
                    continue
                spec = _record_spec(
                    int(mode_ids[sample_index, record_index]),
                    int(anchor_query[sample_index, record_index]),
                    field_query[sample_index, record_index],
                    field_cardinality[sample_index, record_index],
                    field_mask[sample_index, record_index],
                )
                decoded = self.record_decoder.forward_group(
                    spec, query_states[sample_index], boundary.candidates, sample_index
                )
                sample.append((decoded.object_logits, decoded.assign_logits))
            rows.append(sample)
        return rows

    def _boundary_loss(
        self,
        text_states,
        text_mask,
        query_states,
        query_mask,
        boundary,
        classification_logits,
        supervision,
        soft_iou_scale,
        consistency_scale,
    ):
        """Combine boundary, classification, record, and relation terms."""
        cfg = self.config.boundary_config
        total = text_states.sum() * 0.0
        losses = {}
        mention_pairs = supervision.get("mention_pairs")
        if mention_pairs is not None:
            if supervision.get("mention_mask") is None:
                raise ValueError("mention_pairs require mention_mask")
            if boundary.candidates is None or boundary.start_logits is None:
                raise ValueError("boundary targets require candidate and marginal logits")
            candidates = boundary.candidates
            terms = loss_gliner2.boundary_training_loss(
                start_logits=boundary.start_logits,
                end_logits=boundary.end_logits,
                inside_logits=boundary.inside_logits,
                pair_logits=candidates.pair_logits,
                candidate_indices=candidates.indices,
                candidate_valid=candidates.valid_mask,
                proposal_logits=candidates.proposal_logits,
                proposal_gold=getattr(candidates, "gold_mask", None),
                null_logits=boundary.null_logits,
                count_log_rates=boundary.count_log_rates,
                mention_pairs=mention_pairs,
                mention_mask=supervision["mention_mask"],
                query_mask=query_mask,
                text_mask=text_mask,
                settings=cfg,
                training=self.training,
                start_targets=supervision.get("start_targets"),
                end_targets=supervision.get("end_targets"),
                inside_targets=supervision.get("inside_targets"),
                soft_iou_scale=soft_iou_scale,
                consistency_scale=consistency_scale,
            )
            losses.update(terms)
            total = total + terms["loss"]
        if supervision.get("classification_targets") is not None:
            cls_loss = loss_gliner2.mean_classification_loss(
                classification_logits,
                supervision["classification_targets"],
                cfg.classification_loss_weight,
            )
            losses["classification_loss"] = cls_loss
            total = total + cls_loss
        if supervision.get("record_groups") is not None and supervision.get("dense_records") is not None:
            raise ValueError("pass record_groups or dense_records, not both")
        record_groups = supervision.get("record_groups")
        if record_groups is not None:
            if self.record_decoder is None or boundary.candidates is None:
                raise ValueError("record_groups require enable_records")
            if boundary.candidates.candidate_states is None:
                raise ValueError("record_groups require candidate states")
            parts = []
            for sample_index, groups in enumerate(record_groups):
                for group in groups:
                    decoded = self.record_decoder.forward_group(
                        loss_gliner2.build_record_spec(group),
                        query_states[sample_index],
                        boundary.candidates,
                        sample_index,
                    )
                    parts.append(
                        loss_gliner2.compute_record_group_loss(decoded, loss_gliner2.build_record_targets(group))
                    )
            packed = loss_gliner2.aggregate_record_losses(parts, cfg.record_loss_weight)
            losses["record_object_loss"] = packed["object"]
            losses["record_field_loss"] = packed["field"]
            total = total + packed["total"]
        dense_records = supervision.get("dense_records")
        if dense_records is not None:
            if isinstance(dense_records, dict):
                dense_records = SimpleNamespace(**dense_records)
            packed = loss_gliner2.dense_record_batch_loss(dense_records)
            losses["record_object_loss"] = packed["object_loss"]
            losses["record_field_loss"] = packed["field_loss"]
            total = total + cfg.record_loss_weight * (packed["object_loss"] + packed["field_loss"])
        if supervision.get("relation_gold_pairs") is not None:
            if self.relation_scorer is None or self.relation_pair_generator is None:
                raise ValueError("relation targets require enable_relations")
            if boundary.candidates is None:
                raise ValueError("relation targets require boundary candidates")
            routing = supervision.get("relation_routing")
            rel_states = supervision.get("relation_query_states")
            if rel_states is None and routing is not None:
                rel_states = _states_from_routing(
                    query_states,
                    routing,
                    bool(self.config.boundary_config.directional_relation_states),
                )
            if routing is None or rel_states is None or supervision.get("relation_gold_mask") is None:
                raise ValueError("relation targets require routing, query states, and a gold mask")
            batch = text_states.shape[0]
            routing = tuple(item.to(text_states.device) for item in routing)
            pairs = self.relation_pair_generator.generate_batched(
                boundary.candidates,
                [None] * batch,
                [[] for _ in range(batch)],
                compact=False,
                routing=routing,
            )
            logits = self.relation_scorer(text_states, rel_states.to(text_states.device), boundary.candidates, pairs)
            rel_loss = loss_gliner2.sparse_relation_loss(
                logits,
                pairs,
                supervision["relation_gold_pairs"],
                supervision["relation_gold_mask"],
                cfg.relation_loss_weight,
            )
            losses["relation_loss"] = rel_loss
            total = total + rel_loss
        total = total + loss_gliner2.head_touch((self.record_decoder, self.relation_scorer), text_states.device)
        losses["loss"] = total
        return total, losses


_RECORD_ID_TO_MODE = {1: "natural", 2: "latent", 3: "anchorless"}


def _has_boundary_targets(supervision) -> bool:
    """True when the batch contains relation edges or record groups."""
    records = supervision.get("record_groups")
    if records and any(records):
        return True
    mask = supervision.get("relation_gold_mask")
    return mask is not None and bool(mask.any())


def _merged_labels(labels, targets):
    """Merge two target dicts. The processor emits the keys the loss reads."""
    if labels is None and targets is None:
        return None
    if labels is not None and not isinstance(labels, dict):
        raise TypeError("labels must be a dict")
    if targets is not None and not isinstance(targets, dict):
        raise TypeError("targets must be a dict")
    if labels is None:
        return targets
    if targets is None:
        return labels
    return {**labels, **targets}


def _states_from_routing(query_states, routing, directional):
    """Relation queries from boolean head and tail membership."""
    head_member, tail_member = routing[0], routing[1]
    head_member = head_member.to(device=query_states.device, dtype=query_states.dtype)
    tail_member = tail_member.to(device=query_states.device, dtype=query_states.dtype)
    head = torch.einsum("brq,bqh->brh", head_member, query_states)
    tail = torch.einsum("brq,bqh->brh", tail_member, query_states)
    if directional:
        return torch.cat((head, tail), dim=-1)
    return (head + tail) * 0.5


def _record_spec(mode_id, anchor, field_query, field_cardinality, field_mask):
    """Record schema object `RecordHead.forward_group` reads."""
    try:
        mode = _RECORD_ID_TO_MODE[int(mode_id)]
    except KeyError:
        raise ValueError(f"unknown record mode id {mode_id}") from None
    fields = []
    for query_id, cardinality, keep in zip(field_query.tolist(), field_cardinality.tolist(), field_mask.tolist()):
        if not keep:
            continue
        fields.append(
            SimpleNamespace(
                query_id=int(query_id),
                cardinality=SimpleNamespace(is_scalar=int(cardinality) in (1, 2)),
            )
        )
    anchor_id = int(anchor)
    return SimpleNamespace(
        mode=mode,
        fields=fields,
        anchor_query_id=None if anchor_id < 0 else anchor_id,
    )


def _load_encoder(encoder_config):
    """Build the encoder from its config. DeBERTa rejects SDPA and falls back to eager."""
    try:
        return AutoModel.from_config(encoder_config)
    except (ValueError, RuntimeError):
        encoder_config._attn_implementation = "eager"
        return AutoModel.from_config(encoder_config)


def _classification_logits(cls_states, cls_mask, cls_groups, task_ids, group_mask, classifier):
    """One raw logit vector per classification group."""
    if cls_groups is None or task_ids is None:
        return None
    batch = cls_states.shape[0]
    rows = []
    for index in range(batch):
        sample = []
        for group in range(task_ids.shape[1]):
            if not bool(group_mask[index, group]) or int(task_ids[index, group]) != _CLASSIFICATION_TASK:
                continue
            states = _rows(cls_states[index], cls_mask[index], cls_groups[index], group)
            if states.numel() == 0:
                sample.append(states.new_zeros(0))
            else:
                sample.append(classifier(states).squeeze(-1))
        rows.append(sample)
    return rows


def count_conditioned_scores(span_rep: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
    """Score spans with count-conditioned field states."""
    return torch.einsum("lkd,bpd->bplk", span_rep, projected)


def _span_logits(
    text_states,
    text_mask,
    query_states,
    query_mask,
    query_groups,
    prompt_states,
    prompt_mask,
    prompt_groups,
    task_ids,
    group_mask,
    span_rep,
    count_pred,
    count_embed,
    max_width,
):
    """Pre-sigmoid span logits and instance counts, one entry per span group."""
    if query_groups is None or task_ids is None or prompt_states is None:
        return None, None
    span_rows = []
    count_rows = []
    for index in range(text_states.shape[0]):
        words = text_states[index][text_mask[index]]
        length = words.shape[0]
        rep = None
        if length:
            indices = _span_indices(length, max_width, words.device)
            rep = span_rep(words.unsqueeze(0), indices).squeeze(0)
        sample_spans = []
        sample_counts = []
        for group in range(task_ids.shape[1]):
            if not bool(group_mask[index, group]) or int(task_ids[index, group]) == _CLASSIFICATION_TASK:
                continue
            fields = _rows(query_states[index], query_mask[index], query_groups[index], group)
            prompt = _rows(prompt_states[index], prompt_mask[index], prompt_groups[index], group)
            if prompt.numel() == 0 or fields.numel() == 0 or rep is None:
                width = max_width
                sample_spans.append(text_states.new_zeros(0, fields.shape[0], length, width))
                sample_counts.append(0)
                continue
            predicted = int(count_pred(prompt[:1]).argmax(dim=-1).item())
            sample_counts.append(predicted)
            if predicted <= 0:
                sample_spans.append(text_states.new_zeros(0, fields.shape[0], length, max_width))
                continue
            projected = count_embed(fields, predicted)
            sample_spans.append(count_conditioned_scores(rep, projected))
        span_rows.append(sample_spans)
        count_rows.append(sample_counts)
    return span_rows, count_rows


__all__ = [
    "Gliner2PreTrainedModel",
    "Gliner2Model",
    "Gliner2ForSchemaExtraction",
    "Gliner2ModelOutput",
    "Gliner2SchemaExtractionOutput",
]
