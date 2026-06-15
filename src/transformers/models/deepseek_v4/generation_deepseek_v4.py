# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Static caches for DeepSeek-V4 generation. All compressor buffers are
pre-allocated so `cache_implementation="static"` runs under
`torch.compile(fullgraph=True)` without re-tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...cache_utils import StaticSlidingWindowLayer
from ...utils.import_utils import is_torchdynamo_compiling


if TYPE_CHECKING:
    from .configuration_deepseek_v4 import DeepseekV4Config


@dataclass
class _CompressorState:
    """All buffers and counters one named compressor needs. Tensors are filled
    in by `lazy_initialization` so they pick up the model's device and dtype.
    """

    buffer_dim: int
    compressed_dim: int
    overlap_dim: int = 0  # CSA only; HCA has non-overlapping windows
    buffer_kv: torch.Tensor | None = None
    buffer_gate: torch.Tensor | None = None
    compressed_kv: torch.Tensor | None = None
    overlap_kv: torch.Tensor | None = None
    overlap_gate: torch.Tensor | None = None
    cumulative_position: int = 0


class DeepseekV4HCAStaticCache(StaticSlidingWindowLayer):
    r"""Static HCA cache: sliding-window K=V buffers from the parent class plus,
    per named compressor, a rolling source-token buffer
    `(B, compress_rate, head_dim)` and compressed-entry storage
    `(B, ceil(max_cache_len / compress_rate), head_dim)`.

    Each decode step writes a candidate entry into the open window's slot and
    overwrites it until that window closes. Attention masks out un-closed slots,
    so the intermediate writes don't reach any query.
    """

    def __init__(self, config: DeepseekV4Config, max_cache_len: int):
        super().__init__(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
        self.config = config
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        # Ceil so the final partial window has a slot reserved.
        self._max_compressed_entries = (max_cache_len + self.compress_rate - 1) // self.compress_rate
        # HCA has just one compressor entry. CSA's subclass overwrites this dict
        # to add `"indexer"` with different feature widths.
        self.compressors: dict[str, _CompressorState] = {
            "compressor": _CompressorState(buffer_dim=config.head_dim, compressed_dim=config.head_dim),
        }

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Allocate the per-compressor buffers on the first update."""
        super().lazy_initialization(key_states, value_states)
        batch, dtype, device = key_states.shape[0], self.dtype, self.device
        for state in self.compressors.values():
            state.buffer_kv = torch.zeros(batch, self.compress_rate, state.buffer_dim, dtype=dtype, device=device)
            state.buffer_gate = torch.zeros(batch, self.compress_rate, state.buffer_dim, dtype=dtype, device=device)
            state.compressed_kv = torch.zeros(
                batch, self._max_compressed_entries, state.compressed_dim, dtype=dtype, device=device
            )
            if state.overlap_dim:
                state.overlap_kv = torch.zeros(
                    batch, self.compress_rate, state.overlap_dim, dtype=dtype, device=device
                )
                state.overlap_gate = torch.zeros(
                    batch, self.compress_rate, state.overlap_dim, dtype=dtype, device=device
                )
        if not is_torchdynamo_compiling():
            for state in self.compressors.values():
                torch._dynamo.mark_static_address(state.buffer_kv)
                torch._dynamo.mark_static_address(state.buffer_gate)
                torch._dynamo.mark_static_address(state.compressed_kv)
                if state.overlap_kv is not None:
                    torch._dynamo.mark_static_address(state.overlap_kv)
                    torch._dynamo.mark_static_address(state.overlap_gate)

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Append `(kv, gate)` to the named compressor's rolling buffer and
        return `(chunk_kv, chunk_gate, first_window_position)` — the chunk is
        the longest window-aligned prefix ready to compress.

        For a single-token call (decode) the buffer is shifted left by one and
        the new token appended; the whole buffer comes back so the compressor
        can emit a candidate even before the first window fills (left slots are
        zero). For a multi-token call (prefill) the buffer's real tail is
        concatenated with `kv`, the longest window-aligned prefix is sliced
        off, and the buffer is refreshed with the stream's last `compress_rate`
        tokens.
        """
        state = self.compressors[name]
        new_n = kv.shape[1]
        cum_before = state.cumulative_position
        first_window_position = (cum_before // self.compress_rate) * self.compress_rate

        if new_n == 1:
            new_buffer_kv = torch.cat([state.buffer_kv[:, 1:], kv], dim=1)
            new_buffer_gate = torch.cat([state.buffer_gate[:, 1:], gate], dim=1)
            state.buffer_kv.copy_(new_buffer_kv)
            state.buffer_gate.copy_(new_buffer_gate)
            state.cumulative_position = cum_before + 1
            return state.buffer_kv, state.buffer_gate, first_window_position

        real_in_buffer = min(cum_before, self.compress_rate)
        if real_in_buffer > 0:
            prior_kv = state.buffer_kv[:, -real_in_buffer:]
            prior_gate = state.buffer_gate[:, -real_in_buffer:]
            full_kv = torch.cat([prior_kv, kv], dim=1)
            full_gate = torch.cat([prior_gate, gate], dim=1)
        else:
            full_kv, full_gate = kv, gate
        usable = (full_kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv = full_kv[:, :usable]
        chunk_gate = full_gate[:, :usable]

        state.buffer_kv.zero_()
        state.buffer_gate.zero_()
        tail = min(full_kv.shape[1], self.compress_rate)
        if tail > 0:
            state.buffer_kv[:, -tail:].copy_(full_kv[:, -tail:])
            state.buffer_gate[:, -tail:].copy_(full_gate[:, -tail:])
        state.cumulative_position = cum_before + new_n
        return chunk_kv, chunk_gate, first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        """Write the new compressed entries into the named compressor's storage
        and return the full storage tensor. Decode writes one candidate per step
        into the open window's slot; prefill writes a contiguous run starting at
        the first newly-closed window's slot. Attention's causal mask hides
        un-closed slots, so candidate writes don't leak into the output.
        """
        state = self.compressors[name]
        n_new = compressed.shape[1]
        if n_new == 0:
            return state.compressed_kv
        # Span is 1 in decode (one candidate per token) and compress_rate in prefill
        # (one entry per closed window). cumulative_position already advanced inside
        # store_compression_weights, so back out by n_new * span to recover the slot
        # of the first new entry.
        span = 1 if n_new == 1 else self.compress_rate
        start_slot = (state.cumulative_position - n_new * span) // self.compress_rate
        buf = state.compressed_kv
        buf[:, start_slot : start_slot + n_new].copy_(compressed.to(buf.dtype))
        return buf

    def reset(self) -> None:
        super().reset()
        for state in self.compressors.values():
            if state.buffer_kv is not None:
                state.buffer_kv.zero_()
                state.buffer_gate.zero_()
                state.compressed_kv.zero_()
            if state.overlap_kv is not None:
                state.overlap_kv.zero_()
                state.overlap_gate.zero_()
            state.cumulative_position = 0


class DeepseekV4CSAStaticCache(DeepseekV4HCAStaticCache):
    r"""Static CSA cache. Adds an `"indexer"` compressor and per-compressor
    overlap state on top of HCA's single `"compressor"`.

    Feature widths reflect the two-series window scheme: `kv_proj` /
    `gate_proj` emit `2 * head_dim` per token (Ca and Cb halves) and the
    softmax-gated combine produces `head_dim`-wide entries. The indexer
    follows the same shape at `index_head_dim`.

    Overlap state carries the previous window's Ca slice into the next
    gating combine. Starting at zeros matches the dynamic cache's first-call
    `None`, since the consumer assigns into the slot rather than adding.
    """

    def __init__(self, config: DeepseekV4Config, max_cache_len: int):
        super().__init__(config=config, max_cache_len=max_cache_len)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self._max_compressed_entries = (max_cache_len + self.compress_rate - 1) // self.compress_rate
        h, ih = config.head_dim, config.index_head_dim
        self.compressors = {
            "compressor": _CompressorState(buffer_dim=2 * h, compressed_dim=h, overlap_dim=h),
            "indexer": _CompressorState(buffer_dim=2 * ih, compressed_dim=ih, overlap_dim=ih),
        }

    def update_overlap_state(
        self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the saved Ca slice (zeros on the first call) and stash this
        chunk's last-window Ca for the next call. Only the `:head_dim` slice
        is kept; Cb has already been folded into the previous compressed entry.
        """
        state = self.compressors[name]
        prior_kv = state.overlap_kv.clone()
        prior_gate = state.overlap_gate.clone()
        state.overlap_kv.copy_(chunk_kv[:, -1, :, :head_dim])
        state.overlap_gate.copy_(chunk_gate[:, -1, :, :head_dim])
        return prior_kv, prior_gate
