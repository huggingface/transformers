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
"""Static cache classes for DeepSeek-V4 generation.

The dynamic counterparts (``DeepseekV4HCACache``, ``DeepseekV4CSACache``) live
in ``modeling_deepseek_v4.py`` and are used during training and short-context
inference. The classes here pre-allocate every compressor buffer at construction
time so ``model.generate(..., cache_implementation="static")`` and
``torch.compile(fullgraph=True)`` can both run without re-tracing on cache growth.
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
    """All the buffers and counters one named compressor needs in a static cache.

    HCA has a single compressor entry (``"compressor"``). CSA adds a second one
    (``"indexer"``) with different feature widths. Each entry's tensors get
    allocated lazily on the first ``update`` call so the actual device/dtype
    matches the live model parameters.
    """

    buffer_dim: int
    compressed_dim: int
    # Width of the per-window Ca slice persisted across CSA forward calls. HCA
    # uses non-overlapping windows so this stays 0 / unused.
    overlap_dim: int = 0
    buffer_kv: torch.Tensor | None = None
    buffer_gate: torch.Tensor | None = None
    compressed_kv: torch.Tensor | None = None
    overlap_kv: torch.Tensor | None = None
    overlap_gate: torch.Tensor | None = None
    cumulative_position: int = 0


class DeepseekV4HCAStaticCache(StaticSlidingWindowLayer):
    r"""Static counterpart to ``DeepseekV4HCACache`` for HCA blocks.

    Same compressor-state API as the dynamic cache (``store_compression_weights``,
    ``update_compressor_states``) but everything sits in pre-allocated tensors so
    the whole forward is dynamo-traceable. On top of the parent's sliding-window
    K=V buffers we maintain, per named compressor:

      * a rolling source-token buffer of shape ``(B, compress_rate, head_dim)``
        — the most recent ``compress_rate`` tokens, zero-padded on the left
        until the first full window;
      * pre-allocated storage for emitted compressed entries of shape
        ``(B, ceil(max_cache_len / compress_rate), head_dim)``;
      * an integer counter tracking how many source tokens have been written.

    During decode, the compressor emits one *candidate* entry per token into the
    slot of the currently-open window. The same slot is overwritten on every
    step until the window closes, at which point the write commits the real
    entry. Attention masks out un-closed slots, so the in-progress overwrites
    are never read by a query.

    No ``layer_type`` attribute is set on this class — ``__init_subclass__``
    would register it in ``LAYER_TYPE_CACHE_MAPPING`` and clobber the dynamic
    counterpart's registration. ``StaticCache.__init__`` dispatches to this
    class via an explicit branch instead.
    """

    def __init__(self, config: DeepseekV4Config, max_cache_len: int):
        super().__init__(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
        self.config = config
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        # Ceil so the final partial window has a slot reserved.
        self._max_compressed_entries = (max_cache_len + self.compress_rate - 1) // self.compress_rate
        # HCA has just one compressor entry. CSA's subclass overwrites this dict
        # to add ``"indexer"`` with different feature widths.
        self.compressors: dict[str, _CompressorState] = {
            "compressor": _CompressorState(buffer_dim=config.head_dim, compressed_dim=config.head_dim),
        }

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Allocate every compressor's tensors at first call. Runs alongside the
        parent's sliding K=V allocation so the dispatch order doesn't matter."""
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
        """Place new projected ``(kv, gate)`` into the named compressor's rolling
        buffer and return the chunk that's ready to compress.

        In decode (one new token), we shift the buffer left by one slot, append
        the new token at the right, and hand the whole buffer back so the
        compressor can always emit a candidate entry. The buffer's zero padding
        on the left covers the case where the first window isn't full yet.

        In prefill (many new tokens), we concatenate the buffer's real tail
        (``min(cum, compress_rate)`` tokens) with the new ``kv``, slice off the
        longest window-aligned prefix, and refresh the rolling buffer with the
        last ``compress_rate`` tokens of the combined stream.

        Returns ``(chunk_kv, chunk_gate, first_window_position)``.
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
        and return the full storage tensor.

        Decode writes one candidate per step into the in-progress window's slot
        (overwritten until the window closes). Prefill writes a contiguous run
        of ``n_new`` entries starting at the first newly-closed window's slot.
        Attention's existing ``causal_threshold = (position_ids + 1) //
        compress_rate`` mask hides slots whose window hasn't closed yet, so
        the still-unwritten / candidate-only slots don't affect output.
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
    r"""Static counterpart to ``DeepseekV4CSACache`` for CSA blocks.

    Adds an ``"indexer"`` compressor on top of HCA's single ``"compressor"``,
    plus per-compressor overlap state for the two-series window scheme. The
    feature widths differ: CSA's ``kv_proj``/``gate_proj`` emit ``2 * head_dim``
    per token (the Ca and Cb halves), while the softmax-gated combine produces
    ``head_dim``-wide compressed entries. The indexer mirrors this at
    ``index_head_dim``.

    Overlap state holds the previous window's Ca slice so the gating consumer
    can read it back on the next call. The slot starts as zeros — the consumer
    does an assignment (``new_kv[:, 0, :ratio] = prior_kv``) rather than an
    add, which matches the dynamic cache's first-call ``None``.
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
        """Swap the saved Ca slice with the most recent window's Ca slice.

        Returns the prior call's slice (zeros on first call) and stashes the
        current chunk's last-window Ca for the next call. Only ``:head_dim``
        is kept — Cb has already been folded into the previous compressed
        entry by the gating combine.
        """
        state = self.compressors[name]
        prior_kv = state.overlap_kv.clone()
        prior_gate = state.overlap_gate.clone()
        state.overlap_kv.copy_(chunk_kv[:, -1, :, :head_dim])
        state.overlap_gate.copy_(chunk_gate[:, -1, :, :head_dim])
        return prior_kv, prior_gate
