# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Grouping the MoE routing ids by expert with a counting sort.

The grouped experts path indexes five tensors off the top-k expert ids: the permutation that groups the
token-expert pairs by expert, the ids in that order, the per-expert offsets the grouped GEMM reads, the token
each grouped row came from, and the inverse permutation that puts the expert outputs back in token order.
`torch.sort` plus `torch.histc` plus `cumsum` produce the first three in 40 us per layer at decode shapes,
almost independently of size, because a general radix sort pays a fixed cost, and the caller builds the other
two with a division, an `arange` and a scatter. The ids are small integers in a known range, so counting them
and placing each one at its rank gives all five: 17 us inside a cuda graph at 1024 token-expert pairs, which is
~6% of a Qwen3-30B-A3B decode step at tp4.

Ids at or above `num_experts` are expert-parallel sentinels. They share one bucket past the last expert, so
they land at the tail in the order the sort-based path leaves them, `offsets` never reaches them and the grouped
GEMM skips their rows. Keeping them in the permutation is what makes it a permutation, so the inverse is total.
"""

import torch

from ..utils import is_triton_available


if is_triton_available():
    import triton
    import triton.language as tl


@triton.jit
def _count(ids_ptr, counts_ptr, S, E, BLOCK: tl.constexpr):
    """One atomic per pair: `counts[id] += 1`, with every sentinel id counted in bucket `E`."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = off < S
    ids = tl.load(ids_ptr + off, mask=live, other=E)
    tl.atomic_add(counts_ptr + tl.minimum(ids, E), 1, mask=live)


@triton.jit
def _scan(counts_ptr, starts_ptr, offs_ptr, E, BLOCK_E: tl.constexpr):
    """Exclusive scan of the counts into each bucket's start, and the inclusive scan the grouped GEMM reads.

    The scan spans the sentinel bucket so its rows start after the last expert's, but `offs` stops at the
    experts, which is how the GEMM comes to skip the sentinel tail.
    """
    e = tl.arange(0, BLOCK_E)
    c = tl.load(counts_ptr + e, mask=e <= E, other=0)
    incl = tl.cumsum(c, axis=0)
    tl.store(starts_ptr + e, incl - c, mask=e <= E)
    tl.store(offs_ptr + e, incl, mask=e < E)


@triton.jit
def _place(ids_ptr, starts_ptr, cursor_ptr, perm_ptr, ids_g_ptr, inv_ptr, rows_ptr, S, E, K, BLOCK: tl.constexpr):
    """One atomic per pair for its rank inside its bucket, then the five tensors that rank determines."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = off < S
    ids = tl.minimum(tl.load(ids_ptr + off, mask=live, other=E), E)
    rank = tl.atomic_add(cursor_ptr + ids, 1, mask=live)
    pos = tl.load(starts_ptr + ids, mask=live, other=0) + rank
    tl.store(perm_ptr + pos, off, mask=live)
    tl.store(ids_g_ptr + pos, ids, mask=live)
    tl.store(rows_ptr + pos, off // K, mask=live)
    tl.store(inv_ptr + off, pos, mask=live)


def counting_sort_route(expert_ids: torch.Tensor, num_experts: int, num_top_k: int):
    """`(ids_grouped, perm, offsets, rows, inv_perm)` for `expert_ids`, grouped by expert. CUDA only.

    `rows` is the token index of each grouped row, i.e. `perm // num_top_k`, and `inv_perm` inverts `perm`.
    Every output is written in full, so they are allocated uninitialized.
    """
    num_pairs = expert_ids.numel()
    device = expert_ids.device
    ids = expert_ids if expert_ids.dtype == torch.int32 else expert_ids.to(torch.int32)
    i32 = {"dtype": torch.int32, "device": device}
    # one bucket past the experts holds the sentinels, and doubles as the per-expert cursor in `_place`
    counts = torch.zeros(num_experts + 1, **i32)
    starts = torch.empty(num_experts + 1, **i32)
    offsets = torch.empty(num_experts, **i32)
    perm, ids_grouped, rows, inv_perm = (torch.empty(num_pairs, **i32) for _ in range(4))
    block = 256
    grid = (triton.cdiv(num_pairs, block),)
    _count[grid](ids, counts, num_pairs, num_experts, BLOCK=block)
    _scan[(1,)](counts, starts, offsets, num_experts, BLOCK_E=triton.next_power_of_2(num_experts + 1))
    counts.zero_()
    _place[grid](ids, starts, counts, perm, ids_grouped, inv_perm, rows, num_pairs, num_experts, num_top_k,
                 BLOCK=block)
    return ids_grouped, perm, offsets, rows, inv_perm
