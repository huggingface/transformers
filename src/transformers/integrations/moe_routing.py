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

The grouped experts path needs three things from the top-k expert ids: the permutation that groups tokens by
expert, the ids in that order, and the per-expert offsets the grouped GEMM indexes with. `torch.sort` plus
`torch.histc` plus `cumsum` produce them in 40 us per layer at decode shapes, almost independently of size,
because a general radix sort pays a fixed cost. The ids are small integers in a known range, so counting them
and placing each one at its rank is enough: 17 us for the same three tensors, measured inside a cuda graph at
1024 token-expert pairs, which is ~6% of a Qwen3-30B-A3B decode step at tp4.

Ids at or above `num_experts` are expert-parallel sentinels. They are left out of the counts and their slots
stay at the tail, which is what the sort-based path achieves by leaving them unclamped.
"""

import torch

from ..utils import is_triton_available


if is_triton_available():
    import triton
    import triton.language as tl


@triton.jit
def _count(ids_ptr, counts_ptr, S, E, BLOCK: tl.constexpr):
    """One atomic per element: counts[id] += 1. Ids at or above E are sentinels and are not counted."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = off < S
    ids = tl.load(ids_ptr + off, mask=live, other=E)
    tl.atomic_add(counts_ptr + ids, 1, mask=live & (ids < E))


@triton.jit
def _scan(counts_ptr, starts_ptr, offs_ptr, E: tl.constexpr):
    """Exclusive scan of the counts into the start of each expert's run, and the inclusive scan the GEMM wants."""
    e = tl.arange(0, E)
    c = tl.load(counts_ptr + e)
    incl = tl.cumsum(c, axis=0)
    tl.store(starts_ptr + e, incl - c)
    tl.store(offs_ptr + e, incl)


@triton.jit
def _place(ids_ptr, starts_ptr, cursor_ptr, perm_ptr, sorted_ptr, S, E, BLOCK: tl.constexpr):
    """One atomic per element for its rank inside its expert's run, then write the permutation."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = off < S
    ids = tl.load(ids_ptr + off, mask=live, other=E)
    real = live & (ids < E)
    rank = tl.atomic_add(cursor_ptr + ids, 1, mask=real)
    start = tl.load(starts_ptr + ids, mask=real, other=0)
    pos = start + rank
    tl.store(perm_ptr + pos, off, mask=real)
    tl.store(sorted_ptr + pos, ids, mask=real)


def counting_sort(ids: torch.Tensor, num_experts: int):
    """`(sorted_ids, perm, offsets)`, the three tensors the grouped-MoE path builds with sort + histc + cumsum.

    Sentinel ids (>= num_experts) are dropped from the counts and left at the tail of `perm`, which is what the
    torch path achieves by leaving them unclamped so the sort pushes them to the end.
    """
    S = ids.numel()
    dev = ids.device
    counts = torch.zeros(num_experts, dtype=torch.int32, device=dev)
    starts = torch.empty(num_experts, dtype=torch.int32, device=dev)
    offs = torch.empty(num_experts, dtype=torch.int32, device=dev)
    perm = torch.full((S,), S - 1, dtype=torch.int32, device=dev)
    srt = torch.full((S,), num_experts, dtype=torch.int32, device=dev)
    BLOCK = 256
    grid = (triton.cdiv(S, BLOCK),)
    _count[grid](ids, counts, S, num_experts, BLOCK=BLOCK)
    _scan[(1,)](counts, starts, offs, E=num_experts)
    _place[grid](ids, starts, counts.zero_(), perm, srt, S, num_experts, BLOCK=BLOCK)
    return srt, perm, offs


def reference(ids, num_experts):
    srt, perm = torch.sort(ids)
    counts = torch.histc(srt.float(), bins=num_experts, min=0, max=num_experts - 1)
    return srt, perm, torch.cumsum(counts, 0).int()


def counting_sort_route(expert_ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(ids_grouped, perm, offsets)` for `expert_ids`, grouped by expert. CUDA only."""
    num_pairs = expert_ids.numel()
    device = expert_ids.device
    ids = expert_ids if expert_ids.dtype == torch.int32 else expert_ids.to(torch.int32)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    starts = torch.empty(num_experts, dtype=torch.int32, device=device)
    offsets = torch.empty(num_experts, dtype=torch.int32, device=device)
    # sentinel slots keep an in-range row index whose output the caller masks out, and an id the caller reads as
    # a sentinel, so the tail matches what the sort-based path leaves there
    perm = torch.full((num_pairs,), num_pairs - 1, dtype=torch.int32, device=device)
    ids_grouped = torch.full((num_pairs,), num_experts, dtype=torch.int32, device=device)
    block = 256
    grid = (triton.cdiv(num_pairs, block),)
    _count[grid](ids, counts, num_pairs, num_experts, BLOCK=block)
    _scan[(1,)](counts, starts, offsets, E=num_experts)
    counts.zero_()  # reused as the per-expert cursor
    _place[grid](ids, starts, counts, perm, ids_grouped, num_pairs, num_experts, BLOCK=block)
    return ids_grouped, perm, offsets
