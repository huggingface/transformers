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

import os

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


# A softmax over the experts, then the top k of it, then the renormalisation, is three library calls that
# together take 0.77 ms of a 15.5 ms Qwen3-30B-A3B decode step at tp4 (`gatherTopK` 0.44, `bitonicSortKVInPlace`
# 0.23, softmax 0.10) to read 65 KB of logits. They are launch-bound, not bandwidth-bound: a token's whole
# expert row fits in one program's registers, so one kernel can do the softmax and k passes of argmax over it.
# vLLM does the same thing in one kernel (`topk_softmax`).


@triton.jit
def _softmax_topk_fwd(logits_ptr, vals_ptr, idx_ptr, T, E, K: tl.constexpr, NORM: tl.constexpr,
                      BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_E: tl.constexpr):
    """The k largest of `softmax(logits)` per token, and where they were, renormalised over the k if asked.

    Experts past `E` load as -inf, so they contribute nothing to the softmax sum and are never selected. Each
    pass takes the largest remaining probability and buries it, so the k indices come out distinct and in
    descending order, which is what `torch.topk` returns. The picks accumulate in a (tokens, k) tile because a
    triton loop can only walk a `range`.
    """
    off_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    live_t = off_t < T
    off_e = tl.arange(0, BLOCK_E)
    off_k = tl.arange(0, BLOCK_K)
    keep = live_t[:, None] & (off_e < E)[None, :]
    logits = tl.load(logits_ptr + off_t[:, None] * E + off_e[None, :], mask=keep, other=float("-inf"))
    logits = logits.to(tl.float32)
    probs = tl.exp(logits - tl.max(logits, axis=1)[:, None])
    probs = probs / tl.sum(probs, axis=1)[:, None]

    values = tl.zeros((BLOCK_T, BLOCK_K), tl.float32)
    wheres = tl.zeros((BLOCK_T, BLOCK_K), tl.int32)
    total = tl.zeros((BLOCK_T,), tl.float32)
    for k in range(K):
        at = tl.argmax(probs, axis=1)
        value = tl.max(probs, axis=1)
        is_k = off_k[None, :] == k
        values = tl.where(is_k, value[:, None], values)
        wheres = tl.where(is_k, at[:, None].to(tl.int32), wheres)
        total += value
        probs = tl.where(off_e[None, :] == at[:, None], float("-inf"), probs)

    if NORM:
        values = values / total[:, None]
    keep_k = live_t[:, None] & (off_k < K)[None, :]
    tl.store(vals_ptr + off_t[:, None] * K + off_k[None, :], values.to(vals_ptr.dtype.element_ty), mask=keep_k)
    tl.store(idx_ptr + off_t[:, None] * K + off_k[None, :], wheres.to(idx_ptr.dtype.element_ty), mask=keep_k)


@triton.jit
def _softmax_topk_bwd(logits_ptr, idx_ptr, dvals_ptr, dlogits_ptr, T, E, K: tl.constexpr, NORM: tl.constexpr,
                      BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr):
    """The gradient of the above, from the saved logits and indices.

    Selecting is a gather, so its gradient scatters back to the k positions and the softmax Jacobian carries it
    to every expert: `d_logits = p * (d_p - sum(p * d_p))`. Renormalising adds one term shared by all k,
    `d_v_i = d_w_i / total - sum_j(d_w_j v_j) / total**2`, so the k positions are walked twice: once for the
    two sums, once to scatter.
    """
    off_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    live_t = off_t < T
    off_e = tl.arange(0, BLOCK_E)
    keep = live_t[:, None] & (off_e < E)[None, :]
    logits = tl.load(logits_ptr + off_t[:, None] * E + off_e[None, :], mask=keep, other=float("-inf"))
    logits = logits.to(tl.float32)
    probs = tl.exp(logits - tl.max(logits, axis=1)[:, None])
    probs = probs / tl.sum(probs, axis=1)[:, None]

    total = tl.zeros((BLOCK_T,), tl.float32)
    dotted = tl.zeros((BLOCK_T,), tl.float32)
    for k in range(K):
        at = tl.load(idx_ptr + off_t * K + k, mask=live_t, other=0)
        d_out = tl.load(dvals_ptr + off_t * K + k, mask=live_t, other=0.0).to(tl.float32)
        value = tl.sum(tl.where(off_e[None, :] == at[:, None], probs, 0.0), axis=1)
        total += value
        dotted += d_out * value

    d_probs = tl.zeros((BLOCK_T, BLOCK_E), tl.float32)
    for k in range(K):
        at = tl.load(idx_ptr + off_t * K + k, mask=live_t, other=0)
        d_out = tl.load(dvals_ptr + off_t * K + k, mask=live_t, other=0.0).to(tl.float32)
        d_value = d_out / total - dotted / (total * total) if NORM else d_out
        d_probs += tl.where(off_e[None, :] == at[:, None], d_value[:, None], 0.0)

    d_logits = probs * (d_probs - tl.sum(probs * d_probs, axis=1)[:, None])
    tl.store(dlogits_ptr + off_t[:, None] * E + off_e[None, :], d_logits.to(dlogits_ptr.dtype.element_ty),
             mask=keep)


class _SoftmaxTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, top_k, norm):
        num_tokens, num_experts = logits.shape
        vals = torch.empty(num_tokens, top_k, dtype=logits.dtype, device=logits.device)
        idx = torch.empty(num_tokens, top_k, dtype=torch.int64, device=logits.device)
        block_t, block_e = 4, triton.next_power_of_2(num_experts)
        _softmax_topk_fwd[(triton.cdiv(num_tokens, block_t),)](
            logits, vals, idx, num_tokens, num_experts, K=top_k, NORM=norm, BLOCK_T=block_t,
            BLOCK_K=triton.next_power_of_2(top_k), BLOCK_E=block_e
        )
        ctx.save_for_backward(logits, idx)
        ctx.top_k, ctx.norm = top_k, norm
        return vals, idx

    @staticmethod
    def backward(ctx, grad_vals, _grad_idx):
        logits, idx = ctx.saved_tensors
        num_tokens, num_experts = logits.shape
        d_logits = torch.empty_like(logits)
        block_t, block_e = 4, triton.next_power_of_2(num_experts)
        _softmax_topk_bwd[(triton.cdiv(num_tokens, block_t),)](
            logits, idx, grad_vals.contiguous(), d_logits, num_tokens, num_experts,
            K=ctx.top_k, NORM=ctx.norm, BLOCK_T=block_t, BLOCK_E=block_e
        )
        return d_logits, None, None


def fused_softmax_topk_available(logits: torch.Tensor) -> bool:
    """Whether to route this router through the fused kernel. Opt-in, CUDA and triton only."""
    return (
        os.environ.get("HF_MOE_FUSED_ROUTER") == "1"
        and is_triton_available()
        and logits.is_cuda
        and logits.dim() == 2
        and logits.stride(1) == 1
    )


def fused_softmax_topk(logits: torch.Tensor, top_k: int, norm: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """`topk(softmax(logits, dtype=float32), top_k)`, renormalised over the k when `norm`, in one kernel.

    The values come back in the logits' dtype, as the routers cast them.
    """
    return _SoftmaxTopK.apply(logits, top_k, norm)
