"""Prototype (bench only, not for upstream as is): bf16 routed experts for decode-sized batches, dispatch fused.

`experts_implementation="proto_decode"`: four launches per MoE layer instead of the ~20 of `grouped_mm` (sort, gathers,
histogram, cumsum, two grouped GEMMs, activation, weights, scatter):

    route    (1 launch): counting sort of the S = tokens * top_k routed slots by expert (`_proto_fp8_experts`)
    gate_up  (E x I tiles): gather the expert's rows, gate and up GEMMs, SiLU(gate) * up, bf16 intermediate
    down     (E x H tiles): down GEMM, routing weight, scatter back to the flat (token, slot) row
    topk sum (1 launch): sum the top_k rows of each token

One program per (expert, N tile) walks that expert's routed rows, so at decode each weight tile is read once. The dots
take the weight tile as first operand (128 rows), which Triton lowers to a tensor-core MMA even with 16 routed rows.
Above `PROTO_DECODE_MAX_ROWS_PER_EXPERT` routed rows per expert on average (prefill), `grouped_mm` runs instead.
"""

import os

import torch
import triton
import triton.language as tl

from ._proto_fp8_experts import decode_route, decode_topk_sum


MAX_ROWS_PER_EXPERT = float(os.environ.get("PROTO_DECODE_MAX_ROWS_PER_EXPERT", "16"))


@triton.jit
def _gate_up_kernel(
    Hidden,  # (T, H) bf16
    Perm,  # (S,) int32: expert-sorted position -> flat (token, slot) index
    ExpertStart,  # (E + 1,) int32
    GateUp,  # (E, 2I, H): gate rows then up rows
    Inter,  # (S, I) bf16, expert-sorted
    stride_h,
    stride_gu_e,
    stride_gu_n,
    NUM_TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTER_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gate_ptrs = GateUp + e * stride_gu_e + offs_n[:, None] * stride_gu_n + offs_k[None, :]
    up_ptrs = gate_ptrs + INTER_DIM * stride_gu_n
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        token = (tl.load(Perm + rows, mask=row_mask, other=0) // NUM_TOP_K).to(tl.int64)
        a_ptrs = Hidden + token[None, :] * stride_h + offs_k[:, None]  # (BLOCK_K, BLOCK_M)
        acc_gate = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k0 in range(0, HIDDEN_DIM, BLOCK_K):
            a = tl.load(a_ptrs + k0, mask=row_mask[None, :], other=0.0)
            acc_gate = tl.dot(tl.load(gate_ptrs + k0), a, acc_gate)
            acc_up = tl.dot(tl.load(up_ptrs + k0), a, acc_up)
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up
        tl.store(
            Inter + rows[None, :] * INTER_DIM + offs_n[:, None], act.to(Inter.dtype.element_ty), mask=row_mask[None, :]
        )


@triton.jit
def _down_kernel(
    Inter,  # (S, I) bf16, expert-sorted
    Perm,  # (S,) int32
    ExpertStart,  # (E + 1,) int32
    Down,  # (E, H, I)
    Weights,  # (S,) routing weight per flat (token, slot)
    Out,  # (S, H) flat (token, slot) rows
    stride_d_e,
    stride_d_n,
    HIDDEN_DIM: tl.constexpr,
    INTER_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    w_ptrs = Down + e * stride_d_e + offs_n[:, None] * stride_d_n + offs_k[None, :]
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        a_ptrs = Inter + rows[None, :] * INTER_DIM + offs_k[:, None]  # (BLOCK_K, BLOCK_M)
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k0 in range(0, INTER_DIM, BLOCK_K):
            acc = tl.dot(tl.load(w_ptrs + k0), tl.load(a_ptrs + k0, mask=row_mask[None, :], other=0.0), acc)
        slot = tl.load(Perm + rows, mask=row_mask, other=0).to(tl.int64)
        weight = tl.load(Weights + slot, mask=row_mask, other=0.0).to(tl.float32)
        acc = acc * weight[None, :]
        tl.store(
            Out + slot[None, :] * HIDDEN_DIM + offs_n[:, None], acc.to(Out.dtype.element_ty), mask=row_mask[None, :]
        )


def _block_m(num_slots: int, num_experts: int) -> int:
    rows_per_expert = num_slots / num_experts
    return 16 if rows_per_expert <= 10 else 32 if rows_per_expert <= 24 else 64


def proto_decode_experts(hidden_states, top_k_index, top_k_weights, gate_up, down):
    """`(T, H)`: route, gate_up, down, top-k sum. `gate_up` is (E, 2I, H), `down` (E, H, I), both bf16 or fp16."""
    num_experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    num_top_k = top_k_index.shape[-1]
    num_slots = top_k_index.numel()
    top_k_index = top_k_index.contiguous()
    top_k_weights = top_k_weights.contiguous()
    perm, expert_start = decode_route(top_k_index, num_experts)
    block_m = _block_m(num_slots, num_experts)
    inter_out = torch.empty(num_slots, inter, dtype=hidden_states.dtype, device=hidden_states.device)
    _gate_up_kernel[(num_experts, inter // 128)](
        hidden_states, perm, expert_start, gate_up, inter_out,
        hidden_states.stride(0), gate_up.stride(0), gate_up.stride(1),
        NUM_TOP_K=num_top_k, HIDDEN_DIM=hidden, INTER_DIM=inter, BLOCK_M=block_m, BLOCK_N=128, BLOCK_K=64,
        num_warps=4, num_stages=4,
    )  # fmt: skip
    slots_out = torch.empty(num_slots, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
    _down_kernel[(num_experts, hidden // 128)](
        inter_out, perm, expert_start, down, top_k_weights, slots_out, down.stride(0), down.stride(1),
        HIDDEN_DIM=hidden, INTER_DIM=inter, BLOCK_M=block_m, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=4,
    )  # fmt: skip
    return decode_topk_sum(slots_out, top_k_index, num_experts)


def proto_decode_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """`experts_implementation="proto_decode"`: the fused path for decode-sized batches, `grouped_mm` above."""
    from .moe import _default_apply_gate, grouped_mm_experts_forward

    usable = (
        self.has_gate
        and self._apply_gate.__func__ is _default_apply_gate
        and type(self.act_fn).__name__ in ("SiLU", "SiLUActivation")
        and not self.has_post_expert_norm
        and not self.has_bias
        and not self.is_transposed
        and not self._is_expert_parallel
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and self.gate_up_proj.dtype == hidden_states.dtype
        and top_k_index.numel() <= MAX_ROWS_PER_EXPERT * self.num_experts
    )
    if not usable:
        return grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    return proto_decode_experts(hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj)
