"""Prototype (bench only, not for upstream as is): bf16 routed experts for decode-sized batches, dispatch fused.

`experts_implementation="proto_decode"`: four launches per MoE layer instead of the ~20 of `grouped_mm` (sort, gathers,
histogram, cumsum, two grouped GEMMs, activation, weights, scatter):

    route    (1 launch): counting sort of the S = tokens * top_k routed slots by expert (`_proto_fp8_experts`)
    gate_up  (E x I tiles): gather the expert's rows, gate and up GEMMs, SiLU(gate) * up, bf16 intermediate
    down     (E x H tiles): down GEMM, routing weight, scatter back to the flat (token, slot) row
    topk sum (1 launch): sum the top_k rows of each token

One program per (expert, N tile) walks that expert's routed rows, so at decode each weight tile is read once. The dots
take the weight tile as first operand (128 rows), which Triton lowers to a tensor-core MMA even with 16 routed rows.
Above `PROTO_DECODE_MAX_ROWS_PER_EXPERT` routed rows per expert on average (prefill), the block-aligned variant below
runs instead.
"""

import os

import torch
import triton
import triton.language as tl

from ._proto_fp8_experts import decode_route, decode_topk_sum


# The tile configs were picked on an H100 (228 KB of shared memory per block); RTX PRO 6000 and L40S have ~100 KB
_SHARED_MEMORY = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin


def _fit_stages(num_stages: int, block_m: int, block_n: int, block_k: int, num_b_tiles: int) -> int:
    """The most pipeline stages, up to `num_stages`, whose bf16 tiles fit in the shared memory of the device. Triton
    keeps `num_stages - 1` copies of the A tile and of the `num_b_tiles` B tiles."""
    tile_bytes = (block_m + num_b_tiles * block_n) * block_k * 2
    return max(1, min(num_stages, _SHARED_MEMORY // tile_bytes + 1))


MAX_ROWS_PER_EXPERT = float(os.environ.get("PROTO_DECODE_MAX_ROWS_PER_EXPERT", "0"))
# [scratch] number of real tokens of the step ([1] int32 on device), set by continuous batching when it pads the step:
# the slots of padding tokens go to no expert
VALID_TOKENS: torch.Tensor | None = None


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
    return 16 if rows_per_expert <= 10 else 32 if rows_per_expert <= 24 else 64 if rows_per_expert <= 48 else 128


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
        num_warps=4, num_stages=_fit_stages(4, block_m, 128, 64, 2),
    )  # fmt: skip
    slots_out = torch.empty(num_slots, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
    _down_kernel[(num_experts, hidden // 128)](
        inter_out, perm, expert_start, down, top_k_weights, slots_out, down.stride(0), down.stride(1),
        HIDDEN_DIM=hidden, INTER_DIM=inter, BLOCK_M=block_m, BLOCK_N=128, BLOCK_K=64, num_warps=4,
        num_stages=_fit_stages(4, block_m, 128, 64, 1),
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
    )
    if not usable:
        return grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    if top_k_index.numel() <= MAX_ROWS_PER_EXPERT * self.num_experts:
        return proto_decode_experts(hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj)
    return proto_aligned_experts(hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj)


# ---- block-aligned variant: each expert's routed rows padded to BLOCK_M, one program per (row block, N tile) -----------
#
# Same idea as vLLM's `moe_align_block_size` + `fused_moe`: the row blocks of all the experts form one grid dimension,
# so an expert with many rows gets many programs (no serial walk), which scales to prefill-sized steps.


@triton.jit
def _align_count_kernel(
    ExpertIds, Count, S, ValidTokens, NUM_EXPERTS: tl.constexpr, TOP_K: tl.constexpr, HAS_VALID: tl.constexpr,
    BLOCK: tl.constexpr,
):  # fmt: skip
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    num_slots = tl.minimum(S, tl.load(ValidTokens) * TOP_K) if HAS_VALID else S
    expert = tl.load(ExpertIds + slots, mask=slots < num_slots, other=NUM_EXPERTS).to(tl.int32)
    tl.atomic_add(Count + expert, 1, mask=(slots < num_slots) & (expert < NUM_EXPERTS))


@triton.jit
def _align_offsets_kernel(
    Count, Cursor, BlockEnd, NumBlocks, NUM_EXPERTS: tl.constexpr, PADDED_E: tl.constexpr, BLOCK_M: tl.constexpr
):
    # one program: padded start of every expert (in rows), its last row block + 1, the number of row blocks in use
    experts = tl.arange(0, PADDED_E)
    count = tl.load(Count + experts, mask=experts < NUM_EXPERTS, other=0)
    blocks = (count + BLOCK_M - 1) // BLOCK_M
    block_end = tl.cumsum(blocks, 0)
    tl.store(Cursor + experts, (block_end - blocks) * BLOCK_M, mask=experts < NUM_EXPERTS)
    tl.store(BlockEnd + experts, block_end, mask=experts < NUM_EXPERTS)
    tl.store(NumBlocks, tl.sum(blocks, 0))


@triton.jit
def _align_block_expert_kernel(
    BlockEnd, BlockExpert, MAX_BLOCKS, NUM_EXPERTS: tl.constexpr, PADDED_E: tl.constexpr, BLOCK: tl.constexpr
):
    # expert of row block b = number of experts whose blocks end at or before b
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    experts = tl.arange(0, PADDED_E)
    block_end = tl.load(BlockEnd + experts, mask=experts < NUM_EXPERTS, other=2147483647)
    e = tl.sum((block_end[None, :] <= b[:, None]).to(tl.int32), axis=1)
    tl.store(BlockExpert + b, e, mask=b < MAX_BLOCKS)


@triton.jit
def _align_scatter_kernel(
    ExpertIds, SortedSlots, Cursor, S, ValidTokens, NUM_EXPERTS: tl.constexpr, TOP_K: tl.constexpr,
    HAS_VALID: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    # each slot claims the next position of its expert; the order inside an expert is that of the race (rows are
    # computed independently, so no output depends on it)
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    num_slots = tl.minimum(S, tl.load(ValidTokens) * TOP_K) if HAS_VALID else S
    expert = tl.load(ExpertIds + slots, mask=slots < num_slots, other=NUM_EXPERTS).to(tl.int32)
    valid = (slots < num_slots) & (expert < NUM_EXPERTS)
    position = tl.atomic_add(Cursor + expert, 1, mask=valid)
    tl.store(SortedSlots + position, slots, mask=valid)


def align_route(top_k_index, num_experts, block_m):
    """`(sorted_slots, block_expert, num_blocks)`: slots grouped by expert, each expert padded to `block_m` rows with
    the sentinel S, the expert of each row block, and the number of row blocks in use (on device)."""
    num_slots = top_k_index.numel()
    device = top_k_index.device
    max_blocks = triton.cdiv(num_slots, block_m) + num_experts
    count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    cursor = torch.empty(num_experts, dtype=torch.int32, device=device)
    block_expert = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_blocks = torch.empty(1, dtype=torch.int32, device=device)
    sorted_slots = torch.full((max_blocks * block_m,), num_slots, dtype=torch.int32, device=device)
    grid = (triton.cdiv(num_slots, 256),)
    valid = VALID_TOKENS
    has_valid = valid is not None
    valid = valid if has_valid else count  # unused pointer
    top_k = top_k_index.shape[-1]
    _align_count_kernel[grid](
        top_k_index, count, num_slots, valid, NUM_EXPERTS=num_experts, TOP_K=top_k, HAS_VALID=has_valid, BLOCK=256,
        num_warps=4,
    )
    padded_e = triton.next_power_of_2(num_experts)
    block_end = torch.empty(num_experts, dtype=torch.int32, device=device)
    _align_offsets_kernel[(1,)](
        count, cursor, block_end, num_blocks, NUM_EXPERTS=num_experts, PADDED_E=padded_e, BLOCK_M=block_m, num_warps=4
    )
    _align_block_expert_kernel[(triton.cdiv(max_blocks, 32),)](
        block_end, block_expert, max_blocks, NUM_EXPERTS=num_experts, PADDED_E=padded_e, BLOCK=32, num_warps=4
    )
    _align_scatter_kernel[grid](
        top_k_index, sorted_slots, cursor, num_slots, valid, NUM_EXPERTS=num_experts, TOP_K=top_k,
        HAS_VALID=has_valid, BLOCK=256, num_warps=4,
    )
    return sorted_slots, block_expert, num_blocks


@triton.jit
def _aligned_gate_up_kernel(
    Hidden, SortedSlots, BlockExpert, NumBlocks, GateUp, Inter,
    stride_h, stride_gu_e, stride_gu_n, S,
    NUM_TOP_K: tl.constexpr, HIDDEN_DIM: tl.constexpr, INTER_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    if b >= tl.load(NumBlocks):
        return
    pid_n = tl.program_id(1)
    e = tl.load(BlockExpert + b).to(tl.int64)
    rows = b * BLOCK_M + tl.arange(0, BLOCK_M)
    slot = tl.load(SortedSlots + rows)
    row_mask = slot < S
    token = (slot // NUM_TOP_K).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = Hidden + token[:, None] * stride_h + offs_k[None, :]  # (BLOCK_M, BLOCK_K)
    gate_ptrs = GateUp + e * stride_gu_e + offs_n[None, :] * stride_gu_n + offs_k[:, None]  # (BLOCK_K, BLOCK_N)
    up_ptrs = gate_ptrs + INTER_DIM * stride_gu_n
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, HIDDEN_DIM, BLOCK_K):
        a = tl.load(a_ptrs + k0, mask=row_mask[:, None], other=0.0)
        acc_gate = tl.dot(a, tl.load(gate_ptrs + k0), acc_gate)
        acc_up = tl.dot(a, tl.load(up_ptrs + k0), acc_up)
    act = acc_gate * tl.sigmoid(acc_gate) * acc_up
    tl.store(Inter + rows[:, None].to(tl.int64) * INTER_DIM + offs_n[None, :], act.to(Inter.dtype.element_ty))


@triton.jit
def _aligned_down_kernel(
    Inter, SortedSlots, BlockExpert, NumBlocks, Down, Weights, Out,
    stride_d_e, stride_d_n, S,
    HIDDEN_DIM: tl.constexpr, INTER_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    if b >= tl.load(NumBlocks):
        return
    pid_n = tl.program_id(1)
    e = tl.load(BlockExpert + b).to(tl.int64)
    rows = b * BLOCK_M + tl.arange(0, BLOCK_M)
    slot = tl.load(SortedSlots + rows)
    row_mask = slot < S
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = Inter + rows[:, None].to(tl.int64) * INTER_DIM + offs_k[None, :]  # (BLOCK_M, BLOCK_K)
    w_ptrs = Down + e * stride_d_e + offs_n[None, :] * stride_d_n + offs_k[:, None]  # (BLOCK_K, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, INTER_DIM, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs + k0), tl.load(w_ptrs + k0), acc)
    weight = tl.load(Weights + slot, mask=row_mask, other=0.0).to(tl.float32)
    acc = acc * weight[:, None]
    tl.store(
        Out + slot[:, None].to(tl.int64) * HIDDEN_DIM + offs_n[None, :],
        acc.to(Out.dtype.element_ty),
        mask=row_mask[:, None],
    )


_ALIGNED_CFG = os.environ.get("PROTO_ALIGNED_CFG")  # "bm,bn,bk,warps,stages/bn,bk,warps,stages" to override


def _aligned_config(num_tokens: int, num_slots: int, num_experts: int):
    """`(gate_up config, down config)`, each (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages). By token count up to
    512 (vLLM's bf16 defaults on Hopper), then by mean routed rows per expert (picked on one H100 at the Qwen3-30B-A3B
    shapes, `bigserve/cbopt/moe/sweep_down.py`)."""
    if _ALIGNED_CFG:
        gate_up, down = _ALIGNED_CFG.split("/")
        gate_up = tuple(int(c) for c in gate_up.split(","))
        return gate_up, (gate_up[0], *(int(c) for c in down.split(",")))
    if num_tokens <= 32:
        return (16, 64, 128, 4, 4), (16, 64, 128, 4, 4)
    if num_tokens <= 96:
        return (32, 64, 128, 4, 3), (32, 64, 128, 4, 3)
    if num_slots / num_experts <= 96:
        return (64, 64, 64, 4, 3), (64, 128, 64, 4, 2)
    return (128, 128, 64, 8, 4), (128, 256, 64, 8, 4)


def proto_aligned_experts(hidden_states, top_k_index, top_k_weights, gate_up, down, config=None, down_config=None):
    num_experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    num_top_k = top_k_index.shape[-1]
    num_slots = top_k_index.numel()
    top_k_index = top_k_index.contiguous()
    top_k_weights = top_k_weights.contiguous()
    if config is None:
        config, down_config = _aligned_config(hidden_states.shape[0], num_slots, num_experts)
    block_m, block_n, block_k, num_warps, num_stages = config
    _, down_n, down_k, down_warps, down_stages = down_config or config
    sorted_slots, block_expert, num_blocks = align_route(top_k_index, num_experts, block_m)
    max_blocks = block_expert.numel()
    inter_out = torch.empty(max_blocks * block_m, inter, dtype=hidden_states.dtype, device=hidden_states.device)
    _aligned_gate_up_kernel[(max_blocks, inter // block_n)](
        hidden_states, sorted_slots, block_expert, num_blocks, gate_up, inter_out,
        hidden_states.stride(0), gate_up.stride(0), gate_up.stride(1), num_slots,
        NUM_TOP_K=num_top_k, HIDDEN_DIM=hidden, INTER_DIM=inter, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=_fit_stages(num_stages, block_m, block_n, block_k, 2),
    )  # fmt: skip
    slots_out = torch.empty(num_slots, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
    _aligned_down_kernel[(max_blocks, hidden // down_n)](
        inter_out, sorted_slots, block_expert, num_blocks, down, top_k_weights, slots_out,
        down.stride(0), down.stride(1), num_slots,
        HIDDEN_DIM=hidden, INTER_DIM=inter, BLOCK_M=block_m, BLOCK_N=down_n, BLOCK_K=down_k,
        num_warps=down_warps, num_stages=_fit_stages(down_stages, block_m, down_n, down_k, 1),
    )  # fmt: skip
    return decode_topk_sum(slots_out, top_k_index, num_experts)
