"""Prototype (bench only, not for upstream as is) routed-experts forwards for block-FP8 MoE at inference.

Two implementations of `FP8Experts.forward(hidden_states, top_k_index, top_k_weights)`, both off by default:

- `PROTO_EXPERTS_TRTLLM=1`: flashinfer's TRT-LLM routed FP8 MoE, called the way vLLM's `TrtLlmFp8ExpertsModular` does.
  The weights are re-laid once at load (`trtllm_prepare`, bytes only moved), the originals are kept. Blackwell only.
- `PROTO_EXPERTS_DECODE=1`: a Triton kernel pair with the dispatch fused and a decode schedule: one program per
  (expert, N tile) walking that expert's routed rows. Candidate for `kernels-community/finegrained-fp8`.

This file imports only torch and triton so that the benchmarks can load it without transformers.
"""

import os

import torch
import triton
import triton.language as tl


FP8 = torch.float8_e4m3fn
PROTO_EXPERTS_TRTLLM = os.environ.get("PROTO_EXPERTS_TRTLLM", "0") == "1"
PROTO_EXPERTS_DECODE = os.environ.get("PROTO_EXPERTS_DECODE", "0") == "1"
CALLS = {"trtllm": 0, "decode": 0}  # host-side launches of each forward (a CUDA graph replay does not count)


# ---- shared: per-token, per-128-group activation quant -----------------------------------------------------------------


@triton.jit
def _act_quant_kernel(X, Y, S, stride_x, stride_y, stride_s_m, stride_s_g, GROUP: tl.constexpr):
    # one program = one row x one 128-wide group. Same recipe as vLLM's `per_token_group_quant_fp8`: amax floored at
    # 1e-10, scale = amax / 448, values clamped to the e4m3 range
    m = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    cols = g * GROUP + tl.arange(0, GROUP)
    x = tl.load(X + m * stride_x + cols).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)), 1e-10) / 448.0
    y = tl.minimum(tl.maximum(x / s, -448.0), 448.0)
    tl.store(Y + m * stride_y + cols, y.to(Y.dtype.element_ty))
    tl.store(S + m * stride_s_m + g * stride_s_g, s)


def act_quant(x: torch.Tensor, scales_group_major: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] -> (e4m3 [M, K], fp32 scales [M, K / 128], or [K / 128, M] with `scales_group_major`). One launch."""
    assert x.dim() == 2 and x.stride(1) == 1 and x.shape[1] % 128 == 0
    m, k = x.shape
    y = torch.empty(m, k, dtype=FP8, device=x.device)
    s = torch.empty((k // 128, m) if scales_group_major else (m, k // 128), dtype=torch.float32, device=x.device)
    stride_s_m, stride_s_g = (1, m) if scales_group_major else (k // 128, 1)
    _act_quant_kernel[(m, k // 128)](x, y, s, x.stride(0), k, stride_s_m, stride_s_g, GROUP=128, num_warps=1)
    return y, s


# ---- track 1: flashinfer TRT-LLM routed FP8 MoE ------------------------------------------------------------------------

_TRTLLM_MIN_BLOCK_SCALE = 1e-10  # vLLM clamps the block scales of dead experts (`prepare_fp8_moe_layer_for_fi`)


def trtllm_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 10


def trtllm_prepare(gate_up, gate_up_scale, down, down_scale):
    """Re-lay one (rank's) stack of block-FP8 expert weights for the TRT-LLM kernels, as vLLM does at load:
    `[up; gate]` row order, rows shuffled for the kernel's 64-row epilogue tile, K blocked by 128
    (`[E, K / 128, rows, 128]`, "BlockMajorK"). The fp8 bytes are only moved, the fp32 block scales only reordered.
    """
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    num_experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    assert gate_up.dtype == FP8 and down.dtype == FP8, (gate_up.dtype, down.dtype)
    assert gate_up_scale.dtype == torch.float32 and down_scale.dtype == torch.float32
    assert hidden % 128 == 0 and inter % 128 == 0, (hidden, inter)
    assert tuple(down.shape) == (num_experts, hidden, inter), (down.shape, gate_up.shape)
    assert tuple(gate_up_scale.shape) == (num_experts, two_i // 128, hidden // 128), gate_up_scale.shape
    assert tuple(down_scale.shape) == (num_experts, hidden // 128, inter // 128), down_scale.shape

    def block_major_k(weight, swap_halves):
        e, m, k = weight.shape
        # rows[new_row] = old_row; the helper only reads the shape
        rows = get_shuffle_matrix_a_row_indices(torch.empty(m, 1, dtype=torch.uint8), 64).to(weight.device)
        if swap_halves:
            rows = (rows + m // 2) % m
        assert rows.min().item() == 0 and rows.max().item() == m - 1 and rows.unique().numel() == m
        blocked = weight.view(torch.uint8).view(e, m, k // 128, 128)
        return blocked.permute(0, 2, 1, 3)[:, :, rows, :].contiguous().view(FP8)

    n_gate = inter // 128
    w13 = block_major_k(gate_up, swap_halves=True)
    s13 = torch.cat([gate_up_scale[:, n_gate:], gate_up_scale[:, :n_gate]], dim=1).clamp(min=_TRTLLM_MIN_BLOCK_SCALE)
    w2 = block_major_k(down, swap_halves=False)
    s2 = down_scale.clamp(min=_TRTLLM_MIN_BLOCK_SCALE)
    assert tuple(w13.shape) == (num_experts, hidden // 128, two_i, 128)
    assert tuple(w2.shape) == (num_experts, inter // 128, hidden, 128)
    return w13, s13.contiguous(), w2, s2.contiguous()


def trtllm_experts_forward(prepared, hidden_states, top_k_index, top_k_weights):
    """Activation quant (1 launch), then routing, gemm1, activation, gemm2 and finalize inside flashinfer."""
    import flashinfer
    from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

    CALLS["trtllm"] += 1
    w13, s13, w2, s2 = prepared
    num_experts, _, two_i, _ = w13.shape
    num_tokens, hidden = hidden_states.shape
    x_q, x_s = act_quant(hidden_states, scales_group_major=True)
    out = torch.empty(num_tokens, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
    flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe(
        topk_ids=(top_k_index.to(torch.int32), top_k_weights.to(torch.float32)),
        routing_bias=None,
        hidden_states=x_q,
        hidden_states_scale=x_s,
        gemm1_weights=w13,
        gemm1_weights_scale=s13,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        num_experts=num_experts,
        top_k=top_k_index.shape[-1],
        n_group=None,
        topk_group=None,
        intermediate_size=two_i // 2,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # not used with precomputed routing
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        do_finalize=True,
        output=out,
        tune_max_num_tokens=8192,
    )
    return out


# ---- track 2: Triton kernel pair, dispatch fused, one program per (expert, N tile) -------------------------------------
#
#   route    (1 launch): counting sort of the S = tokens * top_k routed slots by expert, without atomics
#   gate_up  (E x I tiles): gather the expert's rows, quantize them, gate and up GEMMs, SiLU, requant to fp8
#   down     (E x H tiles): down GEMM on the fp8 intermediate, routing weight, scatter back to (token, slot) rows
#   topk sum (1 launch): sum the top_k rows of each token
#
# A program whose expert has no row returns before touching the weights, and an expert with at most BLOCK_M rows has
# each of its weight tiles read exactly once. Above BLOCK_M rows the program walks the expert's rows in tiles and
# reads the weights again: the configs below keep that rare for the row counts decoding produces.

_ROUTE_BLOCK = 16
_ROUTE_CHUNK = 256


@triton.jit
def _route_kernel(ExpertIds, Perm, ExpertStart, S, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    """Stable counting sort by comparison, O(S^2 / programs): fine for the few thousand slots of a decode step.

    Programs [0, cdiv(S, BLOCK)) place BLOCK slots each: `Perm[rank] = slot` with rank = #(ids smaller) + #(equal ids
    in an earlier slot). The programs after them write `ExpertStart[e] = #(ids < e)` for e in [0, NUM_EXPERTS].
    Slots of non-local experts (id >= NUM_EXPERTS, expert parallel) sort to the tail, past `ExpertStart[NUM_EXPERTS]`.
    """
    pid = tl.program_id(0)
    num_slot_programs = tl.cdiv(S, BLOCK)
    if pid < num_slot_programs:
        slots = pid * BLOCK + tl.arange(0, BLOCK)
        mine = tl.load(ExpertIds + slots, mask=slots < S, other=0).to(tl.int32)
        rank = tl.zeros((BLOCK,), dtype=tl.int32)
        for c in range(0, S, CHUNK):
            others_slot = c + tl.arange(0, CHUNK)
            others = tl.load(ExpertIds + others_slot, mask=others_slot < S, other=2147483647).to(tl.int32)
            before = (others[None, :] < mine[:, None]) | (
                (others[None, :] == mine[:, None]) & (others_slot[None, :] < slots[:, None])
            )
            rank += tl.sum(before.to(tl.int32), axis=1)
        tl.store(Perm + rank, slots, mask=slots < S)
    else:
        experts = (pid - num_slot_programs) * BLOCK + tl.arange(0, BLOCK)
        start = tl.zeros((BLOCK,), dtype=tl.int32)
        for c in range(0, S, CHUNK):
            others_slot = c + tl.arange(0, CHUNK)
            others = tl.load(ExpertIds + others_slot, mask=others_slot < S, other=2147483647).to(tl.int32)
            start += tl.sum((others[None, :] < experts[:, None]).to(tl.int32), axis=1)
        tl.store(ExpertStart + experts, start, mask=experts <= NUM_EXPERTS)


# fp16 operands of the FOLD variants carry activation * scales: this power of two keeps them inside fp16's normal range
# (products of an activation by a weight block scale sit around 1e-4), and is divided out exactly after the loop
_FOLD_SHIFT = 1024.0


@triton.jit
def _decode_gate_up_kernel(
    Hidden,  # (T, H) activations, bf16 (or e4m3 with PREQUANT)
    HiddenScale,  # (T, H / 128) fp32, only read with PREQUANT
    Perm,  # (S,) int32, expert-sorted position -> flat (token, slot) index
    ExpertStart,  # (E + 1,) int32
    GateUp,  # (E, 2I, H) e4m3
    GateUpScale,  # (E, 2I / 128, H / 128) fp32
    Inter,  # (S, I) e4m3, expert-sorted
    InterScale,  # (S, I / 128) fp32
    stride_h,
    stride_hs,
    stride_gu_e,
    stride_gu_n,
    stride_gus_e,
    stride_gus_n,
    NUM_TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # multiple of 128 dividing INTERMEDIATE_DIM
    PREQUANT: tl.constexpr,
    FOLD: tl.constexpr,  # needs BLOCK_N == 128
    SHIFT: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_k = tl.arange(0, 128)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_base = GateUp + e * stride_gu_e + offs_k[:, None] + offs_n[None, :] * stride_gu_n
    up_base = gate_base + INTERMEDIATE_DIM * stride_gu_n
    if BLOCK_N == 128:
        # one weight scale per (tile, K block)
        gate_s_base = GateUpScale + e * stride_gus_e + pid_n * stride_gus_n
    else:
        gate_s_base = GateUpScale + e * stride_gus_e + (offs_n // 128) * stride_gus_n
    up_s_base = gate_s_base + (INTERMEDIATE_DIM // 128) * stride_gus_n
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        token = (tl.load(Perm + rows, mask=row_mask, other=0) // NUM_TOP_K).to(tl.int64)
        a_base = Hidden + token[:, None] * stride_h + offs_k[None, :]
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kb in range(0, HIDDEN_DIM // 128):
            if PREQUANT:
                a = tl.load(a_base + kb * 128, mask=row_mask[:, None], other=0.0)
                a_s = tl.load(HiddenScale + token * stride_hs + kb, mask=row_mask, other=0.0)
            else:
                a_raw = tl.load(a_base + kb * 128, mask=row_mask[:, None], other=0.0).to(tl.float32)
                a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
                a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)
            w_gate = tl.load(gate_base + kb * 128)
            w_up = tl.load(up_base + kb * 128)
            s_gate = tl.load(gate_s_base + kb)
            s_up = tl.load(up_s_base + kb)
            if FOLD:
                # The dot runs on fp16 operands either way (fp8 is widened for the MMA): give it the dequantized
                # activation times the weight block scale, and let the MMA accumulate over K. No fp32 multiply-add
                # per output element and K block; the fp16 rounding of the operand is 2^-11 relative.
                a_f = a.to(tl.float32) * a_s[:, None]
                acc_gate = tl.dot((a_f * (s_gate * SHIFT)).to(tl.float16), w_gate.to(tl.float16), acc_gate)
                acc_up = tl.dot((a_f * (s_up * SHIFT)).to(tl.float16), w_up.to(tl.float16), acc_up)
            elif BLOCK_N == 128:
                acc_gate += tl.dot(a, w_gate) * (a_s * s_gate)[:, None]
                acc_up += tl.dot(a, w_up) * (a_s * s_up)[:, None]
            else:
                acc_gate += tl.dot(a, w_gate) * a_s[:, None] * s_gate[None, :]
                acc_up += tl.dot(a, w_up) * a_s[:, None] * s_up[None, :]
        if FOLD:
            acc_gate = acc_gate * (1.0 / SHIFT)
            acc_up = acc_up * (1.0 / SHIFT)
        # SiLU(gate) * up in fp32, then one fp8 requant per (row, 128-wide block)
        act = tl.reshape(acc_gate * tl.sigmoid(acc_gate) * acc_up, (BLOCK_M, BLOCK_N // 128, 128))
        act_s = tl.max(tl.abs(act), axis=2) / 448.0
        act_q = tl.reshape(act / tl.maximum(act_s[:, :, None], 1e-12), (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)
        tl.store(Inter + rows[:, None] * INTERMEDIATE_DIM + offs_n[None, :], act_q, mask=row_mask[:, None])
        blocks = pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)
        tl.store(
            InterScale + rows[:, None] * (INTERMEDIATE_DIM // 128) + blocks[None, :], act_s, mask=row_mask[:, None]
        )


@triton.jit
def _decode_down_kernel(
    Inter,  # (S, I) e4m3, expert-sorted
    InterScale,  # (S, I / 128) fp32
    Perm,  # (S,) int32
    ExpertStart,  # (E + 1,) int32
    Down,  # (E, H, I) e4m3
    DownScale,  # (E, H / 128, I / 128) fp32
    Weights,  # (S,) routing weight per flat (token, slot)
    Out,  # (S, H) flat (token, slot) rows
    stride_d_e,
    stride_d_n,
    stride_ds_e,
    stride_ds_n,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # multiple of 128 dividing HIDDEN_DIM
    FOLD: tl.constexpr,  # needs BLOCK_N == 128
    SHIFT: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_k = tl.arange(0, 128)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    w_base = Down + e * stride_d_e + offs_k[:, None] + offs_n[None, :] * stride_d_n
    if BLOCK_N == 128:
        w_s_base = DownScale + e * stride_ds_e + pid_n * stride_ds_n
    else:
        w_s_base = DownScale + e * stride_ds_e + (offs_n // 128) * stride_ds_n
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kb in range(0, INTERMEDIATE_DIM // 128):
            a = tl.load(
                Inter + rows[:, None] * INTERMEDIATE_DIM + (kb * 128 + offs_k)[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            a_s = tl.load(InterScale + rows * (INTERMEDIATE_DIM // 128) + kb, mask=row_mask, other=0.0)
            w = tl.load(w_base + kb * 128)
            w_s = tl.load(w_s_base + kb)
            if FOLD:
                a_f = (a.to(tl.float32) * (a_s * (w_s * SHIFT))[:, None]).to(tl.float16)
                acc = tl.dot(a_f, w.to(tl.float16), acc)
            elif BLOCK_N == 128:
                acc += tl.dot(a, w) * (a_s * w_s)[:, None]
            else:
                acc += tl.dot(a, w) * a_s[:, None] * w_s[None, :]
        # routing weight and scatter back to the flat (token, slot) row: the top-k sum is then a plain reduction
        slot = tl.load(Perm + rows, mask=row_mask, other=0).to(tl.int64)
        weight = tl.load(Weights + slot, mask=row_mask, other=0.0).to(tl.float32)
        if FOLD:
            weight = weight * (1.0 / SHIFT)
        acc = acc * weight[:, None]
        tl.store(
            Out + slot[:, None] * HIDDEN_DIM + offs_n[None, :], acc.to(Out.dtype.element_ty), mask=row_mask[:, None]
        )


@triton.jit
def _topk_sum_kernel(
    Slots, ExpertIds, Out, H, NUM_TOP_K: tl.constexpr, NUM_EXPERTS: tl.constexpr, BLOCK_H: tl.constexpr
):
    t = tl.program_id(0).to(tl.int64)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(NUM_TOP_K):
        # rows of non-local experts (expert parallel) are never written by the down kernel
        local = tl.load(ExpertIds + t * NUM_TOP_K + k) < NUM_EXPERTS
        acc += tl.load(Slots + (t * NUM_TOP_K + k) * H + offs_h, mask=mask & local, other=0.0).to(tl.float32)
    tl.store(Out + t * H + offs_h, acc.to(Out.dtype.element_ty), mask=mask)


# Fixed configs, chosen offline (gap2/experts REPORT): no in-band autotune. Keyed on the average number of routed rows
# per expert, which is known from the shapes alone: (max rows per expert, gate_up config, down config), each config
# being (BLOCK_M, BLOCK_N, num_warps, num_stages).
DECODE_CONFIGS = [
    (4, (16, 128, 4, 3), (16, 128, 4, 3)),
    (16, (16, 128, 4, 3), (16, 128, 4, 3)),
    (48, (32, 128, 8, 3), (32, 128, 8, 3)),
]
DECODE_MAX_SLOTS = 16384  # the routing kernel is quadratic in the number of slots


def decode_config(num_slots: int, num_experts: int):
    """The config for this many routed slots, or `None` when the decode schedule should not be used."""
    if num_slots > DECODE_MAX_SLOTS:
        return None
    for max_rows, gate_up_config, down_config in DECODE_CONFIGS:
        if num_slots <= max_rows * num_experts:
            return gate_up_config, down_config
    return None


def decode_route(top_k_index, num_experts):
    """`(perm, expert_start)`: expert-sorted position -> flat (token, slot) index, and each expert's first position."""
    num_slots = top_k_index.numel()
    perm = torch.empty(num_slots, dtype=torch.int32, device=top_k_index.device)
    expert_start = torch.empty(num_experts + 1, dtype=torch.int32, device=top_k_index.device)
    _route_kernel[(triton.cdiv(num_slots, _ROUTE_BLOCK) + triton.cdiv(num_experts + 1, _ROUTE_BLOCK),)](
        top_k_index, perm, expert_start, num_slots,
        NUM_EXPERTS=num_experts, BLOCK=_ROUTE_BLOCK, CHUNK=_ROUTE_CHUNK, num_warps=4,
    )  # fmt: skip
    return perm, expert_start


def decode_gate_up(hidden_states, perm, expert_start, gate_up, gate_up_scale, num_top_k, config, prequant=False):
    """`(inter, inter_scale)`: SiLU(gate) * up of every routed row, e4m3 with one fp32 scale per 128 columns."""
    block_m, block_n, num_warps, num_stages, fold = (*config, False)[:5]
    num_experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    num_slots = perm.numel()
    device = hidden_states.device
    assert hidden % 128 == 0 and inter % block_n == 0 and block_n % 128 == 0 and (block_n == 128 or not fold)
    inter_q = torch.empty(num_slots, inter, dtype=FP8, device=device)
    inter_s = torch.empty(num_slots, inter // 128, dtype=torch.float32, device=device)
    if prequant:
        x, x_s = act_quant(hidden_states)
        stride_x, stride_xs = hidden, hidden // 128
    else:
        x, x_s = hidden_states, hidden_states
        stride_x, stride_xs = hidden_states.stride(0), 0
    _decode_gate_up_kernel[(num_experts, inter // block_n)](
        x, x_s, perm, expert_start, gate_up, gate_up_scale, inter_q, inter_s,
        stride_x, stride_xs, gate_up.stride(0), gate_up.stride(1), gate_up_scale.stride(0), gate_up_scale.stride(1),
        NUM_TOP_K=num_top_k, HIDDEN_DIM=hidden, INTERMEDIATE_DIM=inter, BLOCK_M=block_m, BLOCK_N=block_n,
        PREQUANT=prequant, FOLD=fold, SHIFT=_FOLD_SHIFT, num_warps=num_warps, num_stages=num_stages,
    )  # fmt: skip
    return inter_q, inter_s


def decode_down(inter_q, inter_s, perm, expert_start, down, down_scale, top_k_weights, out_dtype, config):
    """`(S, hidden)`: every routed row's weighted expert output, in flat (token, slot) order."""
    block_m, block_n, num_warps, num_stages, fold = (*config, False)[:5]
    num_experts, hidden, inter = down.shape
    assert hidden % block_n == 0 and block_n % 128 == 0 and inter % 128 == 0 and (block_n == 128 or not fold)
    slots_out = torch.empty(perm.numel(), hidden, dtype=out_dtype, device=down.device)
    _decode_down_kernel[(num_experts, hidden // block_n)](
        inter_q, inter_s, perm, expert_start, down, down_scale, top_k_weights, slots_out,
        down.stride(0), down.stride(1), down_scale.stride(0), down_scale.stride(1),
        HIDDEN_DIM=hidden, INTERMEDIATE_DIM=inter, BLOCK_M=block_m, BLOCK_N=block_n,
        FOLD=fold, SHIFT=_FOLD_SHIFT, num_warps=num_warps, num_stages=num_stages,
    )  # fmt: skip
    return slots_out


def decode_topk_sum(slots_out, top_k_index, num_experts):
    num_tokens, num_top_k = top_k_index.shape
    hidden = slots_out.shape[1]
    out = torch.empty(num_tokens, hidden, dtype=slots_out.dtype, device=slots_out.device)
    _topk_sum_kernel[(num_tokens, triton.cdiv(hidden, 512))](
        slots_out, top_k_index, out, hidden,
        NUM_TOP_K=num_top_k, NUM_EXPERTS=num_experts, BLOCK_H=512, num_warps=4,
    )  # fmt: skip
    return out


def decode_experts_forward(
    hidden_states,
    top_k_index,
    top_k_weights,
    gate_up,
    gate_up_scale,
    down,
    down_scale,
    gate_up_config=None,
    down_config=None,
    prequant: bool = False,
):
    """Block-FP8 routed experts in 4 launches (5 with `prequant`). Returns `(num_tokens, hidden)` in the input dtype."""
    CALLS["decode"] += 1
    num_experts = gate_up.shape[0]
    num_top_k = top_k_index.shape[-1]
    if gate_up_config is None:
        gate_up_config, down_config = decode_config(top_k_index.numel(), num_experts)
    assert hidden_states.stride(1) == 1 and top_k_index.is_contiguous() and top_k_weights.is_contiguous()
    assert gate_up.stride(2) == 1 and down.stride(2) == 1 and gate_up_scale.stride(2) == 1 and down_scale.stride(2) == 1
    perm, expert_start = decode_route(top_k_index, num_experts)
    inter_q, inter_s = decode_gate_up(
        hidden_states, perm, expert_start, gate_up, gate_up_scale, num_top_k, gate_up_config, prequant
    )
    slots_out = decode_down(
        inter_q, inter_s, perm, expert_start, down, down_scale, top_k_weights, hidden_states.dtype, down_config
    )
    return decode_topk_sum(slots_out, top_k_index, num_experts)
