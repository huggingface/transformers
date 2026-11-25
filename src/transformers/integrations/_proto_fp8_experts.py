"""Prototype (bench only, not for upstream as is) routed-experts forwards for block-FP8 MoE at inference.

Two implementations of `FP8Experts.forward(hidden_states, top_k_index, top_k_weights)`, both off by default:

- `PROTO_EXPERTS_TRTLLM=1`: flashinfer's TRT-LLM routed FP8 MoE, called the way vLLM's `TrtLlmFp8ExpertsModular` does.
  The weights are re-laid once at load (`trtllm_prepare`, bytes only moved), the originals are kept. Blackwell only.
  With `PROTO_EXPERTS_TRTLLM_INPLACE=1` there is no copy: the parameters themselves are re-laid in place
  (`trtllm_relayout_`) while the continuous batching loop runs, and given back in the checkpoint's layout otherwise.
- `PROTO_EXPERTS_DECODE=1`: a Triton kernel pair with the dispatch fused and a decode schedule: one program per
  (expert, N tile) walking that expert's routed rows. Candidate for `kernels-community/finegrained-fp8`.
  `PROTO_EXPERTS_DECODE=fold` uses one activation scale per (routed row, weight tile) in the first kernel.

This file imports only torch and triton so that the benchmarks can load it without transformers.
"""

import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


FP8 = torch.float8_e4m3fn
PROTO_EXPERTS_TRTLLM = os.environ.get("PROTO_EXPERTS_TRTLLM", "0") == "1"
PROTO_EXPERTS_DECODE = os.environ.get("PROTO_EXPERTS_DECODE", "0") in ("1", "fold")
CALLS = {"trtllm": 0, "decode": 0}  # host-side launches of each forward (a CUDA graph replay does not count)


# ---- shared: per-token, per-128-group activation quant -----------------------------------------------------------------


_QUANT_GROUPS = 16


@triton.jit
def _act_quant_kernel(
    X, Y, S, stride_x, stride_y, stride_s_m, stride_s_g, NUM_GROUPS, GROUPS: tl.constexpr, WRITE_Q: tl.constexpr
):
    # one program = one row x GROUPS 128-wide groups. Same recipe as vLLM's `per_token_group_quant_fp8`: amax floored
    # at 1e-10, scale = amax / 448, values clamped to the e4m3 range
    m = tl.program_id(0).to(tl.int64)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    group_mask = groups < NUM_GROUPS
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(X + m * stride_x + cols, mask=group_mask[:, None], other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10) / 448.0
    if WRITE_Q:
        y = tl.minimum(tl.maximum(x / s[:, None], -448.0), 448.0)
        tl.store(Y + m * stride_y + cols, y.to(Y.dtype.element_ty), mask=group_mask[:, None])
    tl.store(S + m * stride_s_m + groups * stride_s_g, s, mask=group_mask)


def act_quant(x: torch.Tensor, scales_group_major: bool = False, scales_only: bool = False):
    """x [M, K] -> (e4m3 [M, K], fp32 scales [M, K / 128], or [K / 128, M] with `scales_group_major`). One launch.
    `scales_only` skips the e4m3 values and returns `(None, scales)`."""
    assert x.dim() == 2 and x.stride(1) == 1 and x.shape[1] % 128 == 0
    m, k = x.shape
    s = torch.empty((k // 128, m) if scales_group_major else (m, k // 128), dtype=torch.float32, device=x.device)
    y = None if scales_only else torch.empty(m, k, dtype=FP8, device=x.device)
    stride_s_m, stride_s_g = (1, m) if scales_group_major else (k // 128, 1)
    _act_quant_kernel[(m, triton.cdiv(k // 128, _QUANT_GROUPS))](
        x, s if scales_only else y, s, x.stride(0), k, stride_s_m, stride_s_g, k // 128,
        GROUPS=_QUANT_GROUPS, WRITE_Q=not scales_only, num_warps=2,
    )  # fmt: skip
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


def trtllm_relayout_(weight, swap_halves, inverse, chunk_bytes=256 << 20):
    """`trtllm_prepare`'s re-layout of one `[E, rows, K]` fp8 stack done in place (`inverse=False`), or undone in place
    (`inverse=True`). The tensor keeps its shape, storage and address: only the order of its bytes changes, so CUDA
    graphs captured on it stay valid. A few experts at a time go through a temporary of at most `chunk_bytes`."""
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    num_experts, m, k = weight.shape
    assert weight.dtype == FP8 and weight.is_contiguous() and k % 128 == 0, (weight.dtype, weight.shape)
    if os.environ.get("PROTO_EXPERTS_RELAYOUT_FAST", "0") == "1":
        return _trtllm_relayout_fast_(weight, swap_halves, inverse, chunk_bytes)
    rows = get_shuffle_matrix_a_row_indices(torch.empty(m, 1, dtype=torch.uint8), 64)  # rows[new_row] = old_row
    if swap_halves:
        rows = (rows + m // 2) % m
    if inverse:
        rows = torch.argsort(rows)  # old_row -> new_row
    rows = rows.to(weight.device)
    raw = weight.view(torch.uint8)
    step = max(1, chunk_bytes // (m * k))
    for start in range(0, num_experts, step):
        chunk = raw[start : start + step]
        n = chunk.shape[0]
        if inverse:  # [n, K / 128, rows, 128] -> [n, rows, K / 128, 128]
            new = chunk.view(n, k // 128, m, 128).permute(0, 2, 1, 3)[:, rows]
        else:  # [n, rows, K / 128, 128] -> [n, K / 128, rows, 128]
            new = chunk.view(n, m, k // 128, 128).permute(0, 2, 1, 3)[:, :, rows]
        chunk.copy_(new.reshape(n, m, k))


_RELAYOUT_ROWS = {}


def _trtllm_relayout_fast_(weight, swap_halves, inverse, chunk_bytes):
    """[scratch, gap4 pause] PROTO_EXPERTS_RELAYOUT_FAST=1: the same permutation as `trtllm_relayout_`, byte for
    byte, in two passes per chunk: one `index_select` that gathers whole 128-byte K blocks (as 16 int64 words, not 128
    single bytes) straight into the permuted order of a temporary, then one flat copy back. The rows index is built once
    per shape (`trtllm_relayout_` builds it on the host and argsorts it at every call)."""
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    num_experts, m, k = weight.shape
    key = (m, swap_halves, inverse, weight.device)
    rows = _RELAYOUT_ROWS.get(key)
    if rows is None:
        rows = get_shuffle_matrix_a_row_indices(torch.empty(m, 1, dtype=torch.uint8), 64)  # rows[new_row] = old_row
        if swap_halves:
            rows = (rows + m // 2) % m
        if inverse:
            rows = torch.argsort(rows)  # old_row -> new_row
        assert rows.min().item() == 0 and rows.max().item() == m - 1 and rows.unique().numel() == m
        rows = _RELAYOUT_ROWS[key] = rows.to(device=weight.device, dtype=torch.int64)
    kb = k // 128
    words = weight.view(torch.int64)  # [E, m, k / 8]: one K block is 16 words
    step = max(1, chunk_bytes // (m * k))
    tmp = torch.empty(min(step, num_experts) * m * k // 8, dtype=torch.int64, device=weight.device)
    for start in range(0, num_experts, step):
        chunk = words[start : start + step]
        n = chunk.shape[0]
        out = tmp[: n * m * k // 8]
        if inverse:  # [n, K / 128, rows, 16] -> [n, rows, K / 128, 16]: out[e, r, j] = chunk[e, j, rows[r]]
            src = chunk.reshape(n, kb, m, 16).permute(0, 2, 1, 3)
            torch.index_select(src, 1, rows, out=out.view(n, m, kb, 16))
        else:  # [n, rows, K / 128, 16] -> [n, K / 128, rows, 16]: out[e, j, r] = chunk[e, rows[r], j]
            src = chunk.reshape(n, m, kb, 16).permute(0, 2, 1, 3)
            torch.index_select(src, 2, rows, out=out.view(n, kb, m, 16))
        chunk.view(-1).copy_(out)


def trtllm_scales_(s13, s2, gate_up_scale, down_scale):
    """`trtllm_prepare`'s block scales written into existing buffers (same addresses for the CUDA graphs)."""
    n_gate = gate_up_scale.shape[1] // 2
    s13[:, :n_gate].copy_(gate_up_scale[:, n_gate:])
    s13[:, n_gate:].copy_(gate_up_scale[:, :n_gate])
    s13.clamp_(min=_TRTLLM_MIN_BLOCK_SCALE)
    s2.copy_(down_scale).clamp_(min=_TRTLLM_MIN_BLOCK_SCALE)


def trtllm_experts_forward(prepared, hidden_states, top_k_index, top_k_weights):
    """Activation quant (1 launch), then routing, gemm1, activation, gemm2 and finalize inside flashinfer."""
    import flashinfer
    from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

    if not torch.compiler.is_compiling():  # a traced count is a guard: dynamo recompiled on every call
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


def trtllm_experts_forward_routing(
    prepared, hidden_states, router_logits, routing_bias, top_k, n_group, topk_group, routed_scaling_factor,
    norm_topk_prob,
):  # fmt: skip
    """`PROTO_EXPERTS_TRTLLM_ROUTING`: the same MoE with the routing inside flashinfer's kernel, as vLLM and SGLang call it
    (DeepSeekV3 routing: sigmoid of the fp32 logits, bias added for the choice only, group-limited top-k, the chosen
    sigmoid scores normalized and scaled). Replaces the router's own top-k and weights."""
    import flashinfer
    from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

    if not torch.compiler.is_compiling():  # a traced count is a guard: dynamo recompiled on every call
        CALLS["trtllm"] += 1
    w13, s13, w2, s2 = prepared
    num_experts, _, two_i, _ = w13.shape
    num_tokens, hidden = hidden_states.shape
    x_q, x_s = act_quant(hidden_states, scales_group_major=True)
    out = torch.empty(num_tokens, hidden, dtype=hidden_states.dtype, device=hidden_states.device)
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=x_q,
        hidden_states_scale=x_s,
        gemm1_weights=w13,
        gemm1_weights_scale=s13,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=two_i // 2,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=2,  # DeepSeekV3
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        tune_max_num_tokens=8192,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        norm_topk_prob=norm_topk_prob,
        output=out,
    )
    return out


def trtllm_autotune(prepared, hidden, top_k, routing: dict | None, group=None):
    """`PROTO_EXPERTS_TRTLLM_AUTOTUNE`, at warm-up only: flashinfer's autotuner measures the TRT-LLM MoE tactics of every
    power-of-2 token count up to 8192 (`tune_max_num_tokens`) for this rank's expert shapes and keeps the fastest in its
    process-wide cache, which every later call (and CUDA graph capture) reads. One eager call per form of the op
    (precomputed top-k, and routing inside when `routing` gives its arguments). `group` (a gloo group of the TP ranks)
    averages the timings across ranks so they all pick the same tactics."""
    from flashinfer.autotuner import autotune, set_autotune_process_group

    w13 = prepared[0]
    num_experts, device = w13.shape[0], w13.device
    g = torch.Generator(device=device).manual_seed(0)
    n = 8192
    x = torch.randn(n, hidden, device=device, dtype=torch.bfloat16, generator=g)
    logits = torch.randn(n, num_experts, device=device, dtype=torch.float32, generator=g)
    top_k_weights, top_k_index = torch.topk(logits.sigmoid(), top_k, dim=-1)
    # PROTO_EXPERTS_TRTLLM_AUTOTUNE_CACHE=<dir>: the chosen tactics persist in a small JSON file per rank (as vLLM does),
    # loaded on entry and written back on exit, so only the first start of a machine pays the tuning. The name carries
    # what the tactics depend on; flashinfer also rejects a file whose kernels no longer match
    cache = None
    cache_dir = os.environ.get("PROTO_EXPERTS_TRTLLM_AUTOTUNE_CACHE")
    if cache_dir:
        import flashinfer
        import torch.distributed as dist

        rank, world = (dist.get_rank(group), dist.get_world_size(group)) if group is not None else (0, 1)
        gpu = torch.cuda.get_device_name(device).replace(" ", "_")
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(
            cache_dir, f"trtllm_moe_{gpu}_flashinfer{flashinfer.__version__}_e{num_experts}_h{hidden}_tp{world}_rank{rank}.json"
        )
    set_autotune_process_group(group)
    try:
        with torch.no_grad(), autotune(True, cache=cache):
            trtllm_experts_forward(prepared, x, top_k_index, top_k_weights)
            if routing is not None:
                trtllm_experts_forward_routing(prepared, x, logits, top_k=top_k, **routing)
    finally:
        set_autotune_process_group(None)
    torch.cuda.synchronize(device)


# ---- track 2: Triton kernel pair, dispatch fused, one program per (expert, N tile) -------------------------------------
#
#   route    (1 launch): counting sort of the S = tokens * top_k routed slots by expert, without atomics
#   quant    (1 launch): the activations to e4m3, per (token, 128 columns), once for all the programs that read them
#   gate_up  (E x I tiles): gather the expert's rows, gate and up GEMMs, SiLU, requant to fp8
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


_ROUTE_QUADRATIC_MAX_SLOTS = 2048
_ROUTE_ATOMIC_BLOCK = 256


@triton.jit
def _route_count_kernel(ExpertIds, Count, S, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(ExpertIds + slots, mask=slots < S, other=NUM_EXPERTS).to(tl.int32)
    tl.atomic_add(Count + expert, 1, mask=(slots < S) & (expert < NUM_EXPERTS))


@triton.jit
def _route_offsets_kernel(Count, ExpertStart, Cursor, NUM_EXPERTS: tl.constexpr, PADDED: tl.constexpr):
    experts = tl.arange(0, PADDED)
    count = tl.load(Count + experts, mask=experts < NUM_EXPERTS, other=0)
    end = tl.cumsum(count, 0)
    tl.store(ExpertStart, 0)
    tl.store(ExpertStart + 1 + experts, end, mask=experts < NUM_EXPERTS)
    tl.store(Cursor + experts, end - count, mask=experts < NUM_EXPERTS)


@triton.jit
def _route_scatter_kernel(ExpertIds, Perm, Cursor, S, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    # Each slot claims the next position of its expert. The order inside an expert depends on the race, which no
    # output depends on: every routed row is computed on its own.
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(ExpertIds + slots, mask=slots < S, other=NUM_EXPERTS).to(tl.int32)
    valid = (slots < S) & (expert < NUM_EXPERTS)
    position = tl.atomic_add(Cursor + expert, 1, mask=valid)
    tl.store(Perm + position, slots, mask=valid)


@triton.jit
def _decode_gate_up_rows_m_kernel(
    Hidden,  # (T, H) activations, bf16 (or e4m3 with PREQUANT)
    HiddenScale,  # (T, H / 128) fp32, only read with PREQUANT
    Perm,  # (S,) int32, expert-sorted position -> flat (token, slot) index
    ExpertStart,  # (E + 1,) int32
    GateUp,  # (E, 2I, H) e4m3
    GateUpScale,  # (E, 2I / 128, H / 128) fp32
    Inter,  # (S, I) expert-sorted: e4m3, or fp32 with RAW_OUT
    InterScale,  # (S, I / 128) fp32, not written with RAW_OUT
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
    BLOCK_N: tl.constexpr,  # 128, or a smaller power of two with RAW_OUT
    PREQUANT: tl.constexpr,
    RAW_OUT: tl.constexpr,  # store SiLU(gate) * up in fp32 and leave the fp8 requant to `act_quant`
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
    # one weight scale per (128 rows, K block): a tile of at most 128 rows has a single one
    gate_s_base = GateUpScale + e * stride_gus_e + ((pid_n * BLOCK_N) // 128) * stride_gus_n
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
            acc_gate += tl.dot(a, w_gate) * (a_s * s_gate)[:, None]
            acc_up += tl.dot(a, w_up) * (a_s * s_up)[:, None]
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up  # SiLU(gate) * up, fp32
        if RAW_OUT:
            tl.store(Inter + rows[:, None] * INTERMEDIATE_DIM + offs_n[None, :], act, mask=row_mask[:, None])
        else:
            # one fp8 requant per (row, 128-wide block)
            act_s = tl.max(tl.abs(act), axis=1) / 448.0
            act_q = (act / tl.maximum(act_s[:, None], 1e-12)).to(tl.float8e4nv)
            tl.store(Inter + rows[:, None] * INTERMEDIATE_DIM + offs_n[None, :], act_q, mask=row_mask[:, None])
            tl.store(InterScale + rows * (INTERMEDIATE_DIM // 128) + pid_n, act_s, mask=row_mask)


@triton.jit
def _decode_down_rows_m_kernel(
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
    BLOCK_N: tl.constexpr,  # power of two dividing HIDDEN_DIM
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
    if BLOCK_N <= 128:
        # one weight scale per (128 rows, K block): a tile of at most 128 rows has a single one
        w_s_base = DownScale + e * stride_ds_e + ((pid_n * BLOCK_N) // 128) * stride_ds_n
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
            if BLOCK_N <= 128:
                acc += tl.dot(a, w) * (a_s * w_s)[:, None]
            else:
                acc += tl.dot(a, w) * a_s[:, None] * w_s[None, :]
        # routing weight and scatter back to the flat (token, slot) row: the top-k sum is then a plain reduction
        slot = tl.load(Perm + rows, mask=row_mask, other=0).to(tl.int64)
        weight = tl.load(Weights + slot, mask=row_mask, other=0.0).to(tl.float32)
        acc = acc * weight[:, None]
        tl.store(
            Out + slot[:, None] * HIDDEN_DIM + offs_n[None, :], acc.to(Out.dtype.element_ty), mask=row_mask[:, None]
        )


# The same two kernels with the weights as the first operand of the dot: tiles of 128 weight rows x BLOCK_M routed rows.
# Triton lowers an fp8 dot to a native fp8 tensor-core MMA (tcgen05 on Blackwell, wgmma on Hopper) only when the first
# operand has at least 64 rows and the program at least 4 warps. With the routed rows first (16 or 32 of them at
# decode) it widens every fp8 byte to fp16 and uses `mma.sync` (`gap2/experts/scripts/probe_dot_shapes.py`).


@triton.jit
def _decode_gate_up_kernel(
    Hidden,  # (T, H) e4m3 activations (always pre-quantized here)
    HiddenScale,  # (T, H / 128) fp32
    Perm,  # (S,) int32, expert-sorted position -> flat (token, slot) index
    ExpertStart,  # (E + 1,) int32
    GateUp,  # (E, 2I, H) e4m3
    GateUpTiles,  # tensor descriptor of the same weights (only read with TMA)
    GateUpScale,  # (E, 2I / 128, H / 128) fp32
    Inter,  # (S, I) expert-sorted: e4m3, or fp32 with RAW_OUT
    InterScale,  # (S, I / 128) fp32, not written with RAW_OUT
    stride_h,
    stride_hs,
    stride_gu_e,
    stride_gu_n,
    stride_gus_e,
    stride_gus_n,
    NUM_TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # routed rows per tile (the dot's N)
    BLOCK_N: tl.constexpr,  # weight rows per tile (the dot's M): 128, or 64 with RAW_OUT
    PREQUANT: tl.constexpr,
    RAW_OUT: tl.constexpr,  # store SiLU(gate) * up in fp32 and leave the fp8 requant to `act_quant`
    TMA: tl.constexpr,  # load the weight tiles through `GateUpTiles` (TMA on Hopper and Blackwell)
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_k = tl.arange(0, 128)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_base = GateUp + e * stride_gu_e + offs_n[:, None] * stride_gu_n + offs_k[None, :]
    up_base = gate_base + INTERMEDIATE_DIM * stride_gu_n
    # with TMA, `GateUpTiles` describes the weights as a (E * 2I, H) matrix cut in boxes of BLOCK_N rows x 128 columns
    gate_row = tl.program_id(0) * (2 * INTERMEDIATE_DIM) + pid_n * BLOCK_N
    # one weight scale per (128 rows, K block): a tile of at most 128 rows has a single one
    gate_s_base = GateUpScale + e * stride_gus_e + ((pid_n * BLOCK_N) // 128) * stride_gus_n
    up_s_base = gate_s_base + (INTERMEDIATE_DIM // 128) * stride_gus_n
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        token = (tl.load(Perm + rows, mask=row_mask, other=0) // NUM_TOP_K).to(tl.int64)
        a_base = Hidden + offs_k[:, None] + token[None, :] * stride_h
        acc_gate = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for kb in range(0, HIDDEN_DIM // 128):
            a = tl.load(a_base + kb * 128, mask=row_mask[None, :], other=0.0)
            a_s = tl.load(HiddenScale + token * stride_hs + kb, mask=row_mask, other=0.0)
            if TMA:
                w_gate = GateUpTiles.load([gate_row, kb * 128])
                w_up = GateUpTiles.load([gate_row + INTERMEDIATE_DIM, kb * 128])
            else:
                w_gate = tl.load(gate_base + kb * 128)
                w_up = tl.load(up_base + kb * 128)
            s_gate = tl.load(gate_s_base + kb)
            s_up = tl.load(up_s_base + kb)
            acc_gate += tl.dot(w_gate, a) * (a_s * s_gate)[None, :]
            acc_up += tl.dot(w_up, a) * (a_s * s_up)[None, :]
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up  # SiLU(gate) * up, fp32
        if RAW_OUT:
            tl.store(Inter + rows[None, :] * INTERMEDIATE_DIM + offs_n[:, None], act, mask=row_mask[None, :])
        else:
            # one fp8 requant per (row, 128-wide block)
            act_s = tl.max(tl.abs(act), axis=0) / 448.0
            act_q = (act / tl.maximum(act_s[None, :], 1e-12)).to(tl.float8e4nv)
            tl.store(Inter + rows[None, :] * INTERMEDIATE_DIM + offs_n[:, None], act_q, mask=row_mask[None, :])
            tl.store(InterScale + rows * (INTERMEDIATE_DIM // 128) + pid_n, act_s, mask=row_mask)


@triton.jit
def _decode_gate_up_fold_kernel(
    Hidden,  # (T, H) activations, not quantized
    HiddenScale,  # (T, H / 128) fp32: max |activation| / 448 per (token, 128 columns)
    Perm,  # (S,) int32, expert-sorted position -> flat (token, slot) index
    ExpertStart,  # (E + 1,) int32
    GateUpTiles,  # tensor descriptor of the (E * 2I, H) e4m3 weights, boxes of 128 rows x 128 columns
    GateUpScale,  # (E, 2I / 128, H / 128) fp32
    Inter,  # (S, I) e4m3, expert-sorted
    InterScale,  # (S, I / 128) fp32
    stride_h,
    stride_hs,
    stride_gus_e,
    stride_gus_n,
    NUM_TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # routed rows per tile (the dot's N)
    K_BLOCKS_PADDED: tl.constexpr,  # next power of two of HIDDEN_DIM / 128
):
    """gate_up with one activation scale per (routed row, weight tile) instead of one per (row, 128 columns).

    A block-scaled dot is `sum_kb a_s[kb] * w_s[kb] * dot(a_q[kb], w_q[kb])`: the scales change with the K block, so
    the usual loop reads the MMA result back and does an fp32 multiply-add per output element and K block. Here the
    activation is quantized inside the program as `a * w_s[kb] / S`, with `S = max_kb(amax[kb] * w_s[kb]) / 448`
    known before the loop from the per-block maxima, so every K block shares the scale `S` and the tensor core
    accumulates over the whole reduction. e4m3 is a floating-point format: a block whose scaled values are small
    keeps its 3 mantissa bits down to 2^-15 of the row's largest block.
    """
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_k = tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    gate_row = tl.program_id(0) * (2 * INTERMEDIATE_DIM) + pid_n * 128
    gate_s_base = GateUpScale + e * stride_gus_e + pid_n * stride_gus_n
    up_s_base = gate_s_base + (INTERMEDIATE_DIM // 128) * stride_gus_n
    k_blocks = tl.arange(0, K_BLOCKS_PADDED)
    k_mask = k_blocks < HIDDEN_DIM // 128
    s_gate_all = tl.load(gate_s_base + k_blocks, mask=k_mask, other=0.0)
    s_up_all = tl.load(up_s_base + k_blocks, mask=k_mask, other=0.0)
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        token = (tl.load(Perm + rows, mask=row_mask, other=0) // NUM_TOP_K).to(tl.int64)
        a_s_all = tl.load(
            HiddenScale + token[None, :] * stride_hs + k_blocks[:, None],
            mask=k_mask[:, None] & row_mask[None, :],
            other=0.0,
        )
        scale_gate = tl.max(a_s_all * s_gate_all[:, None], axis=0)
        scale_up = tl.max(a_s_all * s_up_all[:, None], axis=0)
        inv_gate = 1.0 / tl.maximum(scale_gate, 1e-30)
        inv_up = 1.0 / tl.maximum(scale_up, 1e-30)
        a_base = Hidden + offs_k[:, None] + token[None, :] * stride_h
        acc_gate = tl.zeros((128, BLOCK_M), dtype=tl.float32)
        acc_up = tl.zeros((128, BLOCK_M), dtype=tl.float32)
        for kb in range(0, HIDDEN_DIM // 128):
            a = tl.load(a_base + kb * 128, mask=row_mask[None, :], other=0.0).to(tl.float32)
            s_gate = tl.load(gate_s_base + kb)
            s_up = tl.load(up_s_base + kb)
            a_gate = tl.minimum(tl.maximum(a * (s_gate * inv_gate)[None, :], -448.0), 448.0).to(tl.float8e4nv)
            a_up = tl.minimum(tl.maximum(a * (s_up * inv_up)[None, :], -448.0), 448.0).to(tl.float8e4nv)
            acc_gate = tl.dot(GateUpTiles.load([gate_row, kb * 128]), a_gate, acc_gate)
            acc_up = tl.dot(GateUpTiles.load([gate_row + INTERMEDIATE_DIM, kb * 128]), a_up, acc_up)
        gate = acc_gate * scale_gate[None, :]
        up = acc_up * scale_up[None, :]
        act = gate * tl.sigmoid(gate) * up  # SiLU(gate) * up, fp32
        # one fp8 requant per (row, 128-wide block)
        act_s = tl.max(tl.abs(act), axis=0) / 448.0
        act_q = (act / tl.maximum(act_s[None, :], 1e-12)).to(tl.float8e4nv)
        tl.store(Inter + rows[None, :] * INTERMEDIATE_DIM + offs_n[:, None], act_q, mask=row_mask[None, :])
        tl.store(InterScale + rows * (INTERMEDIATE_DIM // 128) + pid_n, act_s, mask=row_mask)


@triton.jit
def _decode_down_kernel(
    Inter,  # (S, I) e4m3, expert-sorted
    InterScale,  # (S, I / 128) fp32
    Perm,  # (S,) int32
    ExpertStart,  # (E + 1,) int32
    Down,  # (E, H, I) e4m3
    DownTiles,  # tensor descriptor of the same weights (only read with TMA)
    DownScale,  # (E, H / 128, I / 128) fp32
    Weights,  # (S,) routing weight per flat (token, slot)
    Out,  # (S, H) flat (token, slot) rows
    stride_d_e,
    stride_d_n,
    stride_ds_e,
    stride_ds_n,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # routed rows per tile (the dot's N)
    BLOCK_N: tl.constexpr,  # weight rows per tile (the dot's M), power of two dividing HIDDEN_DIM
    TMA: tl.constexpr,  # load the weight tiles through `DownTiles` (TMA on Hopper and Blackwell)
):
    e = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_k = tl.arange(0, 128)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    w_base = Down + e * stride_d_e + offs_n[:, None] * stride_d_n + offs_k[None, :]
    # with TMA, `DownTiles` describes the weights as a (E * H, I) matrix cut in boxes of BLOCK_N rows x 128 columns
    w_row = tl.program_id(0) * HIDDEN_DIM + pid_n * BLOCK_N
    if BLOCK_N <= 128:
        # one weight scale per (128 rows, K block): a tile of at most 128 rows has a single one
        w_s_base = DownScale + e * stride_ds_e + ((pid_n * BLOCK_N) // 128) * stride_ds_n
    else:
        w_s_base = DownScale + e * stride_ds_e + (offs_n // 128) * stride_ds_n
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for kb in range(0, INTERMEDIATE_DIM // 128):
            a = tl.load(
                Inter + (kb * 128 + offs_k)[:, None] + rows[None, :] * INTERMEDIATE_DIM,
                mask=row_mask[None, :],
                other=0.0,
            )
            a_s = tl.load(InterScale + rows * (INTERMEDIATE_DIM // 128) + kb, mask=row_mask, other=0.0)
            if TMA:
                w = DownTiles.load([w_row, kb * 128])
            else:
                w = tl.load(w_base + kb * 128)
            w_s = tl.load(w_s_base + kb)
            if BLOCK_N <= 128:
                acc += tl.dot(w, a) * (a_s * w_s)[None, :]
            else:
                acc += tl.dot(w, a) * w_s[:, None] * a_s[None, :]
        # routing weight and scatter back to the flat (token, slot) row: the top-k sum is then a plain reduction
        slot = tl.load(Perm + rows, mask=row_mask, other=0).to(tl.int64)
        weight = tl.load(Weights + slot, mask=row_mask, other=0.0).to(tl.float32)
        acc = acc * weight[None, :]
        tl.store(
            Out + slot[None, :] * HIDDEN_DIM + offs_n[:, None], acc.to(Out.dtype.element_ty), mask=row_mask[None, :]
        )


@triton.jit
def _decode_down_tiles_kernel(
    Inter,  # (S, I) e4m3, expert-sorted
    InterScale,  # (S, I / 128) fp32
    Perm,  # (S,) int32
    ExpertStart,  # (E + 1,) int32
    DownTiles,  # tensor descriptor of the (E * H, I) e4m3 weights, boxes of BLOCK_N rows x 128 columns
    DownScale,  # (E, H / 128, I / 128) fp32
    Weights,  # (S,) routing weight per flat (token, slot)
    Out,  # (S, H) flat (token, slot) rows
    stride_ds_e,
    stride_ds_n,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # routed rows per tile (the dot's N)
    BLOCK_N: tl.constexpr,  # weight rows per tile (the dot's M): 128 or 256
    TILES: tl.constexpr,  # weight tiles walked by one program
):
    """The down kernel with one program per (expert, TILES weight tiles) instead of one per (expert, tile).

    The reduction is short (I / 128 blocks, 2 on a TP8 rank), so a program per tile is mostly start-up: it finds its
    expert's rows, loads their routing metadata, and stops after two weight loads. Walking several tiles in one
    program pays that once and gives the loop enough iterations for the weight loads to be pipelined.
    """
    e = tl.program_id(0).to(tl.int64)
    start = tl.load(ExpertStart + e)
    end = tl.load(ExpertStart + e + 1)
    if start >= end:
        return
    offs_k = tl.arange(0, 128)
    first_n = tl.program_id(1) * (TILES * BLOCK_N)
    w_row = tl.program_id(0) * HIDDEN_DIM + first_n
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < end
        slot = tl.load(Perm + rows, mask=row_mask, other=0).to(tl.int64)
        weight = tl.load(Weights + slot, mask=row_mask, other=0.0).to(tl.float32)
        for t in range(0, TILES):
            offs_n = first_n + t * BLOCK_N + tl.arange(0, BLOCK_N)
            if BLOCK_N <= 128:
                w_s_base = DownScale + e * stride_ds_e + ((first_n + t * BLOCK_N) // 128) * stride_ds_n
            else:
                w_s_base = DownScale + e * stride_ds_e + (offs_n // 128) * stride_ds_n
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for kb in tl.static_range(INTERMEDIATE_DIM // 128):
                a = tl.load(
                    Inter + (kb * 128 + offs_k)[:, None] + rows[None, :] * INTERMEDIATE_DIM,
                    mask=row_mask[None, :],
                    other=0.0,
                )
                a_s = tl.load(InterScale + rows * (INTERMEDIATE_DIM // 128) + kb, mask=row_mask, other=0.0)
                w = DownTiles.load([w_row + t * BLOCK_N, kb * 128])
                w_s = tl.load(w_s_base + kb)
                if BLOCK_N <= 128:
                    acc += tl.dot(w, a) * (a_s * w_s)[None, :]
                else:
                    acc += tl.dot(w, a) * w_s[:, None] * a_s[None, :]
            acc = acc * weight[None, :]
            tl.store(
                Out + slot[None, :] * HIDDEN_DIM + offs_n[:, None],
                acc.to(Out.dtype.element_ty),
                mask=row_mask[None, :],
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


# Fixed configs, chosen offline on one B300 at the GLM-5.3 TP8 rank shapes (gap2/experts REPORT, jobs 460445, 460506,
# 460524, 460685): no in-band autotune. Keyed on the mean number of routed rows per expert, which the shapes give
# without reading anything back: (max mean rows per expert, routed rows per tile, num_warps of gate_up). The tile is
# sized so that almost every expert fits in one (its weights are then read exactly once). Above the last entry the
# programs walk their expert's rows in tiles of 64: still ahead of the hub's persistent schedule at 150-250 rows per
# expert.
DECODE_TILES = [(3, 8, 4), (10, 16, 4), (22, 32, 4)]
DECODE_LARGEST_TILE = (64, 8)


def decode_config(num_slots: int, num_experts: int, hidden: int, inter: int, fold: bool = False):
    """`(gate_up config, down config)` for this many routed slots. A config is (routed rows per tile, weight rows per
    tile, num_warps, num_stages, variant[, weight tiles per program])."""
    block_m, num_warps = DECODE_LARGEST_TILE
    for max_rows, tile_rows, tile_warps in DECODE_TILES:
        if num_slots <= max_rows * num_experts:
            block_m, num_warps = tile_rows, tile_warps
            break
    gate_up_config = (block_m, 128, num_warps, 3, "fold" if fold else "tma")
    if block_m == 8 and hidden % (6 * 128) == 0 and inter <= 512:
        # a couple of rows per expert and a short reduction: 6 weight tiles per program (see the kernel's docstring)
        return gate_up_config, (8, 128, 4, 2, "tma", 6)
    return gate_up_config, (min(block_m, 16), 256, 4, 2, "tma")


def decode_route(top_k_index, num_experts):
    """`(perm, expert_start)`: expert-sorted position -> flat (token, slot) index, and each expert's first position.
    One launch (comparison sort, deterministic) up to 2048 slots, an atomic counting sort in 4 launches above."""
    num_slots = top_k_index.numel()
    device = top_k_index.device
    perm = torch.empty(num_slots, dtype=torch.int32, device=device)
    expert_start = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    if num_slots <= _ROUTE_QUADRATIC_MAX_SLOTS:
        _route_kernel[(triton.cdiv(num_slots, _ROUTE_BLOCK) + triton.cdiv(num_experts + 1, _ROUTE_BLOCK),)](
            top_k_index, perm, expert_start, num_slots,
            NUM_EXPERTS=num_experts, BLOCK=_ROUTE_BLOCK, CHUNK=_ROUTE_CHUNK, num_warps=4,
        )  # fmt: skip
        return perm, expert_start
    count = torch.zeros(num_experts, dtype=torch.int32, device=device)
    cursor = torch.empty(num_experts, dtype=torch.int32, device=device)
    grid = (triton.cdiv(num_slots, _ROUTE_ATOMIC_BLOCK),)
    _route_count_kernel[grid](
        top_k_index, count, num_slots, NUM_EXPERTS=num_experts, BLOCK=_ROUTE_ATOMIC_BLOCK, num_warps=4
    )
    _route_offsets_kernel[(1,)](
        count, expert_start, cursor, NUM_EXPERTS=num_experts, PADDED=triton.next_power_of_2(num_experts), num_warps=4
    )
    _route_scatter_kernel[grid](
        top_k_index, perm, cursor, num_slots, NUM_EXPERTS=num_experts, BLOCK=_ROUTE_ATOMIC_BLOCK, num_warps=4
    )
    return perm, expert_start


def decode_gate_up(hidden_states, perm, expert_start, gate_up, gate_up_scale, num_top_k, config, prequant=False):
    """`(inter, inter_scale)`: SiLU(gate) * up of every routed row, e4m3 with one fp32 scale per 128 columns.
    `config` is (BLOCK_M, BLOCK_N, num_warps, num_stages). With an N tile narrower than the 128-wide scale block the
    kernel stores fp32 and the requant is one more small launch."""
    block_m, block_n, num_warps, num_stages, variant = (*config, "tma")[:5]
    if variant == "fold":
        num_experts, two_i, hidden = gate_up.shape
        inter = two_i // 2
        assert block_n == 128 and hidden % 128 == 0 and inter % 128 == 0 and hidden_states.stride(1) == 1
        inter_q = torch.empty(perm.numel(), inter, dtype=FP8, device=gate_up.device)
        inter_s = torch.empty(perm.numel(), inter // 128, dtype=torch.float32, device=gate_up.device)
        _, x_s = act_quant(hidden_states, scales_only=True)
        _decode_gate_up_fold_kernel[(num_experts, inter // 128)](
            hidden_states, x_s, perm, expert_start,
            TensorDescriptor.from_tensor(gate_up.view(num_experts * two_i, hidden), [128, 128]), gate_up_scale,
            inter_q, inter_s, hidden_states.stride(0), hidden // 128, gate_up_scale.stride(0), gate_up_scale.stride(1),
            NUM_TOP_K=num_top_k, HIDDEN_DIM=hidden, INTERMEDIATE_DIM=inter, BLOCK_M=block_m,
            K_BLOCKS_PADDED=triton.next_power_of_2(hidden // 128), num_warps=num_warps, num_stages=num_stages,
        )  # fmt: skip
        return inter_q, inter_s
    weights_first = variant != "rows"
    prequant = prequant or weights_first
    num_experts, two_i, hidden = gate_up.shape
    inter = two_i // 2
    num_slots = perm.numel()
    device = hidden_states.device
    raw_out = block_n < 128
    assert hidden % 128 == 0 and inter % 128 == 0 and 128 % block_n == 0
    if raw_out:
        inter_q = torch.empty(num_slots, inter, dtype=torch.float32, device=device)
        inter_s = inter_q
    else:
        inter_q = torch.empty(num_slots, inter, dtype=FP8, device=device)
        inter_s = torch.empty(num_slots, inter // 128, dtype=torch.float32, device=device)
    if prequant:
        x, x_s = act_quant(hidden_states)
        stride_x, stride_xs = hidden, hidden // 128
    else:
        x, x_s = hidden_states, hidden_states
        stride_x, stride_xs = hidden_states.stride(0), 0
    args = (
        x, x_s, perm, expert_start, gate_up, gate_up_scale, inter_q, inter_s,
        stride_x, stride_xs, gate_up.stride(0), gate_up.stride(1), gate_up_scale.stride(0), gate_up_scale.stride(1),
    )  # fmt: skip
    meta = {
        "NUM_TOP_K": num_top_k, "HIDDEN_DIM": hidden, "INTERMEDIATE_DIM": inter, "BLOCK_M": block_m,
        "BLOCK_N": block_n, "PREQUANT": prequant, "RAW_OUT": raw_out, "num_warps": num_warps, "num_stages": num_stages,
    }  # fmt: skip
    if weights_first:
        tiles = gate_up
        if variant == "tma":
            tiles = TensorDescriptor.from_tensor(gate_up.view(num_experts * two_i, hidden), [block_n, 128])
        _decode_gate_up_kernel[(num_experts, inter // block_n)](
            *args[:5], tiles, *args[5:], **meta, TMA=variant == "tma"
        )
    else:
        _decode_gate_up_rows_m_kernel[(num_experts, inter // block_n)](*args, **meta)
    if raw_out:
        return act_quant(inter_q)
    return inter_q, inter_s


def decode_down(inter_q, inter_s, perm, expert_start, down, down_scale, top_k_weights, out_dtype, config):
    """`(S, hidden)`: every routed row's weighted expert output, in flat (token, slot) order."""
    block_m, block_n, num_warps, num_stages = config[:4]
    variant = config[4] if len(config) > 4 else "tma"
    tiles = config[5] if len(config) > 5 else 1
    num_experts, hidden, inter = down.shape
    if tiles > 1:
        # one program per (expert, `tiles` weight tiles); the unrolled K loop is meant for short reductions
        assert variant == "tma" and hidden % (tiles * block_n) == 0 and inter % 128 == 0 and inter // 128 <= 4
        slots_out = torch.empty(perm.numel(), hidden, dtype=out_dtype, device=down.device)
        _decode_down_tiles_kernel[(num_experts, hidden // (tiles * block_n))](
            inter_q, inter_s, perm, expert_start,
            TensorDescriptor.from_tensor(down.view(num_experts * hidden, inter), [block_n, 128]), down_scale,
            top_k_weights, slots_out, down_scale.stride(0), down_scale.stride(1),
            HIDDEN_DIM=hidden, INTERMEDIATE_DIM=inter, BLOCK_M=block_m, BLOCK_N=block_n, TILES=tiles,
            num_warps=num_warps, num_stages=num_stages,
        )  # fmt: skip
        return slots_out
    assert hidden % block_n == 0 and inter % 128 == 0 and (block_n % 128 == 0 or 128 % block_n == 0)
    slots_out = torch.empty(perm.numel(), hidden, dtype=out_dtype, device=down.device)
    args = (
        inter_q, inter_s, perm, expert_start, down, down_scale, top_k_weights, slots_out,
        down.stride(0), down.stride(1), down_scale.stride(0), down_scale.stride(1),
    )  # fmt: skip
    meta = {
        "HIDDEN_DIM": hidden, "INTERMEDIATE_DIM": inter, "BLOCK_M": block_m, "BLOCK_N": block_n,
        "num_warps": num_warps, "num_stages": num_stages,
    }  # fmt: skip
    if variant != "rows":
        tiles = down
        if variant == "tma":
            tiles = TensorDescriptor.from_tensor(down.view(num_experts * hidden, inter), [block_n, 128])
        _decode_down_kernel[(num_experts, hidden // block_n)](*args[:5], tiles, *args[5:], **meta, TMA=variant == "tma")
    else:
        _decode_down_rows_m_kernel[(num_experts, hidden // block_n)](*args, **meta)
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
    prequant: bool = True,
    fold: bool = False,
):
    """Block-FP8 routed experts: route, activation quant, gate_up, (requant for narrow tiles), down, top-k sum.
    Returns `(num_tokens, hidden)` in the input dtype."""
    if not torch.compiler.is_compiling():  # a traced count is a guard: dynamo recompiled on every call
        CALLS["decode"] += 1
    num_experts = gate_up.shape[0]
    num_top_k = top_k_index.shape[-1]
    if gate_up_config is None:
        gate_up_config, down_config = decode_config(
            top_k_index.numel(), num_experts, hidden_states.shape[1], gate_up.shape[1] // 2, fold
        )
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
