# [scratch, bench only] Fused Triton kernels for the fixed elementwise chains of the GLM-MoE-DSA decode step, the way
# vLLM fuses them (`_fused_norm_rope_kernel`, its fused Q kernel, `act_and_mul`, TRT-LLM's routing kernel):
#   - `fuse_norm_rope_write`: RMSNorm of the q_a latent; RMSNorm of the kv latent + rotary of the key, written straight
#     to the token's cache row; on indexer layers LayerNorm + rotary of the indexer key, written next to it.
#   - `fuse_q_pack`: rotary of the query's rope part + pack of [absorbed latent | rotated rope] into the head-padded
#     query buffer of the sparse MLA kernel.
#   - `fuse_add_rms_norm`: residual add + RMSNorm, both results.
#   - `fuse_router`: sigmoid, bias, top-k, gather, normalize, scale of the single-group router.
#   - `fuse_silu_mul`: silu(gate) * up.
# Every kernel has ONE fixed launch configuration (no autotune: nothing is benchmarked in band) and is a
# `torch.library.custom_op` with a fake implementation, so the same call works eagerly, under CUDA graph capture and
# inside a torch.compile graph. All of them are inference-only (no autograd formula): callers gate on no-grad.
#
# Numerics: products and sums that the unfused path rounds to bf16 one op at a time (rotary, residual add, silu * up)
# are rounded at the same places here, so they are bit-identical; the norms accumulate in fp32 like `F.rms_norm` /
# `F.layer_norm` but in another order (last-bit differences on a few elements per million).
import torch
import triton
import triton.language as tl
from torch import Tensor


def _pow2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


@triton.jit
def _bf16(x):
    # fp32 -> nearest bf16 value (ties to even), still held as fp32. Done on the bits: written as a cast, LLVM narrows
    # `fptrunc(fsub(fpext a, fpext b))` to a bf16 subtraction whose rounding on NVPTX is not ties-to-even
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _rope_pair(x1, x2, c, s):
    # (x1, x2) rotated by the pair's angle, each product and each sum rounded to bf16 like the torch ops they replace
    a = _bf16(x1 * c)
    b = _bf16(x2 * s)
    e = _bf16(x2 * c)
    d = _bf16(x1 * s)
    return _bf16(a - b).to(tl.bfloat16), _bf16(e + d).to(tl.bfloat16)


@triton.jit
def _norm_rope_write_kernel(
    qa_ptr, qr_ptr, qw_ptr, kva_ptr, kvw_ptr, cos_ptr, sin_ptr, widx_ptr, cache_ptr, ik_ptr, lnw_ptr, lnb_ptr,
    cache_stride, n_rows, cos_stride, eps_q, eps_kv, eps_ln,
    QD: tl.constexpr, KD: tl.constexpr, R: tl.constexpr, ID: tl.constexpr,
    QB: tl.constexpr, KB: tl.constexpr, PB: tl.constexpr, IB: tl.constexpr, HAS_INDEXER: tl.constexpr,
):  # fmt: skip
    m = tl.program_id(0)
    # RMSNorm of the query latent
    oq = tl.arange(0, QB)
    mq = oq < QD
    x = tl.load(qa_ptr + m * QD + oq, mask=mq, other=0.0).to(tl.float32)
    w = tl.load(qw_ptr + oq, mask=mq, other=0.0).to(tl.float32)
    rr = tl.rsqrt(tl.sum(x * x, axis=0) / QD + eps_q)
    tl.store(qr_ptr + m * QD + oq, _bf16(x * rr * w).to(tl.bfloat16), mask=mq)

    # the token's cache row: [kv latent (KD) | rotated key (R) | indexer key (ID)]. The row index is clamped: a bad
    # index must never become an out-of-bounds write
    row = tl.load(widx_ptr + m)
    row = tl.minimum(tl.maximum(row, 0), n_rows - 1)
    base = cache_ptr + row * cache_stride
    src = kva_ptr + m * (KD + R)
    ok = tl.arange(0, KB)
    mk = ok < KD
    kx = tl.load(src + ok, mask=mk, other=0.0).to(tl.float32)
    kw = tl.load(kvw_ptr + ok, mask=mk, other=0.0).to(tl.float32)
    krr = tl.rsqrt(tl.sum(kx * kx, axis=0) / KD + eps_kv)
    tl.store(base + ok, _bf16(kx * krr * kw).to(tl.bfloat16), mask=mk)

    # interleaved rotary of the key: pairs (2i, 2i + 1), output [x1 * cos - x2 * sin | x2 * cos + x1 * sin]
    oh = tl.arange(0, R // 2)
    c = tl.load(cos_ptr + m * cos_stride + oh).to(tl.float32)
    s = tl.load(sin_ptr + m * cos_stride + oh).to(tl.float32)
    x1 = tl.load(src + KD + 2 * oh).to(tl.float32)
    x2 = tl.load(src + KD + 2 * oh + 1).to(tl.float32)
    r1, r2 = _rope_pair(x1, x2, c, s)
    tl.store(base + KD + oh, r1)
    tl.store(base + KD + R // 2 + oh, r2)

    if HAS_INDEXER:
        # LayerNorm of the indexer key, rotary of its first R dims, the rest passes through
        isrc = ik_ptr + m * ID
        oi = tl.arange(0, IB)
        mi = oi < ID
        v = tl.load(isrc + oi, mask=mi, other=0.0).to(tl.float32)
        mean = tl.sum(v, axis=0) / ID
        d = tl.where(mi, v - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(d * d, axis=0) / ID + eps_ln)
        v1 = tl.load(isrc + 2 * oh).to(tl.float32)
        v2 = tl.load(isrc + 2 * oh + 1).to(tl.float32)
        y1 = (v1 - mean) * rstd * tl.load(lnw_ptr + 2 * oh).to(tl.float32) + tl.load(lnb_ptr + 2 * oh).to(tl.float32)
        y2 = (v2 - mean) * rstd * tl.load(lnw_ptr + 2 * oh + 1).to(tl.float32) + tl.load(lnb_ptr + 2 * oh + 1).to(
            tl.float32
        )
        i1, i2 = _rope_pair(_bf16(y1), _bf16(y2), c, s)
        tl.store(base + KD + R + oh, i1)
        tl.store(base + KD + R + R // 2 + oh, i2)
        op = R + tl.arange(0, PB)
        mp = op < ID
        vp = tl.load(isrc + op, mask=mp, other=0.0).to(tl.float32)
        yp = (vp - mean) * rstd * tl.load(lnw_ptr + op, mask=mp, other=0.0).to(tl.float32) + tl.load(
            lnb_ptr + op, mask=mp, other=0.0
        ).to(tl.float32)
        tl.store(base + KD + R + op, _bf16(yp).to(tl.bfloat16), mask=mp)


@triton.jit
def _q_pack_kernel(
    qb_ptr, lat_ptr, cos_ptr, sin_ptr, out_ptr,
    qb_stride_m, lat_stride_h, lat_stride_m, out_stride_m, out_stride_h, cos_stride,
    QK: tl.constexpr, NOPE: tl.constexpr, R: tl.constexpr, KD: tl.constexpr, KB: tl.constexpr,
):  # fmt: skip
    m = tl.program_id(0)
    h = tl.program_id(1)
    out = out_ptr + m * out_stride_m + h * out_stride_h
    ok = tl.arange(0, KB)
    mk = ok < KD
    lat = tl.load(lat_ptr + h * lat_stride_h + m * lat_stride_m + ok, mask=mk, other=0.0)
    tl.store(out + ok, lat, mask=mk)
    oh = tl.arange(0, R // 2)
    c = tl.load(cos_ptr + m * cos_stride + oh).to(tl.float32)
    s = tl.load(sin_ptr + m * cos_stride + oh).to(tl.float32)
    src = qb_ptr + m * qb_stride_m + h * QK + NOPE
    x1 = tl.load(src + 2 * oh).to(tl.float32)
    x2 = tl.load(src + 2 * oh + 1).to(tl.float32)
    r1, r2 = _rope_pair(x1, x2, c, s)
    tl.store(out + KD + oh, r1)
    tl.store(out + KD + R // 2 + oh, r2)


@triton.jit
def _add_rms_norm_kernel(a_ptr, b_ptr, sum_ptr, out_ptr, w_ptr, eps, D: tl.constexpr, DB: tl.constexpr):
    m = tl.program_id(0)
    o = tl.arange(0, DB)
    mask = o < D
    a = tl.load(a_ptr + m * D + o, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + m * D + o, mask=mask, other=0.0).to(tl.float32)
    x = _bf16(a + b)
    tl.store(sum_ptr + m * D + o, x.to(tl.bfloat16), mask=mask)
    w = tl.load(w_ptr + o, mask=mask, other=0.0).to(tl.float32)
    rr = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    tl.store(out_ptr + m * D + o, _bf16(x * rr * w).to(tl.bfloat16), mask=mask)


@triton.jit
def _router_kernel(
    logits_ptr, bias_ptr, idx_ptr, w_ptr, scale,
    E: tl.constexpr, EB: tl.constexpr, K: tl.constexpr, NORM: tl.constexpr,
):  # fmt: skip
    m = tl.program_id(0)
    o = tl.arange(0, EB)
    me = o < E
    s = tl.sigmoid(tl.load(logits_ptr + m * E + o, mask=me, other=0.0))
    choice = tl.where(me, s + tl.load(bias_ptr + o, mask=me, other=0.0), float("-inf"))
    sel = o < 0
    last = 0
    for k in tl.static_range(K):
        i = tl.argmax(choice, axis=0)
        hit = o == i
        choice = tl.where(hit, float("-inf"), choice)
        if k < K - 1:
            sel = sel | hit
        else:
            last = i
    # slot order of `torch.topk(..., sorted=False)` on CUDA: the K - 1 largest in index order, then the K-th largest
    is_last = o == last
    weights = s
    if NORM:
        weights = s / (tl.sum(tl.where(sel | is_last, s, 0.0), axis=0) + 1e-20)
    weights = weights * scale
    pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
    tl.store(idx_ptr + m * K + pos, o.to(tl.int64), mask=sel)
    tl.store(w_ptr + m * K + pos, weights, mask=sel)
    tl.store(idx_ptr + m * K + K - 1, last.to(tl.int64))
    tl.store(w_ptr + m * K + K - 1, tl.sum(tl.where(is_last, weights, 0.0), axis=0))


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = o < n
    g = tl.load(g_ptr + o, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + o, mask=mask, other=0.0).to(tl.float32)
    act = _bf16(g / (1.0 + tl.exp(-g)))
    tl.store(o_ptr + o, _bf16(act * u).to(tl.bfloat16), mask=mask)


# --- custom ops ------------------------------------------------------------------------------------------------------


def _is_bf16(*tensors) -> bool:
    return all(t.dtype == torch.bfloat16 and t.is_cuda for t in tensors)


def _norm_rope_write(
    qa: Tensor,
    q_weight: Tensor,
    kva: Tensor,
    kv_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    write_index: Tensor,
    cache_rows: Tensor,
    index_key: Tensor | None,
    ln_weight: Tensor | None,
    ln_bias: Tensor | None,
    eps_q: float,
    eps_kv: float,
    eps_ln: float,
) -> Tensor:
    """qa [..., M, QD] (q_a_proj output), kva [..., M, KD + R] (kv_a_proj output), cos / sin [..., M, R] (per-pair angle
    in the first R / 2), write_index [M] rows of `cache_rows` [rows, >= KD + R (+ ID)], index_key [..., M, ID] (the
    indexer's `wk` output) or None. Returns the normed query latent, like qa; the cache rows are written in place."""
    qd, kd, rope = qa.shape[-1], kv_weight.shape[0], cos.shape[-1]
    num_tokens = write_index.shape[0]
    has_indexer = index_key is not None
    idim = index_key.shape[-1] if has_indexer else rope
    assert _is_bf16(qa, q_weight, kva, kv_weight, cos, sin, cache_rows) and write_index.dtype == torch.int64
    assert qa.is_contiguous() and kva.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    assert qa.numel() == num_tokens * qd and kva.numel() == num_tokens * (kd + rope) and cos.numel() == num_tokens * rope
    assert cache_rows.dim() == 2 and cache_rows.stride(1) == 1 and rope % 2 == 0 and _pow2(rope) == rope
    assert cache_rows.shape[1] >= kd + rope + (idim if has_indexer else 0)
    if has_indexer:
        assert _is_bf16(index_key, ln_weight, ln_bias) and index_key.is_contiguous()
        assert index_key.numel() == num_tokens * idim and idim >= rope
    q_resid = torch.empty_like(qa)
    _norm_rope_write_kernel[(num_tokens,)](
        qa, q_resid, q_weight, kva, kv_weight, cos, sin, write_index, cache_rows,
        index_key if has_indexer else qa, ln_weight if has_indexer else q_weight, ln_bias if has_indexer else q_weight,
        cache_rows.stride(0), cache_rows.shape[0], rope, eps_q, eps_kv, eps_ln,
        QD=qd, KD=kd, R=rope, ID=idim, QB=_pow2(qd), KB=_pow2(kd), PB=_pow2(max(idim - rope, 1)), IB=_pow2(idim),
        HAS_INDEXER=has_indexer, num_warps=8,
    )  # fmt: skip
    return q_resid


def _norm_rope_write_fake(
    qa, q_weight, kva, kv_weight, cos, sin, write_index, cache_rows, index_key, ln_weight, ln_bias, eps_q, eps_kv, eps_ln
):
    return torch.empty_like(qa)


def _q_pack(qb: Tensor, q_lat: Tensor, cos: Tensor, sin: Tensor, q_abs: Tensor, nope: int) -> None:
    """qb [..., M, H * QK] (q_b_proj output, each head [nope | rope]), q_lat [H, M, KD] (absorbed nope part), cos / sin
    [..., M, R]. Writes heads 0..H-1 of q_abs [M, >= H, KD + R] as [q_lat | rotary(rope)], in place."""
    num_heads, num_tokens, kd = q_lat.shape
    rope = cos.shape[-1]
    qk = nope + rope
    assert _is_bf16(qb, q_lat, cos, sin, q_abs) and qb.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    assert qb.numel() == num_tokens * num_heads * qk and cos.numel() == num_tokens * rope and q_lat.stride(2) == 1
    assert q_abs.shape[0] == num_tokens and q_abs.shape[1] >= num_heads and q_abs.shape[2] == kd + rope
    assert q_abs.stride(2) == 1 and _pow2(rope) == rope
    _q_pack_kernel[(num_tokens, num_heads)](
        qb, q_lat, cos, sin, q_abs,
        num_heads * qk, q_lat.stride(0), q_lat.stride(1), q_abs.stride(0), q_abs.stride(1), rope,
        QK=qk, NOPE=nope, R=rope, KD=kd, KB=_pow2(kd), num_warps=4,
    )  # fmt: skip


def _add_rms_norm(a: Tensor, b: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """(a + b, rms_norm(a + b) * weight) for a, b [..., D]."""
    dim = a.shape[-1]
    assert _is_bf16(a, b, weight) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous()
    total, normed = torch.empty_like(a), torch.empty_like(a)
    _add_rms_norm_kernel[(a.numel() // dim,)](a, b, total, normed, weight, eps, D=dim, DB=_pow2(dim), num_warps=8)
    return total, normed


def _add_rms_norm_fake(a, b, weight, eps):
    return torch.empty_like(a), torch.empty_like(a)


def _router(logits: Tensor, bias: Tensor, top_k: int, scale: float, norm: bool) -> tuple[Tensor, Tensor]:
    """Single-group router: logits [M, E] fp32 -> (top-k expert ids [M, K] int64, their weights [M, K] fp32)."""
    num_tokens, num_experts = logits.shape
    assert logits.dtype == torch.float32 and bias.dtype == torch.float32 and logits.is_contiguous() and logits.is_cuda
    indices = torch.empty((num_tokens, top_k), dtype=torch.int64, device=logits.device)
    weights = torch.empty((num_tokens, top_k), dtype=torch.float32, device=logits.device)
    _router_kernel[(num_tokens,)](
        logits, bias, indices, weights, scale, E=num_experts, EB=_pow2(num_experts), K=top_k, NORM=norm, num_warps=4
    )
    return indices, weights


def _router_fake(logits, bias, top_k, scale, norm):
    return (
        torch.empty((logits.shape[0], top_k), dtype=torch.int64, device=logits.device),
        torch.empty((logits.shape[0], top_k), dtype=torch.float32, device=logits.device),
    )


def _silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """silu(gate) * up."""
    assert _is_bf16(gate, up) and gate.shape == up.shape and gate.is_contiguous() and up.is_contiguous()
    out = torch.empty_like(gate)
    block = 1024
    _silu_mul_kernel[(triton.cdiv(gate.numel(), block),)](gate, up, out, gate.numel(), BLOCK=block, num_warps=4)
    return out


def _silu_mul_fake(gate, up):
    return torch.empty_like(gate)


torch.library.custom_op("transformers::fuse_norm_rope_write", _norm_rope_write, mutates_args=("cache_rows",))
torch.library.register_fake("transformers::fuse_norm_rope_write", _norm_rope_write_fake)
torch.library.custom_op("transformers::fuse_q_pack", _q_pack, mutates_args=("q_abs",))
torch.library.custom_op("transformers::fuse_add_rms_norm", _add_rms_norm, mutates_args=())
torch.library.register_fake("transformers::fuse_add_rms_norm", _add_rms_norm_fake)
torch.library.custom_op("transformers::fuse_router", _router, mutates_args=())
torch.library.register_fake("transformers::fuse_router", _router_fake)
torch.library.custom_op("transformers::fuse_silu_mul", _silu_mul, mutates_args=())
torch.library.register_fake("transformers::fuse_silu_mul", _silu_mul_fake)
