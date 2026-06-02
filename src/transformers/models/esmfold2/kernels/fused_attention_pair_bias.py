"""Fused Triton kernel for AttentionPairBias (forward + backward).

Inspired by cuequivariance's ``attention_pair_bias``
(https://docs.nvidia.com/cuda/cuequivariance/index.html); independently
re-implemented in Triton with a backward pass and no sequence-length gate.

Fuses ``LayerNorm(z) -> z @ w_proj_z.T -> (1-mask)*(-INF)`` into a single
kernel that emits a ``bias[B, H, Q, K]`` tensor, then dispatches the
attention itself to ``torch.nn.functional.scaled_dot_product_attention``.
"""

# ruff: noqa: E402

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

import torch
import triton
import triton.language as tl

# Static config — runtime autotune cold-start is unshippable for inference.
_BIAS_AUTOTUNE_CONFIGS = [
    triton.Config({"TILE_K": 64, "TILE_C": 64}, num_stages=3, num_warps=4)
]


@triton.autotune(
    configs=_BIAS_AUTOTUNE_CONFIGS,
    key=["Q", "K", "DIM_Z", "NUM_HEADS", "HEADS_PER_BLK", "HAS_MASK", "AFFINE"],
)
@triton.jit
def _pair_bias_kernel(
    z_ptr,  # [B, Q, K, DIM_Z]      pair tensor, bf16/fp16/fp32
    mask_ptr,  # [B, K]               key padding mask (bool/uint8)
    w_proj_z_ptr,  # [NUM_HEADS, DIM_Z]   bias-projection weight
    w_ln_ptr,  # [DIM_Z]              LN gamma (or unused)
    b_ln_ptr,  # [DIM_Z]              LN beta  (or unused)
    out_ptr,  # [B, NUM_HEADS, Q, K] fused output bias, same dtype as z
    mean_ptr,  # [B, Q, K]           saved LN mean (fp32) or dummy
    rstd_ptr,  # [B, Q, K]           saved LN rstd (fp32) or dummy
    B,
    Q,
    K,
    NEG_INF: tl.constexpr,
    EPS: tl.constexpr,
    DIM_Z: tl.constexpr,
    DIM_Z_PAD: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEADS_PER_BLK: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_C: tl.constexpr,  # tile over the DIM_Z reduction axis
    HAS_MASK: tl.constexpr,
    AFFINE: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    """One CTA owns (one Q-row × TILE_K K-positions × HEADS_PER_BLK heads).

    Grid: (cdiv(K, TILE_K), Q, B * cdiv(NUM_HEADS, HEADS_PER_BLK)).

    Each thread block computes a tile of the output bias
    ``bias[b, h_blk, q, k_tile]`` and stores it to ``out_ptr``. Layout of
    ``out_ptr`` is ``[B, NUM_HEADS, Q, K]`` so the downstream SDPA call can
    pass it directly as ``attn_mask`` (broadcast-compatible with the standard
    BHQK attention layout).

    When ``SAVE_STATS`` is set (training path), the per-row ``mean`` and
    ``rstd`` are stored into ``mean_ptr``/``rstd_ptr`` of shape ``[B, Q, K]``
    so the backward kernel can avoid recomputing them. Only the first head-block
    in each (B, Q, K) tile writes the stats to avoid races.
    """
    pid_k = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_bh = tl.program_id(2)
    NUM_HEAD_BLKS: tl.constexpr = (NUM_HEADS + HEADS_PER_BLK - 1) // HEADS_PER_BLK
    pid_b = pid_bh // NUM_HEAD_BLKS
    pid_hblk = pid_bh % NUM_HEAD_BLKS

    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    offs_z_full = tl.arange(0, DIM_Z_PAD)
    offs_h = pid_hblk * HEADS_PER_BLK + tl.arange(0, HEADS_PER_BLK)
    mask_k = offs_k < K
    mask_h = offs_h < NUM_HEADS

    z_full_ptrs = (
        z_ptr
        + pid_b * Q * K * DIM_Z
        + pid_q * K * DIM_Z
        + offs_k[:, None] * DIM_Z
        + offs_z_full[None, :]
    )
    mask_z_full = offs_z_full < DIM_Z
    z_full = tl.load(
        z_full_ptrs, mask=mask_k[:, None] & mask_z_full[None, :], other=0.0
    ).to(tl.float32)

    mean = tl.sum(z_full, axis=1) / DIM_Z
    z_centered = z_full - mean[:, None]
    z_centered = tl.where(mask_z_full[None, :], z_centered, 0.0)
    var = tl.sum(z_centered * z_centered, axis=1) / DIM_Z
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Save mean/rstd for backward — only the first head-block writes, all
    # head-blocks have the same value so this avoids racy duplicate writes.
    if SAVE_STATS:
        if pid_hblk == 0:
            stats_ptrs = mean_ptr + pid_b * Q * K + pid_q * K + offs_k
            tl.store(stats_ptrs, mean, mask=mask_k)
            stats_ptrs2 = rstd_ptr + pid_b * Q * K + pid_q * K + offs_k
            tl.store(stats_ptrs2, rstd, mask=mask_k)

    acc = tl.zeros([TILE_K, HEADS_PER_BLK], dtype=tl.float32)
    num_tiles_c = tl.cdiv(DIM_Z, TILE_C)
    for tc in range(0, num_tiles_c):
        offs_c = tc * TILE_C + tl.arange(0, TILE_C)
        mask_c = offs_c < DIM_Z

        z_slice_ptrs = (
            z_ptr
            + pid_b * Q * K * DIM_Z
            + pid_q * K * DIM_Z
            + offs_k[:, None] * DIM_Z
            + offs_c[None, :]
        )
        z_slice = tl.load(
            z_slice_ptrs, mask=mask_k[:, None] & mask_c[None, :], other=0.0
        ).to(tl.float32)

        z_norm = (z_slice - mean[:, None]) * rstd[:, None]
        if AFFINE:
            gamma = tl.load(w_ln_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
            beta = tl.load(b_ln_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            z_norm = z_norm * gamma[None, :] + beta[None, :]
        z_norm = tl.where(mask_c[None, :], z_norm, 0.0)

        w_ptrs = w_proj_z_ptr + (offs_h[None, :] * DIM_Z + offs_c[:, None])
        w_tile = tl.load(w_ptrs, mask=mask_h[None, :] & mask_c[:, None], other=0.0).to(
            tl.float32
        )

        acc = tl.dot(z_norm.to(tl.float32), w_tile, acc, input_precision="tf32x3")

    if HAS_MASK:
        m_tile = tl.load(mask_ptr + pid_b * K + offs_k, mask=mask_k, other=0).to(
            tl.int32
        )
        acc = acc + tl.where(m_tile == 0, NEG_INF, 0.0)[:, None]
    # Mask out-of-bounds K positions (we pad to TILE_K).
    acc = tl.where(mask_k[:, None], acc, NEG_INF)

    out_ptrs = (
        out_ptr
        + pid_b * NUM_HEADS * Q * K
        + offs_h[None, :] * Q * K
        + pid_q * K
        + offs_k[:, None]
    )
    tl.store(
        out_ptrs,
        acc.to(out_ptr.type.element_ty),
        mask=mask_k[:, None] & mask_h[None, :],
    )


# Backward is reduction-heavy (atomics into d_w_proj_z / d_pair_norm_*); modest
# TILE_K=32 balances ILP and register pressure on Hopper.
_BIAS_BWD_CONFIGS = [triton.Config({"TILE_K": 32}, num_stages=3, num_warps=4)]


@triton.autotune(
    configs=_BIAS_BWD_CONFIGS, key=["Q", "K", "DIM_Z", "NUM_HEADS", "AFFINE"]
)
@triton.jit
def _pair_bias_backward_kernel(
    z_ptr,  # [B, Q, K, DIM_Z]            input pair tensor (bf16)
    w_proj_z_ptr,  # [NUM_HEADS, DIM_Z]   bias-projection weight (bf16)
    w_ln_ptr,  # [DIM_Z]                  LN gamma (bf16, or dummy)
    b_ln_ptr,  # [DIM_Z]                  LN beta  (bf16, or dummy)
    mean_ptr,  # [B, Q, K]                saved LN mean (fp32)
    rstd_ptr,  # [B, Q, K]                saved LN rstd (fp32)
    d_bias_ptr,  # [B, NUM_HEADS, Q, K]   upstream gradient (bf16)
    d_z_ptr,  # [B, Q, K, DIM_Z]          output d_z (bf16)
    d_w_proj_z_ptr,  # [NUM_HEADS, DIM_Z] output d_w_proj_z (fp32 accum)
    d_ln_w_ptr,  # [DIM_Z]                output d_pair_norm_w (fp32 accum)
    d_ln_b_ptr,  # [DIM_Z]                output d_pair_norm_b (fp32 accum)
    Q,
    K,
    EPS: tl.constexpr,
    DIM_Z: tl.constexpr,
    DIM_Z_PAD: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS_PAD: tl.constexpr,
    TILE_K: tl.constexpr,
    AFFINE: tl.constexpr,
):
    """Backward for ``fused_pair_bias``.

    Grid: (cdiv(K, TILE_K), Q, B).

    Each CTA processes a (b, q, k_tile) slab and:

    1. Loads ``z[b,q,k,:]``, ``mean``, ``rstd``, ``d_bias[b,:,q,k]``,
       ``w_proj_z[:,:]``, ``gamma``, ``beta``.
    2. Computes ``z_hat = (z - mean) * rstd``, ``z_norm = z_hat * gamma + beta``.
    3. ``d_z_norm[k,c] = sum_h d_bias[h,k] * w_proj_z[h,c]``  (matmul, k×c).
    4. Atomic-add ``d_w_proj_z[h,c] += d_bias[k,h]^T @ z_norm[k,c]``.
    5. Atomic-add ``d_pair_norm_b[c] += sum_k d_z_norm[k,c]``.
    6. Atomic-add ``d_pair_norm_w[c] += sum_k d_z_norm[k,c] * z_hat[k,c]``.
    7. ``d_z_hat = d_z_norm * gamma``.
    8. LN bwd: ``d_z = (d_z_hat - mean_c(d_z_hat) - z_hat * mean_c(d_z_hat * z_hat)) * rstd``.
    9. Store ``d_z``.

    All math is fp32 internally; only loads/stores are bf16. Atomic adds
    target fp32 buffers (bf16 atomics are not supported on Hopper).
    """
    pid_k = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    offs_z = tl.arange(0, DIM_Z_PAD)
    offs_h = tl.arange(0, NUM_HEADS_PAD)
    mask_k = offs_k < K
    mask_z = offs_z < DIM_Z
    mask_h = offs_h < NUM_HEADS

    z_ptrs = (
        z_ptr
        + pid_b * Q * K * DIM_Z
        + pid_q * K * DIM_Z
        + offs_k[:, None] * DIM_Z
        + offs_z[None, :]
    )
    z = tl.load(z_ptrs, mask=mask_k[:, None] & mask_z[None, :], other=0.0).to(
        tl.float32
    )
    mean_ptrs = mean_ptr + pid_b * Q * K + pid_q * K + offs_k
    rstd_ptrs = rstd_ptr + pid_b * Q * K + pid_q * K + offs_k
    mean = tl.load(mean_ptrs, mask=mask_k, other=0.0)
    rstd = tl.load(rstd_ptrs, mask=mask_k, other=0.0)

    z_hat = (z - mean[:, None]) * rstd[:, None]
    z_hat = tl.where(mask_k[:, None] & mask_z[None, :], z_hat, 0.0)

    if AFFINE:
        gamma = tl.load(w_ln_ptr + offs_z, mask=mask_z, other=1.0).to(tl.float32)
        beta = tl.load(b_ln_ptr + offs_z, mask=mask_z, other=0.0).to(tl.float32)
    else:
        gamma = tl.full([DIM_Z_PAD], 1.0, dtype=tl.float32)
        beta = tl.full([DIM_Z_PAD], 0.0, dtype=tl.float32)
    # Recompute normalized output (needed for d_w_proj_z).
    z_norm = z_hat * gamma[None, :] + beta[None, :]
    z_norm = tl.where(mask_k[:, None] & mask_z[None, :], z_norm, 0.0)

    d_bias_ptrs = (
        d_bias_ptr
        + pid_b * NUM_HEADS * Q * K
        + offs_h[None, :] * Q * K
        + pid_q * K
        + offs_k[:, None]
    )
    d_bias = tl.load(d_bias_ptrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0).to(
        tl.float32
    )

    w_proj_ptrs = w_proj_z_ptr + offs_h[:, None] * DIM_Z + offs_z[None, :]
    w_proj = tl.load(w_proj_ptrs, mask=mask_h[:, None] & mask_z[None, :], other=0.0).to(
        tl.float32
    )

    d_z_norm = tl.dot(d_bias, w_proj, input_precision="tf32x3")
    d_z_norm = tl.where(mask_k[:, None] & mask_z[None, :], d_z_norm, 0.0)

    d_w = tl.dot(tl.trans(d_bias), z_norm, input_precision="tf32x3")
    d_w_ptrs = d_w_proj_z_ptr + offs_h[:, None] * DIM_Z + offs_z[None, :]
    tl.atomic_add(d_w_ptrs, d_w, mask=mask_h[:, None] & mask_z[None, :], sem="relaxed")

    if AFFINE:
        d_b_tile = tl.sum(d_z_norm, axis=0)
        tl.atomic_add(d_ln_b_ptr + offs_z, d_b_tile, mask=mask_z, sem="relaxed")

        d_w_tile = tl.sum(d_z_norm * z_hat, axis=0)
        tl.atomic_add(d_ln_w_ptr + offs_z, d_w_tile, mask=mask_z, sem="relaxed")

    d_z_hat = d_z_norm * gamma[None, :]
    d_z_hat = tl.where(mask_k[:, None] & mask_z[None, :], d_z_hat, 0.0)
    sum_dzh = tl.sum(d_z_hat, axis=1)
    sum_dzh_zhat = tl.sum(d_z_hat * z_hat, axis=1)
    mean_dzh = sum_dzh / DIM_Z
    mean_dzh_zhat = sum_dzh_zhat / DIM_Z
    d_z = (d_z_hat - mean_dzh[:, None] - z_hat * mean_dzh_zhat[:, None]) * rstd[:, None]
    d_z = tl.where(mask_k[:, None] & mask_z[None, :], d_z, 0.0)

    d_z_ptrs = (
        d_z_ptr
        + pid_b * Q * K * DIM_Z
        + pid_q * K * DIM_Z
        + offs_k[:, None] * DIM_Z
        + offs_z[None, :]
    )
    tl.store(
        d_z_ptrs,
        d_z.to(d_z_ptr.type.element_ty),
        mask=mask_k[:, None] & mask_z[None, :],
    )


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return max(p, 16)


def _round_up_heads_per_blk(num_heads: int) -> int:
    """Always return 16 — ``tl.dot`` requires M, N, K >= 16 on Hopper.

    For the AF3-style transformer ``num_heads=16``; ``HEADS_PER_BLK=16`` gives
    a single CTA per (b, q, k_tile) and is the cheapest schedule. Head-counts
    below 16 are zero-padded inside the kernel.
    """
    del num_heads  # head-tile is fixed at 16 by the tl.dot lower bound
    return 16


def _launch_forward(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    w_proj_z: torch.Tensor,
    pair_norm_w: torch.Tensor | None,
    pair_norm_b: torch.Tensor | None,
    num_heads: int,
    eps: float,
    inf: float,
    save_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Forward kernel launch helper. Returns ``(bias, mean, rstd)``.

    ``mean``/``rstd`` are ``None`` unless ``save_stats=True``.
    """
    assert z.dim() == 4, f"z must be (B,Q,K,DIM_Z); got {z.shape}"
    B, Q, K, DIM_Z = z.shape
    assert w_proj_z.shape == (
        num_heads,
        DIM_Z,
    ), f"w_proj_z {w_proj_z.shape} ≠ ({num_heads}, {DIM_Z})"

    z = z.contiguous()
    w_proj_z = w_proj_z.contiguous()
    affine = pair_norm_w is not None or pair_norm_b is not None
    if affine:
        if pair_norm_w is None:
            pair_norm_w = torch.ones(DIM_Z, device=z.device, dtype=z.dtype)
        if pair_norm_b is None:
            pair_norm_b = torch.zeros(DIM_Z, device=z.device, dtype=z.dtype)
        pair_norm_w = pair_norm_w.contiguous()
        pair_norm_b = pair_norm_b.contiguous()
    if mask is not None:
        mask = mask.contiguous()
        assert mask.shape == (B, K), f"mask {mask.shape} ≠ ({B}, {K})"

    out = torch.empty((B, num_heads, Q, K), device=z.device, dtype=z.dtype)
    if save_stats:
        mean = torch.empty((B, Q, K), device=z.device, dtype=torch.float32)
        rstd = torch.empty((B, Q, K), device=z.device, dtype=torch.float32)
    else:
        mean = None
        rstd = None
    heads_per_blk = _round_up_heads_per_blk(num_heads)
    DIM_Z_PAD = _next_pow2(DIM_Z)
    _dummy = torch.empty(1, device=z.device, dtype=z.dtype)
    _dummy_f32 = torch.empty(1, device=z.device, dtype=torch.float32)
    num_head_blks = (num_heads + heads_per_blk - 1) // heads_per_blk
    grid = lambda meta: (triton.cdiv(K, meta["TILE_K"]), Q, B * num_head_blks)
    _pair_bias_kernel[grid](
        z,
        mask if mask is not None else _dummy,
        w_proj_z,
        pair_norm_w if affine else _dummy,
        pair_norm_b if affine else _dummy,
        out,
        mean if save_stats else _dummy_f32,
        rstd if save_stats else _dummy_f32,
        B,
        Q,
        K,
        NEG_INF=-float(inf),
        EPS=eps,
        DIM_Z=DIM_Z,
        DIM_Z_PAD=DIM_Z_PAD,
        NUM_HEADS=num_heads,
        HEADS_PER_BLK=heads_per_blk,
        HAS_MASK=mask is not None,
        AFFINE=affine,
        SAVE_STATS=save_stats,
    )
    return out, mean, rstd


def _launch_backward(
    z: torch.Tensor,
    w_proj_z: torch.Tensor,
    pair_norm_w: torch.Tensor | None,
    pair_norm_b: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    d_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Backward kernel launch helper. Returns ``(d_z, d_w_proj_z, d_pair_norm_w, d_pair_norm_b)``.

    The latter two are ``None`` when affine is off.
    """
    B, Q, K, DIM_Z = z.shape
    NUM_HEADS = w_proj_z.shape[0]
    affine = pair_norm_w is not None
    if affine:
        assert pair_norm_b is not None

    z = z.contiguous()
    w_proj_z = w_proj_z.contiguous()
    d_bias = d_bias.contiguous()
    if affine:
        assert pair_norm_w is not None and pair_norm_b is not None
        pair_norm_w = pair_norm_w.contiguous()
        pair_norm_b = pair_norm_b.contiguous()

    d_z = torch.empty_like(z)
    # fp32 accumulators for atomic adds — bf16 atomic_add is not supported on Hopper
    d_w_proj_z_f32 = torch.zeros(
        (NUM_HEADS, DIM_Z), device=z.device, dtype=torch.float32
    )
    if affine:
        d_pair_norm_w_f32 = torch.zeros(DIM_Z, device=z.device, dtype=torch.float32)
        d_pair_norm_b_f32 = torch.zeros(DIM_Z, device=z.device, dtype=torch.float32)
    else:
        d_pair_norm_w_f32 = torch.zeros(1, device=z.device, dtype=torch.float32)
        d_pair_norm_b_f32 = torch.zeros(1, device=z.device, dtype=torch.float32)

    DIM_Z_PAD = _next_pow2(DIM_Z)
    # min 16 to satisfy Triton's tl.dot requirement (M, N, K >= 16 on Hopper).
    NUM_HEADS_PAD = max(16, _next_pow2(NUM_HEADS))
    _dummy = torch.empty(1, device=z.device, dtype=z.dtype)

    grid = lambda meta: (triton.cdiv(K, meta["TILE_K"]), Q, B)
    _pair_bias_backward_kernel[grid](
        z,
        w_proj_z,
        pair_norm_w if affine else _dummy,
        pair_norm_b if affine else _dummy,
        mean,
        rstd,
        d_bias,
        d_z,
        d_w_proj_z_f32,
        d_pair_norm_w_f32,
        d_pair_norm_b_f32,
        Q,
        K,
        EPS=eps,
        DIM_Z=DIM_Z,
        DIM_Z_PAD=DIM_Z_PAD,
        NUM_HEADS=NUM_HEADS,
        NUM_HEADS_PAD=NUM_HEADS_PAD,
        AFFINE=affine,
    )

    d_w_proj_z = d_w_proj_z_f32.to(w_proj_z.dtype)
    d_pair_norm_w: torch.Tensor | None
    d_pair_norm_b: torch.Tensor | None
    if affine:
        assert pair_norm_w is not None and pair_norm_b is not None
        d_pair_norm_w = d_pair_norm_w_f32.to(pair_norm_w.dtype)
        d_pair_norm_b = d_pair_norm_b_f32.to(pair_norm_b.dtype)
    else:
        d_pair_norm_w = None
        d_pair_norm_b = None

    return d_z, d_w_proj_z, d_pair_norm_w, d_pair_norm_b


class FusedPairBias(torch.autograd.Function):
    """Autograd wrapper around ``_pair_bias_kernel`` and ``_pair_bias_backward_kernel``.

    Forward saves ``(z, w_proj_z, pair_norm_w, pair_norm_b, mean, rstd)`` so that
    the backward kernel can recompute ``z_hat`` without re-doing the full
    LN reduction. ``mask`` is non-differentiable; we save it in ``ctx`` only as
    a marker (the kernel applies it inside the forward, but mask positions get
    -INF and so contribute 0 gradient through softmax → no special handling
    needed for ``d_bias``).
    """

    @staticmethod
    def forward(ctx, z, mask, w_proj_z, pair_norm_w, pair_norm_b, num_heads, eps, inf):
        out, mean, rstd = _launch_forward(
            z,
            mask,
            w_proj_z,
            pair_norm_w,
            pair_norm_b,
            num_heads,
            eps,
            inf,
            save_stats=True,
        )
        affine = pair_norm_w is not None or pair_norm_b is not None
        ctx.save_for_backward(
            z,
            w_proj_z,
            pair_norm_w if affine else None,
            pair_norm_b if affine else None,
            mean,
            rstd,
        )
        ctx.affine = affine
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, d_bias):
        z, w_proj_z, pair_norm_w, pair_norm_b, mean, rstd = ctx.saved_tensors
        d_z, d_w_proj_z, d_pair_norm_w, d_pair_norm_b = _launch_backward(
            z,
            w_proj_z,
            pair_norm_w,
            pair_norm_b,
            mean,
            rstd,
            d_bias.contiguous(),
            ctx.eps,
        )
        # Order must match forward args: (z, mask, w_proj_z, pair_norm_w, pair_norm_b, num_heads, eps, inf)
        return (
            d_z,
            None,  # mask
            d_w_proj_z,
            d_pair_norm_w,
            d_pair_norm_b,
            None,  # num_heads
            None,  # eps
            None,  # inf
        )


@torch._dynamo.disable
def fused_pair_bias(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    w_proj_z: torch.Tensor,
    pair_norm_w: torch.Tensor | None,
    pair_norm_b: torch.Tensor | None,
    *,
    num_heads: int,
    eps: float = 1e-5,
    inf: float = 1e6,
) -> torch.Tensor:
    """Compute ``bias[B, H, Q, K] = LN(z) @ w_proj_z.T + (1-mask)*-INF``.

    Dispatches to the autograd-aware ``FusedPairBias`` path when autograd is
    enabled (i.e. any input requires_grad and we are not in a no_grad/inference
    context). Otherwise falls back to the forward-only kernel.

    Parameters
    ----------
    z : (B, Q, K, DIM_Z)
    mask : (B, K) bool or None. True = keep, False = mask out (-INF added).
    w_proj_z : (num_heads, DIM_Z)
    pair_norm_w, pair_norm_b : (DIM_Z,) or None. Pass both or neither.
    num_heads : int
    eps, inf : LN epsilon and masking-infinity respectively.

    Returns
    -------
    bias : (B, num_heads, Q, K) — same dtype as z.
    """
    use_autograd = torch.is_grad_enabled() and (
        z.requires_grad
        or w_proj_z.requires_grad
        or (pair_norm_w is not None and pair_norm_w.requires_grad)
        or (pair_norm_b is not None and pair_norm_b.requires_grad)
    )
    if use_autograd:
        out_t: torch.Tensor = FusedPairBias.apply(  # type: ignore[assignment]
            z, mask, w_proj_z, pair_norm_w, pair_norm_b, num_heads, eps, inf
        )
        return out_t
    out, _, _ = _launch_forward(
        z,
        mask,
        w_proj_z,
        pair_norm_w,
        pair_norm_b,
        num_heads,
        eps,
        inf,
        save_stats=False,
    )
    return out


def fused_attention_pair_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor | None,
    mask: torch.Tensor | None,
    x_for_gate: torch.Tensor,
    *,
    w_proj_z: torch.Tensor | None,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    pair_norm_w: torch.Tensor | None = None,
    pair_norm_b: torch.Tensor | None = None,
    eps: float = 1e-5,
    inf: float = 1e6,
    precomputed_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-to-end fused AttentionPairBias forward (no-conditioning path).

    Pipeline:
        bias = fused_pair_bias(z, mask, w_proj_z, ln_w, ln_b)   # Triton
        attn = SDPA(q, k, v, attn_mask=bias)                    # cuDNN / Flash
        gate = sigmoid(linear(x_for_gate, w_proj_g))            # cuBLAS
        out  = linear(gate * attn, w_proj_o)                    # cuBLAS

    Parameters
    ----------
    q, k, v : (B, H, Q|K, D) — already-projected query/key/value, head-split.
    z : (B, Q, K, DIM_Z) — raw (unnormed) pair tensor. Ignored when
        ``precomputed_bias`` is supplied.
    mask : (B, K) bool or None. Ignored when ``precomputed_bias`` is supplied.
    x_for_gate : (B, Q, d_model) — pre-norm input for the gate
        ``sigmoid(x @ w_proj_g.T)``. In the production module this is the
        adaln/pre_norm output ``x`` (same tensor that feeds ``q``).
    w_proj_z : (H, DIM_Z). Ignored when ``precomputed_bias`` is supplied.
    w_proj_g, w_proj_o : (d_model, d_model)
    pair_norm_w, pair_norm_b : (DIM_Z,). Ignored when ``precomputed_bias`` is
        supplied.
    precomputed_bias : (B, H, Q, K) optional cached bias tensor. When provided
        the LN+proj+mask Triton kernel is skipped entirely — the bias is reused
        as the SDPA ``attn_mask`` directly. This is the 50× diffusion-step
        amortization path: ``z`` is constant within a loop, so the bias can
        be computed once and reused across all 50 denoise steps. Compute it
        with ``fused_pair_bias`` once, then pass it back here on every step.

    Returns
    -------
    out : (B, Q, d_model)
    """
    B, H, Q, D = q.shape
    d_model = H * D

    if precomputed_bias is not None:
        assert (
            not torch.is_grad_enabled()
        ), "precomputed_bias path is inference-only; autograd is not supported."
        bias = precomputed_bias
    else:
        if z is None or w_proj_z is None:
            raise ValueError(
                "Either precomputed_bias OR (z, w_proj_z) must be supplied"
            )
        bias = fused_pair_bias(
            z, mask, w_proj_z, pair_norm_w, pair_norm_b, num_heads=H, eps=eps, inf=inf
        )  # (B, H, Q, K)

    # Match the dtype expected by cuDNN attention. q/k/v are already bf16 in
    # inference; ``bias`` inherits z's dtype.
    if not torch.compiler.is_compiling():
        with torch.nn.attention.sdpa_kernel(
            backends=[
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ],
            set_priority=True,
        ):
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False
            )
    else:
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False
        )
    attn = attn.transpose(1, 2).contiguous().view(B, Q, d_model)

    gate = torch.sigmoid(torch.nn.functional.linear(x_for_gate, w_proj_g))
    out = torch.nn.functional.linear(gate * attn, w_proj_o)
    return out


__all__ = ["FusedPairBias", "fused_attention_pair_bias", "fused_pair_bias"]
