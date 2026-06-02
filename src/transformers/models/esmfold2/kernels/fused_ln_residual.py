"""Triton LayerNorm (bf16 IO, fp32 stats) with optional fused residual-add in
the *backward* pass.

Inspired by cuequivariance's ``layer_norm_transpose``
(https://docs.nvidia.com/cuda/cuequivariance/index.html); independently
re-implemented in Triton to add the bwd residual-link fusion.

Used by ``trimul_with_residual.triangle_multiplicative_update_with_residual``
for two LN calls:

  * Stage 1 LN: ``x_in = LN(pair)`` with layout ``bijd->bijd``. The downstream
    residual add ``out = residual + delta`` (where ``residual = pair``) flows
    ``grad_out`` straight back to ``grad_pair``. Fusing that add into LN's
    bwd kernel saves one full pair-tensor read + one full write (~250 MB at
    B=5 L=768 c_z=128 bf16). Exposed via :func:`fused_ln_with_residual_link`.

  * Stage 4 LN: ``x_out = LN(einsum(...))`` with layout ``dbij->bijd``. No
    residual to fuse — :func:`fused_ln_transpose` is the plain variant.

The residual-link fusion in the bwd pass is the motivation for shipping this:
the bwd kernel computes ``grad_x = LN_bwd(grad_y, x, w, mean, rstd)`` AND
optionally adds an external ``grad_residual`` tensor into ``grad_x`` in the
same pass — saves one HBM round-trip of the (M, D) tensor.

Forward + backward use a single static ``triton.Config`` each (runtime
autotune cold-start is unshippable for inference paths). The bwd pass
emits per-tile ``grad_w``/``grad_b`` fp32 partials reduced host-side; this
is the standard Triton LN-bwd idiom (see openai/triton tutorial 05).
"""

import torch
import triton
import triton.language as tl

# Layout enum: 0 == "bijd->bijd" (bnd contig), 1 == "dbij->bijd" (dbn contig in).
_LAYOUT_BND_BND = 0
_LAYOUT_DBN_BND = 1

_FWD_TILE_M = 64
_FWD_NUM_WARPS = 8
_FWD_NUM_STAGES = 2

_BWD_TILE_M = 64
_BWD_NUM_WARPS = 8
_BWD_NUM_STAGES = 2


@triton.jit
def _ln_fwd_kernel(
    x_ptr,  # input
    w_ptr,  # [D]
    b_ptr,  # [D]
    out_ptr,  # [M, D] contig output
    mean_ptr,  # [M] fp32
    rstd_ptr,  # [M] fp32
    M,
    D: tl.constexpr,
    EPS: tl.constexpr,
    LAYOUT: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    M64 = M.to(tl.int64)

    offs_m = pid * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    mask_m = offs_m < M64

    if LAYOUT == 0:
        x_ptrs = x_ptr + offs_m[:, None] * D + offs_d[None, :]
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    else:  # LAYOUT == 1: (D, M) contig
        x_ptrs = x_ptr + offs_d[None, :] * M64 + offs_m[:, None]
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=1) / D
    x_c = x - mean[:, None]
    var = tl.sum(x_c * x_c, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    x_hat = x_c * rstd[:, None]

    tl.store(mean_ptr + offs_m, mean, mask=mask_m)
    tl.store(rstd_ptr + offs_m, rstd, mask=mask_m)

    w = tl.load(w_ptr + offs_d).to(tl.float32)
    b = tl.load(b_ptr + offs_d).to(tl.float32)
    y = x_hat * w[None, :] + b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * D + offs_d[None, :]
    tl.store(out_ptrs, y.to(out_ptr.type.element_ty), mask=mask_m[:, None])


@triton.jit
def _ln_bwd_kernel(
    grad_y_ptr,  # [M, D] (always bnd; LN out is bnd)
    x_ptr,  # input — layout LAYOUT
    w_ptr,  # [D]
    mean_ptr,  # [M] fp32
    rstd_ptr,  # [M] fp32
    grad_x_ptr,  # output — layout LAYOUT (same as x)
    grad_w_partial_ptr,  # [num_tiles, D] fp32
    grad_b_partial_ptr,  # [num_tiles, D] fp32
    grad_residual_ptr,  # [M, D] in bnd layout; ignored if HAS_RESIDUAL=0
    M,
    D: tl.constexpr,
    LAYOUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    M64 = M.to(tl.int64)

    offs_m = pid * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    offs_d = tl.arange(0, D).to(tl.int64)
    mask_m = offs_m < M64

    grad_y_ptrs = grad_y_ptr + offs_m[:, None] * D + offs_d[None, :]
    grad_y = tl.load(grad_y_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    if LAYOUT == 0:
        x_ptrs = x_ptr + offs_m[:, None] * D + offs_d[None, :]
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    else:
        x_ptrs = x_ptr + offs_d[None, :] * M64 + offs_m[:, None]
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + offs_m, mask=mask_m, other=0.0)
    rstd = tl.load(rstd_ptr + offs_m, mask=mask_m, other=0.0)
    w = tl.load(w_ptr + offs_d).to(tl.float32)

    x_hat = (x - mean[:, None]) * rstd[:, None]
    wdy = grad_y * w[None, :]
    c1 = tl.sum(wdy, axis=1) / D
    c2 = tl.sum(wdy * x_hat, axis=1) / D
    grad_x = (wdy - (c1[:, None] + x_hat * c2[:, None])) * rstd[:, None]

    if HAS_RESIDUAL:
        gr_ptrs = grad_residual_ptr + offs_m[:, None] * D + offs_d[None, :]
        gr = tl.load(gr_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        grad_x = grad_x + gr

    if LAYOUT == 0:
        gx_ptrs = grad_x_ptr + offs_m[:, None] * D + offs_d[None, :]
        tl.store(gx_ptrs, grad_x.to(grad_x_ptr.type.element_ty), mask=mask_m[:, None])
    else:
        gx_ptrs = grad_x_ptr + offs_d[None, :] * M64 + offs_m[:, None]
        tl.store(gx_ptrs, grad_x.to(grad_x_ptr.type.element_ty), mask=mask_m[:, None])

    # Per-tile partial reduction → write to (num_tiles, D) fp32 buffer.
    # Final reduction (sum over num_tiles) happens host-side as a torch.sum
    # (standard Triton LN-bwd idiom; see openai/triton tutorial 05).
    mask_f = mask_m[:, None].to(tl.float32)
    dw_tile = tl.sum(grad_y * x_hat * mask_f, axis=0)
    db_tile = tl.sum(grad_y * mask_f, axis=0)
    dw_offs = pid * D + offs_d
    db_offs = pid * D + offs_d
    tl.store(grad_w_partial_ptr + dw_offs, dw_tile)
    tl.store(grad_b_partial_ptr + db_offs, db_tile)


def _layout_to_int(layout: str) -> int:
    if layout in ("bijd->bijd", "bnd->bnd"):
        return _LAYOUT_BND_BND
    if layout in ("dbij->bijd", "dbn->bnd"):
        return _LAYOUT_DBN_BND
    raise ValueError(f"unsupported layout {layout!r}")


def _reshape_for_layout(
    x: torch.Tensor, layout: str
) -> tuple[tuple[int, ...], int, int, torch.Tensor]:
    """Reshape x to 2D LN view. Returns (out_shape, M, D, x_view)."""
    if layout == "bijd->bijd":
        B, II, J, D = x.shape
        M = B * II * J
        return (B, II, J, D), M, D, x.contiguous().view(M, D)
    if layout == "bnd->bnd":
        B, N, D = x.shape
        M = B * N
        return (B, N, D), M, D, x.contiguous().view(M, D)
    if layout == "dbij->bijd":
        D, B, II, J = x.shape
        M = B * II * J
        return (B, II, J, D), M, D, x.contiguous().view(D, M)
    if layout == "dbn->bnd":
        D, B, N = x.shape
        M = B * N
        return (B, N, D), M, D, x.contiguous().view(D, M)
    raise ValueError(f"unsupported layout {layout!r}")


def _ln_fwd(
    x_view: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    layout_int: int,
    M: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty((M, D), device=x_view.device, dtype=x_view.dtype)
    mean = torch.empty((M,), device=x_view.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x_view.device, dtype=torch.float32)
    grid = (triton.cdiv(M, _FWD_TILE_M),)
    _ln_fwd_kernel[grid](
        x_view,
        w,
        b,
        out,
        mean,
        rstd,
        M,
        D=D,
        EPS=eps,
        LAYOUT=layout_int,
        TILE_M=_FWD_TILE_M,
        num_warps=_FWD_NUM_WARPS,  # type: ignore[call-arg]
        num_stages=_FWD_NUM_STAGES,  # type: ignore[call-arg]
    )
    return out, mean, rstd


def _ln_bwd(
    grad_y: torch.Tensor,
    x_view: torch.Tensor,
    w: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    layout_int: int,
    grad_residual: torch.Tensor | None,
    M: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout_int == _LAYOUT_BND_BND:
        grad_x = torch.empty((M, D), device=x_view.device, dtype=x_view.dtype)
    else:
        grad_x = torch.empty((D, M), device=x_view.device, dtype=x_view.dtype)

    num_tiles = triton.cdiv(M, _BWD_TILE_M)
    grad_w_partial = torch.empty(
        (num_tiles, D), device=x_view.device, dtype=torch.float32
    )
    grad_b_partial = torch.empty(
        (num_tiles, D), device=x_view.device, dtype=torch.float32
    )

    has_residual = grad_residual is not None
    _dummy = torch.empty((), device=x_view.device, dtype=grad_y.dtype)
    grid = (num_tiles,)
    _ln_bwd_kernel[grid](
        grad_y,
        x_view,
        w,
        mean,
        rstd,
        grad_x,
        grad_w_partial,
        grad_b_partial,
        grad_residual if has_residual else _dummy,
        M,
        D=D,
        LAYOUT=layout_int,
        HAS_RESIDUAL=has_residual,
        TILE_M=_BWD_TILE_M,
        num_warps=_BWD_NUM_WARPS,  # type: ignore[call-arg]
        num_stages=_BWD_NUM_STAGES,  # type: ignore[call-arg]
    )

    grad_w = grad_w_partial.sum(dim=0).to(w.dtype)
    grad_b = grad_b_partial.sum(dim=0).to(w.dtype)
    return grad_x, grad_w, grad_b


class _LayerNormTransposeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, layout: str
    ) -> torch.Tensor:
        layout_int = _layout_to_int(layout)
        out_shape, M, D, x_view = _reshape_for_layout(x, layout)
        out_bnd, mean, rstd = _ln_fwd(x_view, w, b, eps, layout_int, M, D)
        ctx.save_for_backward(x_view, w, mean, rstd)
        ctx.layout_int = layout_int
        ctx.M = M
        ctx.D = D
        ctx.x_orig_shape = x.shape
        return out_bnd.view(*out_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x_view, w, mean, rstd = ctx.saved_tensors
        grad_y = grad_out.contiguous().view(ctx.M, ctx.D)
        grad_x, grad_w, grad_b = _ln_bwd(
            grad_y, x_view, w, mean, rstd, ctx.layout_int, None, ctx.M, ctx.D
        )
        grad_x = grad_x.view(*ctx.x_orig_shape)
        return grad_x, grad_w, grad_b, None, None


def fused_ln_transpose(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-5,
    layout: str = "bijd->bijd",
) -> torch.Tensor:
    """Plain LN replacement for ``layer_norm_transpose`` (no residual fusion)."""
    return _LayerNormTransposeFn.apply(x, w, b, eps, layout)  # type: ignore[return-value]


# Stage-1 LN with residual-add folded into the bwd kernel. The Function returns
# (ln_out, residual_alias); downstream uses ln_out, stage-5 uses residual_alias
# as the residual input. In bwd we receive both grad tensors and fold
# grad_residual_alias into grad_x in-kernel (saves one HBM round-trip).
# residual_alias is a fresh tensor (not a view) so autograd reliably routes
# its grad back through this Function.


class _LayerNormWithResidualLinkFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        residual_link: torch.Tensor,
        eps: float,
        layout: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout_int = _layout_to_int(layout)
        out_shape, M, D, x_view = _reshape_for_layout(x, layout)
        out_bnd, mean, rstd = _ln_fwd(x_view, w, b, eps, layout_int, M, D)
        ctx.save_for_backward(x_view, w, mean, rstd)
        ctx.layout_int = layout_int
        ctx.M = M
        ctx.D = D
        ctx.x_orig_shape = x.shape
        ctx.residual_link_shape = residual_link.shape

        ln_out = out_bnd.view(*out_shape)
        # residual_alias: returning ``residual_link`` itself works in
        # custom Functions — autograd treats the input-as-output case
        # correctly (sums grads back into the input). The downstream
        # graph node ``out = residual_alias + delta`` sees a regular
        # tensor and produces grad of shape == residual_link.
        # Note: we use .view_as for safety so the AutogradMeta is fresh.
        return ln_out, residual_link.view_as(residual_link)

    @staticmethod
    def backward(ctx, grad_ln_out: torch.Tensor, grad_link_pass: torch.Tensor):  # type: ignore[override]
        x_view, w, mean, rstd = ctx.saved_tensors
        grad_y = grad_ln_out.contiguous().view(ctx.M, ctx.D)

        # grad_link_pass shape == residual_link shape.
        if grad_link_pass is None:
            grad_residual = None
        else:
            grad_residual = grad_link_pass.contiguous().view(ctx.M, ctx.D)

        grad_x, grad_w, grad_b = _ln_bwd(
            grad_y, x_view, w, mean, rstd, ctx.layout_int, grad_residual, ctx.M, ctx.D
        )
        grad_x = grad_x.view(*ctx.x_orig_shape)

        # We folded grad_link_pass into grad_x → return None for that input's
        # grad slot, since we've already accounted for it via x's grad path
        # (the caller is expected to pass x == residual_link, so grad_x +=
        # grad_residual is exactly the combined grad on the shared leaf).
        # If caller passes DIFFERENT tensors for x and residual_link, the
        # link's grad is *lost* — that's a usage error we accept by contract.
        return grad_x, grad_w, grad_b, None, None, None


def fused_ln_with_residual_link(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    residual_link: torch.Tensor,
    eps: float = 1e-5,
    layout: str = "bijd->bijd",
) -> tuple[torch.Tensor, torch.Tensor]:
    """LN(x) with a residual-link passthrough that fuses
    ``grad_residual_link`` into the LN backward kernel.

    Contract: ``x`` and ``residual_link`` MUST refer to the same leaf
    tensor (same identity). The returned ``residual_alias`` MUST be used as
    the residual input to the downstream add — otherwise the fusion routes
    the grad incorrectly.

    Returns ``(ln_out, residual_alias)``.
    """
    if x is not residual_link:
        raise ValueError(
            "fused_ln_with_residual_link requires x and residual_link to be the "
            "same tensor instance (the caller must wire pair → both)."
        )
    return _LayerNormWithResidualLinkFn.apply(x, w, b, residual_link, eps, layout)  # type: ignore[return-value]
