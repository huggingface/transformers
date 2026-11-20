"""Prototype Triton kernel for the DSA indexer (WP prefill, bench only; candidate for a hub kernel).

`mqa_logits_masked`: the indexer's scores of a packed batch in one launch, without materialising the per-head scores:

    out[m, n] = sum_h w[m, h] * relu(q[m, h, :] . k[n, :])   if lo[m] <= n < hi[m]   else -inf

`[lo[m], hi[m])` is the range of packed keys query `m` may see (its own request, causal). A tile of queries x keys with
no visible pair writes -inf and skips its dot products, so the work follows sum over requests of queries x visible keys,
not M x N. Inputs are bf16, products are accumulated in fp32 (same values as the fp32 reference, another summation
order). The kernel takes no indices into other tensors: `lo` / `hi` only gate what is computed and written.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mqa_logits_masked_kernel(
    q_ptr, k_ptr, w_ptr, lo_ptr, hi_ptr, out_ptr, M, N, stride_qm, stride_qh, stride_km, stride_wm, stride_om,
    H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < M
    mask_n = offs_n < N
    lo = tl.load(lo_ptr + offs_m, mask=mask_m, other=0)
    hi = tl.load(hi_ptr + offs_m, mask=mask_m, other=0)
    visible = (offs_n[None, :] >= lo[:, None]) & (offs_n[None, :] < hi[:, None]) & mask_n[None, :] & mask_m[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if tl.max(visible.to(tl.int32)) > 0:
        # keys of the tile, loaded transposed: [D, BLOCK_N]
        k = tl.load(k_ptr + offs_n[None, :] * stride_km + offs_d[:, None], mask=mask_n[None, :], other=0.0)
        for h in range(H):
            q = tl.load(
                q_ptr + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :], mask=mask_m[:, None], other=0.0
            )
            w = tl.load(w_ptr + offs_m * stride_wm + h, mask=mask_m, other=0.0)
            acc += tl.maximum(tl.dot(q, k, out_dtype=tl.float32), 0.0) * w[:, None]
    acc = tl.where(visible, acc, float("-inf"))
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def mqa_logits_masked(
    q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, block_m: int = 64,
    block_n: int = 128,
) -> torch.Tensor:
    """q [M, H, D] bf16, k [N, D] bf16, w [M, H] fp32, lo / hi [M] int32 -> [M, N] fp32 masked indexer scores."""
    assert q.dim() == 3 and k.dim() == 2 and w.dim() == 2 and lo.dim() == 1 and hi.dim() == 1
    m, h, d = q.shape
    n = k.shape[0]
    assert k.shape[1] == d and tuple(w.shape) == (m, h) and lo.shape[0] == m and hi.shape[0] == m
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and w.dtype == torch.float32
    assert lo.dtype == torch.int32 and hi.dtype == torch.int32
    assert q.stride(2) == 1 and k.stride(1) == 1 and w.stride(1) == 1 and lo.is_contiguous() and hi.is_contiguous()
    assert d in (64, 128, 256) and m > 0 and n > 0
    out = torch.empty((m, n), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _mqa_logits_masked_kernel[grid](
        q, k, w, lo, hi, out, m, n, q.stride(0), q.stride(1), k.stride(0), w.stride(0), out.stride(0),
        H=h, D=d, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4, num_stages=2,
    )
    return out
