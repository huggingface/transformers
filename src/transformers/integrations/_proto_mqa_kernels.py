"""Prototype Triton kernel for the DSA indexer (WP prefill, bench only; candidate for a hub kernel).

`mqa_logits_masked`: the indexer's scores of a packed batch in one launch, without materialising the per-head scores:

    out[m, n] = sum_h w[m, h] * relu(q[m, h, :] . k[n, :])   if lo[m] <= n < hi[m]   else -inf

`[lo[m], hi[m])` is the range of packed keys query `m` may see (its own request, causal). A tile of queries x keys with
no visible pair writes -inf and skips its dot products, so the work follows sum over requests of queries x visible keys,
not M x N. Inputs are bf16, products are accumulated in fp32 (same values as the fp32 reference, another summation
order). The kernel takes no indices into other tensors: `lo` / `hi` only gate what is computed and written.
"""
import os

import torch
import triton
import triton.language as tl


# longctx WP: PROTO_MQA_ASSERTS=1 copies the index tensors of every launch to the CPU and checks every value the kernels
# read (window bounds, block table entries, request ids) before the launch. A device sync per call: checks only, and
# skipped while a CUDA graph is being captured (the eager pass before each capture runs them)
_ASSERTS = os.environ.get("PROTO_MQA_ASSERTS", "0") == "1"


def _check_windows(lo, hi, n, max_k):
    if not _ASSERTS or torch.cuda.is_current_stream_capturing():
        return
    lo, hi = lo.cpu().long(), hi.cpu().long()
    assert int(lo.min()) >= 0 and bool((hi >= lo).all()) and int(hi.max()) <= n, "window outside the packed keys"
    assert int((hi - lo).max()) <= max_k, f"a window is wider than max_k={max_k}"


def _check_paged(table, req, n_visible, num_rows, max_k, page):
    if not _ASSERTS or torch.cuda.is_current_stream_capturing():
        return
    table, req, n_visible = table.cpu().long(), req.cpu().long(), n_visible.cpu().long()
    assert int(req.min()) >= 0 and int(req.max()) < table.shape[0], "request id outside the block table"
    assert int(n_visible.min()) >= 0 and int(n_visible.max()) <= min(max_k, table.shape[1] * page), "context too long"
    slots = torch.arange(table.shape[1])
    used = slots[None] < ((n_visible + page - 1) // page)[:, None]  # [M, pages]: the table entries each query reads
    entries = table[req][used]
    assert int(entries.min(initial=0)) >= 0, "a read page is missing (-1) in the block table"
    assert int(entries.max(initial=0)) * page + page <= num_rows, "a page lies past the cache rows"


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
    # longctx WP: the kernel's output offsets are int32 (job 467381 faulted on B300 with M = 8,192 and N ~ 500k packed
    # keys): refuse shapes past that range before launching. `mqa_logits_window` has no [M, N] output
    assert m * n < 2**31, f"mqa_logits_masked: [{m}, {n}] output exceeds int32 offsets, use mqa_logits_window"
    out = torch.empty((m, n), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _mqa_logits_masked_kernel[grid](
        q, k, w, lo, hi, out, m, n, q.stride(0), q.stride(1), k.stride(0), w.stride(0), out.stride(0),
        H=h, D=d, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4, num_stages=2,
    )
    return out


# longctx WP (bench only): the indexer's scores of a block-table decode batch, read in place from the paged cache.
@triton.jit(do_not_specialize=["num_rows", "num_requests", "pages_per_request"])
def _paged_mqa_logits_kernel(
    q_ptr, w_ptr, k_ptr, table_ptr, req_ptr, n_visible_ptr, out_ptr, stride_qm, stride_qh, stride_wm, stride_kr,
    stride_tr, stride_om, num_rows, num_requests, pages_per_request,
    H: tl.constexpr, H_PAD: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)  # query token (fastest-varying, so the queries of one request read the same tile together)
    n0 = tl.program_id(1) * BLOCK_N  # first context offset of the tile
    offs_n = n0 + tl.arange(0, BLOCK_N)
    n_visible = tl.load(n_visible_ptr + m)
    # the tile lies in one page (BLOCK_N divides PAGE): one block table entry gives its rows. Every index is clamped,
    # so a wrong table or request id can only give a wrong result, never an out-of-bounds read
    req = tl.minimum(tl.maximum(tl.load(req_ptr + m), 0), num_requests - 1)
    slot = tl.minimum(n0 // PAGE, pages_per_request - 1)
    page = tl.load(table_ptr + req.to(tl.int64) * stride_tr + slot, mask=n0 < n_visible, other=-1)
    # a tile past the query's context (or on a missing page) reads nothing and writes -inf
    visible = (offs_n < n_visible) & (page >= 0)
    rows = page.to(tl.int64) * PAGE + (n0 % PAGE) + tl.arange(0, BLOCK_N)
    rows = tl.minimum(tl.maximum(rows, 0), num_rows - 1)
    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, H_PAD)
    k = tl.load(k_ptr + rows[None, :] * stride_kr + offs_d[:, None], mask=visible[None, :], other=0.0)  # [D, BLOCK_N]
    q = tl.load(q_ptr + m.to(tl.int64) * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :],
                mask=offs_h[:, None] < H, other=0.0)  # [H_PAD, D]
    w = tl.load(w_ptr + m.to(tl.int64) * stride_wm + offs_h, mask=offs_h < H, other=0.0)
    scores = tl.dot(q, k, out_dtype=tl.float32)  # [H_PAD, BLOCK_N], bf16 products accumulated in fp32
    logits = tl.sum(tl.maximum(scores, 0.0) * w[:, None], axis=0)
    logits = tl.where(visible, logits, float("-inf"))
    tl.store(out_ptr + m.to(tl.int64) * stride_om + offs_n, logits)


def paged_mqa_logits(
    q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor, table: torch.Tensor, req: torch.Tensor,
    n_visible: torch.Tensor, max_k: int, page: int, block_n: int = 128, num_warps: int = 8,
) -> torch.Tensor:
    """The DSA indexer's scores of a decode batch on CB's block table, without gathering the keys:

        out[m, o] = sum_h w[m, h] * relu(q[m, h, :] . keys[row(req[m], o), :])   for o < n_visible[m], else -inf

    with row(r, o) = table[r, o // page] * page + o % page, the layer's cache row of token `o` of request `r`.
    q [M, H, D] bf16, w [M, H] fp32 (scales folded in), keys [num_rows, D] bf16 (rows may be strided), table
    [R, pages] int32 (-1 = no page), req / n_visible [M] int32 -> [M, max_k] fp32. A tile past a query's context only
    writes -inf, so the reads follow the real context lengths, not the `max_k` bucket."""
    assert q.dim() == 3 and w.dim() == 2 and keys.dim() == 2 and table.dim() == 2
    m, h, d = q.shape
    assert tuple(w.shape) == (m, h) and keys.shape[1] == d and req.shape == (m,) and n_visible.shape == (m,)
    assert q.dtype == torch.bfloat16 and keys.dtype == torch.bfloat16 and w.dtype == torch.float32
    assert table.dtype == torch.int32 and req.dtype == torch.int32 and n_visible.dtype == torch.int32
    assert q.stride(2) == 1 and keys.stride(1) == 1 and w.stride(1) == 1 and table.stride(1) == 1
    assert req.is_contiguous() and n_visible.is_contiguous() and d in (64, 128, 256)
    assert page % block_n == 0 and max_k % block_n == 0 and m > 0 and max_k > 0
    _check_paged(table, req, n_visible, keys.shape[0], max_k, page)
    out = torch.empty((m, max_k), dtype=torch.float32, device=q.device)
    _paged_mqa_logits_kernel[(m, max_k // block_n)](
        q, w, keys, table, req, n_visible, out, q.stride(0), q.stride(1), w.stride(0), keys.stride(0), table.stride(0),
        out.stride(0), keys.shape[0], table.shape[0], table.shape[1],
        H=h, H_PAD=max(16, triton.next_power_of_2(h)), D=d, PAGE=page, BLOCK_N=block_n, num_warps=num_warps,
    )
    return out


# longctx WP (bench only): the indexer's scores of a packed (varlen) batch in each query's own window, no [M, N] output.
@triton.jit(do_not_specialize=["M", "N", "max_k"])
def _mqa_logits_window_kernel(
    q_ptr, k_ptr, w_ptr, lo_ptr, hi_ptr, out_ptr, M, N, max_k, stride_qm, stride_qh, stride_km, stride_wm, stride_om,
    H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)  # offsets in the query's window
    offs_d = tl.arange(0, D)
    mask_m = offs_m < M
    lo = tl.load(lo_ptr + offs_m, mask=mask_m, other=0)
    hi = tl.load(hi_ptr + offs_m, mask=mask_m, other=0)
    # clamp the windows to the packed keys, so a wrong lo / hi can only give a wrong result
    lo = tl.minimum(tl.maximum(lo, 0), N)
    hi = tl.minimum(tl.maximum(hi, lo), N)
    visible = (offs_o[None, :] < (hi - lo)[:, None]) & mask_m[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # the rows of a tile can belong to several requests (window starts): one pass per distinct start, each pass loads
    # the keys of that start once and keeps the rows that have it. A prefill tile is one or two passes
    pending = mask_m & ((hi - lo) > tl.program_id(1) * BLOCK_N)
    while tl.max(pending.to(tl.int32)) > 0:
        start = tl.min(tl.where(pending, lo, N))
        rows = pending & (lo == start)
        span = tl.max(tl.where(rows, hi - lo, 0))
        keys = (start + offs_o).to(tl.int64)
        k = tl.load(k_ptr + keys[None, :] * stride_km + offs_d[:, None], mask=offs_o[None, :] < span, other=0.0)
        part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for h in range(H):
            q = tl.load(q_ptr + offs_m[:, None].to(tl.int64) * stride_qm + h * stride_qh + offs_d[None, :],
                        mask=rows[:, None], other=0.0)
            w = tl.load(w_ptr + offs_m.to(tl.int64) * stride_wm + h, mask=rows, other=0.0)
            part += tl.maximum(tl.dot(q, k, out_dtype=tl.float32), 0.0) * w[:, None]
        acc = tl.where(rows[:, None], part, acc)
        pending = pending & (lo != start)
    acc = tl.where(visible, acc, float("-inf"))
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * stride_om + offs_o[None, :], acc,
             mask=mask_m[:, None] & (offs_o[None, :] < max_k))


def mqa_logits_window(
    q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, max_k: int,
    block_m: int = 64, block_n: int = 128,
) -> torch.Tensor:
    """The masked indexer scores of `mqa_logits_masked`, in each query's own window instead of the packed keys:

        out[m, o] = sum_h w[m, h] * relu(q[m, h, :] . k[lo[m] + o, :])   if o < hi[m] - lo[m]   else -inf

    q [M, H, D] bf16, k [N, D] bf16, w [M, H] fp32, lo / hi [M] int32 (hi - lo <= max_k) -> [M, max_k] fp32. The work
    and the output follow the windows (M x max_k), not M x N: a varlen batch that packs long contexts (decodes of long
    requests next to a prefill chunk) no longer builds a dense [M, N] matrix. Offsets are int64."""
    assert q.dim() == 3 and k.dim() == 2 and w.dim() == 2 and lo.dim() == 1 and hi.dim() == 1
    m, h, d = q.shape
    n = k.shape[0]
    assert k.shape[1] == d and tuple(w.shape) == (m, h) and lo.shape[0] == m and hi.shape[0] == m
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and w.dtype == torch.float32
    assert lo.dtype == torch.int32 and hi.dtype == torch.int32
    assert q.stride(2) == 1 and k.stride(1) == 1 and w.stride(1) == 1 and lo.is_contiguous() and hi.is_contiguous()
    assert d in (64, 128, 256) and m > 0 and n > 0 and max_k > 0 and n < 2**31
    _check_windows(lo, hi, n, max_k)
    out = torch.empty((m, max_k), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(max_k, block_n))
    _mqa_logits_window_kernel[grid](
        q, k, w, lo, hi, out, m, n, max_k, q.stride(0), q.stride(1), k.stride(0), w.stride(0), out.stride(0),
        H=h, D=d, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4, num_stages=2,
    )
    return out
