# [scratch] Writes the new keys and values of a varlen step into the paged cache before FA3 runs, so FA3 attends over
# the cache alone (its append path costs 7-11 us more per call at decode than this write plus a plain call).
import os

import torch
import triton
import triton.language as tl


PROTO_KV_WRITE = os.environ.get("PROTO_KV_WRITE", "1") == "1"


@triton.jit
def _paged_kv_write_kernel(
    K, V, KCache, VCache, BlockTable, CacheSeqlens, CuSeqlensQ, SeqUsed,
    stride_k, stride_kh, stride_v, stride_vh, stride_page, stride_slot, stride_bt,
    PAGE: tl.constexpr, ROW: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr,
):  # fmt: skip
    seq = tl.program_id(0)
    start = tl.load(CuSeqlensQ + seq)
    num_new = tl.load(CuSeqlensQ + seq + 1) - start
    past = tl.load(CacheSeqlens + seq)
    if tl.program_id(1) == 0:
        tl.store(SeqUsed + seq, past + num_new)
    j = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    if tl.program_id(1) * BLOCK_T >= num_new:
        return
    mask = j < num_new
    pos = past + j
    page = tl.load(BlockTable + seq * stride_bt + pos // PAGE, mask=mask, other=0).to(tl.int64)
    dst = (page * stride_page + (pos % PAGE) * stride_slot)[:, None] + tl.arange(0, ROW)[None, :]
    tokens = (start + j).to(tl.int64)
    cols = tl.arange(0, ROW)[None, :]
    head, dim = cols // HEAD_DIM, cols % HEAD_DIM
    k_src = K + tokens[:, None] * stride_k + head * stride_kh + dim
    v_src = V + tokens[:, None] * stride_v + head * stride_vh + dim
    tl.store(KCache + dst, tl.load(k_src, mask=mask[:, None]), mask=mask[:, None])
    tl.store(VCache + dst, tl.load(v_src, mask=mask[:, None]), mask=mask[:, None])


@torch.compiler.disable  # inductor lowers the symbolic strides of the K/V views into a 3-argument mul
def paged_kv_write(
    k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor,
    cache_seqlens: torch.Tensor, cu_seqlens_q: torch.Tensor, max_seqlen_q: int,
) -> torch.Tensor:  # fmt: skip
    """Writes `k`/`v` ([tokens, kv_heads, head_dim], head_dim contiguous) of each sequence after its `cache_seqlens`
    tokens in its pages of `k_cache`/`v_cache` ([pages, page_size, kv_heads, head_dim]). Returns the sequence lengths
    with the new tokens, for FA3's `cache_seqlens`."""
    num_seqs = cache_seqlens.shape[0]
    row = k.shape[-2] * k.shape[-1]
    seqused = torch.empty_like(cache_seqlens)
    block_t = 16
    _paged_kv_write_kernel[(num_seqs, triton.cdiv(max_seqlen_q, block_t))](
        k, v, k_cache, v_cache, block_table, cache_seqlens, cu_seqlens_q, seqused,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1), k_cache.stride(0), k_cache.stride(1), block_table.stride(0),
        PAGE=k_cache.shape[1], ROW=row, HEAD_DIM=k.shape[-1], BLOCK_T=block_t, num_warps=4,
    )  # fmt: skip
    return seqused

