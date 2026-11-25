# [scratch] Absorbed MLA attention for continuous batching (DeepSeek-V2 / V3). The cache holds one latent row per token
# (kv_lora_rank compressed KV, then the rotated key); instead of expanding every cached row to per-head keys and values
# (`kv_b_proj` over the whole context, every step), the queries are absorbed into the latent space and attention runs
# on the latent rows directly, as a single-KV-head attention with 576-dim keys and 512-dim values:
#   q_latent = q_nope @ W_UK  (per head)      scores = q_latent . kv_latent + q_rope . k_rope
#   out = softmax(scores) @ kv_latent, then @ W_UV (per head) before o_proj
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_varlen_kernel(
    Q, KV, ReadIndex, Out, CuSeqlensQ, CuSeqlensK, scale,
    stride_q_t, stride_q_h, stride_kv_t, stride_o_t, stride_o_h,
    NUM_HEADS: tl.constexpr, D_LATENT: tl.constexpr, D_ROPE: tl.constexpr, HAS_INDEX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):  # fmt: skip
    """One program per (sequence, block of BLOCK_M (query token, head) rows). Rows attend causally, bottom-right
    aligned: the query token i of a sequence with q_len new tokens and kv_len keys sees the keys 0 .. kv_len - q_len + i."""
    seq = tl.program_id(0)
    q_start = tl.load(CuSeqlensQ + seq)
    q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
    k_start = tl.load(CuSeqlensK + seq)
    k_len = tl.load(CuSeqlensK + seq + 1) - k_start
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    token = rows // NUM_HEADS
    head = rows % NUM_HEADS
    row_mask = token < q_len
    if tl.program_id(1) * BLOCK_M >= q_len * NUM_HEADS:
        return
    limit = k_len - q_len + token  # last key each row sees
    offs_l = tl.arange(0, D_LATENT)
    offs_r = tl.arange(0, D_ROPE)
    q_base = Q + (q_start + token).to(tl.int64)[:, None] * stride_q_t + head[:, None] * stride_q_h
    q_lat = tl.load(q_base + offs_l[None, :], mask=row_mask[:, None], other=0.0)
    q_rope = tl.load(q_base + D_LATENT + offs_r[None, :], mask=row_mask[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D_LATENT), dtype=tl.float32)
    max_key = tl.max(tl.where(row_mask, limit, -1)) + 1
    for n0 in range(0, max_key, BLOCK_N):
        keys = n0 + tl.arange(0, BLOCK_N)
        key_mask = keys < k_len
        if HAS_INDEX:  # the rows of the sequence are read in place in the cache, at their physical indices
            kv_rows = tl.load(ReadIndex + k_start + keys, mask=key_mask, other=0)
        else:
            kv_rows = k_start + keys
        kv_base = KV + kv_rows.to(tl.int64)[:, None] * stride_kv_t
        k_lat = tl.load(kv_base + offs_l[None, :], mask=key_mask[:, None], other=0.0)  # (BLOCK_N, D_LATENT)
        k_rope = tl.load(kv_base + D_LATENT + offs_r[None, :], mask=key_mask[:, None], other=0.0)
        s = tl.dot(q_lat, tl.trans(k_lat)) + tl.dot(q_rope, tl.trans(k_rope))
        s = s * scale
        s = tl.where((keys[None, :] <= limit[:, None]) & key_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k_lat.dtype), k_lat)
        m_i = m_new
    acc = acc / l_i[:, None]
    o_ptrs = Out + (q_start + token).to(tl.int64)[:, None] * stride_o_t + head[:, None] * stride_o_h + offs_l[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=row_mask[:, None])


def mla_varlen_attention(q, kv, read_index, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale, d_latent):
    """`q` [tokens, heads, d_latent + d_rope] absorbed queries, `kv` [rows, d_latent + d_rope] latent rows: the keys of
    the sequences are the rows `read_index[cu_seqlens_k[b]:cu_seqlens_k[b + 1]]`, or the rows `cu_seqlens_k[b]` onward
    when `read_index` is None. Returns the latent output [tokens, heads, d_latent]."""
    num_tokens, num_heads, d_qk = q.shape
    out = torch.empty(num_tokens, num_heads, d_latent, dtype=q.dtype, device=q.device)
    num_seqs = cu_seqlens_q.shape[0] - 1
    block_m, block_n = 64, 32
    grid = (num_seqs, triton.cdiv(max_seqlen_q * num_heads, block_m))
    _mla_varlen_kernel[grid](
        q, kv, kv if read_index is None else read_index, out, cu_seqlens_q, cu_seqlens_k, scale,
        q.stride(0), q.stride(1), kv.stride(0), out.stride(0), out.stride(1),
        NUM_HEADS=num_heads, D_LATENT=d_latent, D_ROPE=d_qk - d_latent, HAS_INDEX=read_index is not None,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=2,
    )  # fmt: skip
    return out



def absorbed_mla_forward(attn, q_nope, q_rope, kv_nope, k_rope, cache, kwargs):
    """A continuous batching step of DeepSeek-V2 / V3 attention on the cached latents: `q_nope` [1, heads, tokens,
    nope], `q_rope` [1, heads, tokens, rope], and the new latents `kv_nope` [1, 1, tokens, kv_lora_rank], `k_rope`
    [1, 1, tokens, rope]. Writes the new latents to the cache like `cache.update`, then attends on the cache rows in
    place (no gathered copy). Returns the attention output before `o_proj`, [1, tokens, heads * v_head_dim]."""
    from torch.distributed.tensor import DTensor

    allocator = cache.layer_to_allocator[attn.layer_idx]
    latent_cache, rope_cache = allocator._latent_token_views[attn.layer_idx]
    write_index = kwargs["write_index"][allocator.index]
    read_index = kwargs["read_index"][allocator.index]
    latent_cache.index_copy_(0, write_index, kv_nope.flatten(0, 2))
    rope_cache.index_copy_(0, write_index, k_rope.flatten(0, 2))
    cu_seqlens_k = kwargs["cu_seq_lens_k"][allocator.layer_type].to(torch.int32)
    if read_index.numel() == 0:  # nothing cached to read: the new tokens are the whole sequences
        kv, read_index = torch.cat([kv_nope[0, 0], k_rope[0, 0]], dim=-1), None
    else:
        kv = latent_cache.as_strided((latent_cache.shape[0], allocator.row_numel), (latent_cache.stride(0), 1))

    num_heads = q_nope.shape[1]
    weight = attn.kv_b_proj.weight
    weight = weight.to_local() if isinstance(weight, DTensor) else weight  # tensor parallel: this rank's heads
    weight = weight.view(num_heads, attn.qk_nope_head_dim + attn.v_head_dim, attn.kv_lora_rank)
    w_uk, w_uv = weight[:, : attn.qk_nope_head_dim], weight[:, attn.qk_nope_head_dim :]
    q = torch.cat([torch.einsum("htn,hnc->thc", q_nope[0], w_uk), q_rope[0].transpose(0, 1)], dim=-1)
    out = mla_varlen_attention(
        q, kv, read_index, kwargs["cu_seq_lens_q"], cu_seqlens_k, kwargs["max_length_q"], attn.scaling,
        attn.kv_lora_rank,
    )  # fmt: skip
    return torch.einsum("thc,hvc->thv", out, w_uv).reshape(1, out.shape[0], -1)
