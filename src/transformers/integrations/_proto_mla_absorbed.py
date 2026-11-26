# [scratch] Absorbed MLA attention for continuous batching (DeepSeek-V2 / V3). The cache holds one latent row per token
# (kv_lora_rank compressed KV, then the rotated key); instead of expanding every cached row to per-head keys and values
# (`kv_b_proj` over the whole context, every step), the queries are absorbed into the latent space and attention runs
# on the latent rows directly, as a single-KV-head attention with 576-dim keys and 512-dim values:
#   q_latent = q_nope @ W_UK  (per head)      scores = q_latent . kv_latent + q_rope . k_rope
#   out = softmax(scores) @ kv_latent, then @ W_UV (per head) before o_proj
import importlib.util
import itertools
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_varlen_kernel(
    Q, KV, ReadIndex, Out, PartialOut, PartialLse, CuSeqlensQ, CuSeqlensK, scale,
    stride_q_t, stride_q_h, stride_kv_t, stride_o_t, stride_o_h, num_rows, stride_bt,
    NUM_HEADS: tl.constexpr, D_LATENT: tl.constexpr, D_ROPE: tl.constexpr, D_LATENT_P: tl.constexpr,
    D_ROPE_P: tl.constexpr, HAS_INDEX: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, HAS_BLOCK_TABLE: tl.constexpr, PAGE: tl.constexpr,
):  # fmt: skip
    """One program per (sequence, block of BLOCK_M (query token, head) rows, split of the keys). Rows attend causally,
    bottom-right aligned: the query token i of a sequence with q_len new tokens and kv_len keys sees the keys
    0 .. kv_len - q_len + i. With several splits, each writes its normalized partial output and log-sum-exp, which
    `_mla_combine_kernel` merges (few rows at decode: the splits spread the keys over the SMs)."""
    seq = tl.program_id(0)
    split = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + seq)
    q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
    k_start = tl.load(CuSeqlensK + seq)
    k_len = tl.load(CuSeqlensK + seq + 1) - k_start
    # the head dims padded to powers of 2 (at least 16, for the dots), the padding columns masked to 0
    offs_l = tl.arange(0, D_LATENT_P)
    offs_r = tl.arange(0, D_ROPE_P)
    lat_mask = offs_l < D_LATENT
    rope_mask = offs_r < D_ROPE
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    if tl.program_id(1) * BLOCK_M >= q_len * NUM_HEADS:
        return
    token = rows // NUM_HEADS
    head = rows % NUM_HEADS
    row_mask = token < q_len
    limit = k_len - q_len + token  # last key each row sees
    q_base = Q + (q_start + token).to(tl.int64)[:, None] * stride_q_t + head[:, None] * stride_q_h
    q_lat = tl.load(q_base + offs_l[None, :], mask=row_mask[:, None] & lat_mask[None, :], other=0.0)
    q_rope = tl.load(q_base + D_LATENT + offs_r[None, :], mask=row_mask[:, None] & rope_mask[None, :], other=0.0)
    # finite start so a row that sees no key of this split keeps p = 0 and alpha = 1 (no inf - inf)
    m_i = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D_LATENT_P), dtype=tl.float32)
    max_key = tl.max(tl.where(row_mask, limit, -1)) + 1
    lo, hi = 0, max_key
    if NUM_SPLITS > 1:
        chunk = tl.cdiv(tl.cdiv(max_key, NUM_SPLITS), BLOCK_N) * BLOCK_N
        lo = split * chunk
        hi = tl.minimum(lo + chunk, max_key)
    for n0 in range(lo, hi, BLOCK_N):
        keys = n0 + tl.arange(0, BLOCK_N)
        key_mask = keys < k_len
        if HAS_BLOCK_TABLE:  # key `o` of the sequence is at row `table[seq, o // PAGE] * PAGE + o % PAGE`
            pages = tl.load(ReadIndex + seq * stride_bt + keys // PAGE, mask=key_mask, other=0)
            kv_rows = pages.to(tl.int64) * PAGE + keys % PAGE
        elif HAS_INDEX:  # the rows of the sequence are read in place in the cache, at their physical indices
            kv_rows = tl.load(ReadIndex + k_start + keys, mask=key_mask, other=0)
        else:
            kv_rows = k_start + keys
        kv_base = KV + kv_rows.to(tl.int64)[:, None] * stride_kv_t
        k_lat = tl.load(kv_base + offs_l[None, :], mask=key_mask[:, None] & lat_mask[None, :], other=0.0)
        k_rope = tl.load(kv_base + D_LATENT + offs_r[None, :], mask=key_mask[:, None] & rope_mask[None, :], other=0.0)
        s = tl.dot(q_lat, tl.trans(k_lat)) + tl.dot(q_rope, tl.trans(k_rope))
        s = (s * scale).to(tl.float32)  # under torch.compile, `scale` comes in as an fp64 scalar
        s = tl.where((keys[None, :] <= limit[:, None]) & key_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k_lat.dtype), k_lat)
        m_i = m_new
    acc = acc / tl.maximum(l_i, 1.0e-30)[:, None]
    if NUM_SPLITS == 1:
        o_ptrs = Out + (q_start + token).to(tl.int64)[:, None] * stride_o_t + head[:, None] * stride_o_h
        o_mask = row_mask[:, None] & lat_mask[None, :]
        tl.store(o_ptrs + offs_l[None, :], acc.to(Out.dtype.element_ty), mask=o_mask)
    else:  # partial outputs [splits, tokens * heads, D_LATENT_P] and log-sum-exps [splits, tokens * heads], fp32
        flat = (q_start + token).to(tl.int64) * NUM_HEADS + head
        base = split.to(tl.int64) * num_rows + flat
        tl.store(PartialOut + base[:, None] * D_LATENT_P + offs_l[None, :], acc, mask=row_mask[:, None])
        tl.store(PartialLse + base, m_i + tl.log(l_i), mask=row_mask)


@triton.jit
def _mla_combine_kernel(
    PartialOut, PartialLse, Out, CuSeqlensQ, num_seqs, NUM_HEADS: tl.constexpr, D_LATENT: tl.constexpr,
    D_LATENT_P: tl.constexpr, NUM_SPLITS: tl.constexpr,
):  # fmt: skip
    """One program per (token, head) row: the softmax-weighted sum of the splits' partial outputs. The rows of the
    padding tokens (after the last sequence's) are skipped."""
    row = tl.program_id(0).to(tl.int64)
    num_rows = tl.num_programs(0).to(tl.int64)
    if row >= tl.load(CuSeqlensQ + num_seqs).to(tl.int64) * NUM_HEADS:
        return
    splits = tl.arange(0, NUM_SPLITS)
    lse = tl.load(PartialLse + splits * num_rows + row)
    weights = tl.exp(lse - tl.max(lse, 0))
    weights = weights / tl.sum(weights, 0)
    offs_l = tl.arange(0, D_LATENT_P)
    parts = tl.load(PartialOut + (splits[:, None] * num_rows + row) * D_LATENT_P + offs_l[None, :])
    out = tl.sum(parts * weights[:, None], 0)
    tl.store(Out + row * D_LATENT + offs_l, out.to(Out.dtype.element_ty), mask=offs_l < D_LATENT)


_NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
# Blocks of 64 rows fault on Blackwell (misaligned address, B300, any key block and warp count): 32 rows there
_PREFILL_BLOCK_M = 32 if torch.cuda.get_device_capability()[0] == 10 else 64


def mla_varlen_attention(
    q, kv, read_index, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale, d_latent, block_table=None, page_size=1
):
    """`q` [tokens, heads, d_latent + d_rope] absorbed queries, `kv` [rows, d_latent + d_rope] latent rows: the keys of
    the sequences are the rows `read_index[cu_seqlens_k[b]:cu_seqlens_k[b + 1]]`, or the rows `cu_seqlens_k[b]` onward
    when `read_index` is None, or with a `block_table` [sequences, pages] key `o` of sequence `b` is the row
    `block_table[b, o // page_size] * page_size + o % page_size`. Returns the latent output [tokens, heads, d_latent]."""
    num_tokens, num_heads, d_qk = q.shape
    out = torch.empty(num_tokens, num_heads, d_latent, dtype=q.dtype, device=q.device)
    num_seqs = cu_seqlens_q.shape[0] - 1
    d_latent_p = max(16, triton.next_power_of_2(d_latent))
    # Decode steps take a block of the sequence's heads, and their keys are split so the programs fill the SMs: one
    # program per sequence left most of them idle
    decode = max_seqlen_q == 1
    block_m = min(_PREFILL_BLOCK_M, max(16, triton.next_power_of_2(num_heads))) if decode else _PREFILL_BLOCK_M
    block_n = 32
    row_blocks = triton.cdiv((1 if decode else max_seqlen_q) * num_heads, block_m)
    num_splits = 1
    if decode:  # about one wave of programs (the padded sequences count too): more splits cost more in the merge
        num_splits = min(16, triton.next_power_of_2(triton.cdiv(_NUM_SMS, num_seqs * row_blocks)))
    partial_out = partial_lse = out
    if num_splits > 1:
        partial_out = torch.empty(num_splits, num_tokens * num_heads, d_latent_p, dtype=torch.float32, device=q.device)
        partial_lse = torch.empty(num_splits, num_tokens * num_heads, dtype=torch.float32, device=q.device)
    index = block_table if block_table is not None else (kv if read_index is None else read_index)
    _mla_varlen_kernel[(num_seqs, row_blocks, num_splits)](
        q, kv, index, out, partial_out, partial_lse, cu_seqlens_q, cu_seqlens_k,
        scale, q.stride(0), q.stride(1), kv.stride(0), out.stride(0), out.stride(1), num_tokens * num_heads,
        block_table.stride(0) if block_table is not None else 0,
        NUM_HEADS=num_heads, D_LATENT=d_latent, D_ROPE=d_qk - d_latent,
        D_LATENT_P=d_latent_p, D_ROPE_P=max(16, triton.next_power_of_2(d_qk - d_latent)),
        HAS_INDEX=read_index is not None, BLOCK_M=block_m, BLOCK_N=block_n, NUM_SPLITS=num_splits,
        HAS_BLOCK_TABLE=block_table is not None, PAGE=page_size,
        num_warps=8 if block_m == 64 and not decode else 4, num_stages=3 if decode else 2,
    )  # fmt: skip
    if num_splits > 1:
        _mla_combine_kernel[(num_tokens * num_heads,)](
            partial_out, partial_lse, out, cu_seqlens_q, num_seqs,
            NUM_HEADS=num_heads, D_LATENT=d_latent, D_LATENT_P=d_latent_p, NUM_SPLITS=num_splits, num_warps=4,
        )  # fmt: skip
    return out


# The cache writes and the views of the cache rows do not survive inductor's functionalization (illegal address under
# `default_compile_level`), and the varlen path was never compiled: the steps capture it as it runs eagerly
@torch.compiler.disable
def absorbed_mla_forward(attn, q_nope, q_rope, kv_nope, k_rope, cache, kwargs):
    """A continuous batching step of DeepSeek-V2 / V3 attention on the cached latents: `q_nope` [1, heads, tokens,
    nope], `q_rope` [1, heads, tokens, rope], and the new latents `kv_nope` [1, 1, tokens, kv_lora_rank], `k_rope`
    [1, 1, tokens, rope]. Writes the new latents to the cache like `cache.update`, then attends on the cache rows in
    place (no gathered copy). Returns the attention output before `o_proj`, [1, tokens, heads * v_head_dim]."""
    from torch.distributed.tensor import DTensor

    allocator = cache.layer_to_allocator[attn.layer_idx]
    latent_cache, rope_cache = allocator._latent_token_views[attn.layer_idx]
    cu_seqlens_k = kwargs["cu_seq_lens_k"][allocator.layer_type].to(torch.int32)
    block_table = kwargs.get("block_table")
    if block_table is not None:  # the step reads and writes the cache through the block table
        block_table = block_table[allocator.index]
        if attn.layer_idx == allocator.layer_indices[0]:  # the rows the new tokens go to, the same for every layer
            allocator._step_write_rows = _block_table_write_rows(
                block_table, kwargs["cu_seq_lens_q"], cu_seqlens_k, kv_nope.shape[2], allocator.tokens_per_page,
                allocator.write_trash_index,
            )  # fmt: skip
        write_index, read_index = allocator._step_write_rows, None
    else:
        write_index = kwargs["write_index"][allocator.index]
        read_index = kwargs["read_index"][allocator.index]
    latent_cache.index_copy_(0, write_index, kv_nope.flatten(0, 2))
    rope_cache.index_copy_(0, write_index, k_rope.flatten(0, 2))
    if block_table is None and read_index.numel() == 0:  # nothing cached to read: the new tokens are the sequences
        kv, read_index = torch.cat([kv_nope[0, 0], k_rope[0, 0]], dim=-1), None
    else:
        kv = latent_cache.as_strided((latent_cache.shape[0], allocator.row_numel), (latent_cache.stride(0), 1))

    num_heads = q_nope.shape[1]
    weight = attn.kv_b_proj.weight
    weight = weight.to_local() if isinstance(weight, DTensor) else weight  # tensor parallel: this rank's heads
    weight = weight.view(num_heads, attn.qk_nope_head_dim + attn.v_head_dim, attn.kv_lora_rank)
    w_uk, w_uv = weight[:, : attn.qk_nope_head_dim], weight[:, attn.qk_nope_head_dim :]
    if block_table is not None and is_prefill_step(q_nope.shape[2]) and not torch.cuda.is_current_stream_capturing():
        return _prefill_step(attn, allocator, q_nope, q_rope, kv, block_table, cu_seqlens_k, kwargs, w_uk, w_uv)
    q = torch.cat([torch.einsum("htn,hnc->thc", q_nope[0], w_uk), q_rope[0].transpose(0, 1)], dim=-1)
    out = mla_varlen_attention(
        q, kv, read_index, kwargs["cu_seq_lens_q"], cu_seqlens_k, kwargs["max_length_q"], attn.scaling,
        attn.kv_lora_rank, block_table, allocator.tokens_per_page,
    )  # fmt: skip
    return torch.einsum("thc,hvc->thv", out, w_uv).reshape(1, out.shape[0], -1)


def is_prefill_step(num_tokens: int) -> bool:
    """Block table steps with `MLA_PREFILL_MIN_TOKENS` (padded) query tokens or more hold prompts: the model runner runs
    them eagerly, and their prompts attend through flash attention on decompressed keys and values. The absorbed kernel
    spent ~7 s on one 8,192-token prompt of DeepSeek-V3.1 on a B300 (576-wide keys, one query token per program); the
    decode tokens of the step keep it."""
    from ..generation.continuous_batching.utils import MLA_PREFILL_MIN_TOKENS

    return num_tokens >= MLA_PREFILL_MIN_TOKENS


def _prefill_step(attn, allocator, q_nope, q_rope, kv, block_table, cu_seqlens_k, kwargs, w_uk, w_uv):
    """An eager block table step with prompts: sequences bringing several tokens attend through flash attention on
    their decompressed context (read through the block table), the decode tokens through the absorbed kernel. The
    step's partition is read on the host once, by the group's first layer."""
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    device, page = q_nope.device, allocator.tokens_per_page
    if attn.layer_idx == allocator.layer_indices[0]:
        cu_q = kwargs["cu_seq_lens_q"][: block_table.shape[0] + 1].tolist()
        cu_k = cu_seqlens_k[: block_table.shape[0] + 1].tolist()
        q_lens = [b - a for a, b in zip(cu_q[:-1], cu_q[1:])]
        k_lens = [b - a for a, b in zip(cu_k[:-1], cu_k[1:])]
        prefill = [i for i, n in enumerate(q_lens) if n > 1]
        decode = [i for i, n in enumerate(q_lens) if n == 1]
        plan = {"prefill": None, "decode": None}
        if prefill:
            seqs = torch.tensor(prefill, device=device)
            tokens = torch.cat([torch.arange(cu_q[i], cu_q[i + 1], device=device) for i in prefill])
            # the rows of every key of the prompts' sequences, read through the block table
            key_lens = torch.tensor([k_lens[i] for i in prefill], device=device)
            key_seq = torch.repeat_interleave(seqs, key_lens)
            key_pos = torch.cat([torch.arange(k_lens[i], device=device) for i in prefill])
            key_rows = block_table.long()[key_seq, key_pos // page] * page + key_pos % page
            cu_q_p = torch.tensor([0] + list(itertools.accumulate(q_lens[i] for i in prefill)), device=device, dtype=torch.int32)
            cu_k_p = torch.tensor([0] + list(itertools.accumulate(k_lens[i] for i in prefill)), device=device, dtype=torch.int32)
            max_q, max_k = max(q_lens[i] for i in prefill), max(k_lens[i] for i in prefill)
            plan["prefill"] = (tokens, key_rows, cu_q_p, cu_k_p, max_q, max_k)
        if decode:
            seqs = torch.tensor(decode, device=device)
            tokens = torch.tensor([cu_q[i] for i in decode], device=device)
            cu_q_d = torch.arange(len(decode) + 1, device=device, dtype=torch.int32)
            cu_k_d = torch.tensor([0] + list(itertools.accumulate(k_lens[i] for i in decode)), device=device, dtype=torch.int32)
            plan["decode"] = (tokens, seqs, cu_q_d, cu_k_d)
        allocator._step_prefill_plan = plan
    plan = allocator._step_prefill_plan
    num_tokens, num_heads = q_nope.shape[2], q_nope.shape[1]
    out = q_nope.new_zeros(num_tokens, num_heads, attn.v_head_dim)
    if plan["prefill"] is not None:
        tokens, key_rows, cu_q_p, cu_k_p, max_q, max_k = plan["prefill"]
        latents = kv[key_rows]  # [keys, kv_lora_rank + rope]
        key_states, value_states = attn.expand_kv(
            latents[None, None, :, : attn.kv_lora_rank], latents[None, None, :, attn.kv_lora_rank :]
        )
        query_states = torch.cat([q_nope[:, :, tokens], q_rope[:, :, tokens]], dim=-1)
        if _trtllm_ragged_available(device):
            # [scratch] Blackwell: flashinfer's TRT-LLM ragged DeepSeek prefill, as vLLM runs it (an 8,192-token prompt of
            # DeepSeek-V3.1 at TP4, 32 heads: 0.55 ms vs 2.34 for FA2's varlen call)
            prefill_out = _trtllm_ragged_prefill(
                query_states[0].transpose(0, 1), key_states[0].transpose(0, 1), value_states[0].transpose(0, 1),
                cu_q_p, cu_k_p, max_q, max_k, attn.scaling,
            )  # fmt: skip
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]
            prefill_out, _ = attention_interface(
                attn, query_states, key_states, value_states, None, dropout=0.0, scaling=attn.scaling,
                cu_seq_lens_q=cu_q_p, cu_seq_lens_k=cu_k_p, max_length_q=max_q, max_length_k=max_k,
            )  # fmt: skip
        out[tokens] = prefill_out.reshape(tokens.shape[0], num_heads, attn.v_head_dim)
    if plan["decode"] is not None:
        tokens, seqs, cu_q_d, cu_k_d = plan["decode"]
        q = torch.cat([torch.einsum("htn,hnc->thc", q_nope[0][:, tokens], w_uk), q_rope[0][:, tokens].transpose(0, 1)], dim=-1)
        latent_out = mla_varlen_attention(
            q, kv, None, cu_q_d, cu_k_d, 1, attn.scaling, attn.kv_lora_rank, block_table[seqs], page
        )  # fmt: skip
        out[tokens] = torch.einsum("thc,hvc->thv", latent_out, w_uv)
    return out.reshape(1, num_tokens, -1)


_PROTO_MLA_TRTLLM_PREFILL = os.environ.get("PROTO_MLA_TRTLLM_PREFILL", "1") == "1"
_RAGGED_WORKSPACE: dict = {}


def _trtllm_ragged_available(device: torch.device) -> bool:
    return (
        _PROTO_MLA_TRTLLM_PREFILL
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] == 10
        and importlib.util.find_spec("flashinfer") is not None
    )


def _trtllm_ragged_prefill(query, key, value, cu_seqlens_q, cu_seqlens_k, max_q, max_k, scale):
    """Causal (bottom-right) varlen attention of `query` [tokens, heads, 192] on `key` [keys, heads, 192] and `value`
    [keys, heads, 128]. Returns [tokens, heads, 128]."""
    import flashinfer

    workspace = _RAGGED_WORKSPACE.get(query.device)
    if workspace is None:
        workspace = _RAGGED_WORKSPACE[query.device] = torch.zeros(128 << 20, dtype=torch.uint8, device=query.device)
    seq_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)
    return flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query.contiguous(), key.contiguous(), value.contiguous(), workspace, seq_lens, max_q, max_k, scale, 1.0, 1.0,
        seq_lens.shape[0], -1, cu_seqlens_q, cu_seqlens_k, False, True, False,
    )  # fmt: skip


def _block_table_write_rows(block_table, cu_seqlens_q, cu_seqlens_k, num_tokens, page_size, trash_row):
    """The cache row of each new token of a block table step: token `t` of sequence `b` is at position
    `seqlen_k[b] - seqlen_q[b] + t` of its sequence. Padding tokens (after the last sequence's) go to the trash row."""
    cu_q, cu_k = cu_seqlens_q.long(), cu_seqlens_k.long()
    num_seqs = block_table.shape[0]
    token = torch.arange(num_tokens, device=block_table.device)
    seq = torch.searchsorted(cu_q[1 : num_seqs + 1].contiguous(), token, right=True)
    real = seq < num_seqs
    seq = seq.clamp(max=num_seqs - 1)
    position = (cu_k[seq + 1] - cu_k[seq]) - (cu_q[seq + 1] - cu_q[seq]) + (token - cu_q[seq])
    page = block_table.long()[seq, (position // page_size).clamp(0, block_table.shape[1] - 1)]
    return torch.where(real & (page >= 0), page * page_size + position % page_size, trash_row)
