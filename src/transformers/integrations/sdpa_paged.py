import torch

from ..generation.continuous_batching.cache import PagedAttentionCache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# MPS paged-decode fast path — uses ``paged_decode_attention_f32`` from
# ``ArthurZ/gguf-kernels`` to read K/V through a block-table indirection
# instead of gathering the cache into a contiguous tensor (the existing
# varlen path's ~16 ms/layer bottleneck on MPS — see PR #45977).
# ---------------------------------------------------------------------------

_PAGED_DECODE_OP = None


def _get_mps_paged_decode_op():
    """Return the (cached) ``paged_decode_attention_f32`` torch op handle, or
    ``None`` if the kernel package isn't loaded. Lazy so import order
    doesn't matter."""
    global _PAGED_DECODE_OP
    if _PAGED_DECODE_OP is not None:
        return _PAGED_DECODE_OP if _PAGED_DECODE_OP is not False else None
    try:
        from .gguf_linear import _ensure_metal_kernels
    except Exception:
        _PAGED_DECODE_OP = False
        return None
    mod = _ensure_metal_kernels()
    if mod is None or not hasattr(mod._ops, "paged_decode_attention_f32"):
        _PAGED_DECODE_OP = False
        return None
    op = mod._ops.paged_decode_attention_f32
    _PAGED_DECODE_OP = getattr(op, "default", op)
    return _PAGED_DECODE_OP


def _mps_block_table_path(
    module, query, key, value, cache: PagedAttentionCache,
    block_table_full: torch.Tensor, cu_seq_lens_k: torch.Tensor, **_,
) -> torch.Tensor:
    """Decode-only block-table fast path. Writes the current token's K/V
    into the paged cache then runs ``paged_decode_attention_f32`` reading
    through the block table — no gather, no ``cache.update`` round-trip.

    The cache write slot per request is computed from ``block_table +
    seq_lens`` directly (CB only populates ``write_index`` for the varlen
    path; under ``use_block_table=True`` it's left stale).

    Returns ``attn_output`` shaped like the varlen path's output:
    ``(1, total_q=B, num_heads, head_dim)``.
    """
    op = _get_mps_paged_decode_op()
    assert op is not None
    # Q / K / V come in as (1, num_heads, total_q, head_dim) with one row
    # per request on the total_q axis (CB packs all queries on dim 2).
    # For decode-fast-path total_q == batch_size; q seq len per request = 1.
    batch_size = query.size(2)
    H_Q = query.size(1)
    head_dim = query.size(3)
    H_KV = key.size(1)

    group_idx, layer_idx_in_group = cache.layer_index_to_group_indices[module.layer_idx]
    k_cache = cache.key_cache[layer_idx_in_group]
    v_cache = cache.value_cache[layer_idx_in_group]
    block_table = block_table_full[group_idx]  # (B, max_blocks) int32 (already)

    # seq_lens[i] = current cache length for request i AFTER this step
    # (cu_seq_lens_k from CB already counts the new token).
    seq_lens = cu_seq_lens_k[1 : batch_size + 1] - cu_seq_lens_k[:batch_size]
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)

    # Compute the absolute cache slot to write per request from the block
    # table — the new token lives at position seq_lens-1 within the
    # request's KV sequence, which maps to
    # ``block_table[b, pos // block_size] * block_size + pos % block_size``.
    block_size = cache.block_size
    write_pos = seq_lens.to(torch.long) - 1                             # (B,)
    block_in_seq = write_pos // block_size
    block_id = block_table.to(torch.long).gather(1, block_in_seq.unsqueeze(1)).squeeze(1)
    write_slot = block_id * block_size + (write_pos % block_size)

    # Pack new K/V into cache layout and write at the computed slots.
    key_w = key.transpose(1, 2).squeeze(0)     # (total_q, H_KV, head_dim)
    value_w = value.transpose(1, 2).squeeze(0)
    k_cache.index_put_((write_slot,), key_w)
    v_cache.index_put_((write_slot,), value_w)

    # paged_decode_attention_f32 takes Q (B, H_Q, head_dim).
    q_in = query.permute(2, 1, 0, 3).squeeze(-2).contiguous()  # (B, H_Q, head_dim)
    out = torch.empty(batch_size, H_Q, head_dim, dtype=q_in.dtype, device=q_in.device)
    scale = getattr(module, "scaling", None)
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    op(q_in, k_cache, v_cache, block_table, seq_lens, out, block_size, float(scale))
    return out.unsqueeze(0)  # (1, B, H_Q, head_dim)


def sdpa_attention_paged_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    cache: PagedAttentionCache | None = kwargs.pop("cache", None)
    block_table = kwargs.pop("block_table", None)

    # ---- MPS block-table decode fast path -------------------------------
    # Activates when:
    #   1. block_table is provided (CB switched to use_decode_fast_path)
    #   2. running on MPS
    #   3. ``ArthurZ/gguf-kernels`` exposes ``paged_decode_attention_f32``
    # Reads K/V through the block_table indirection inside one Metal
    # kernel — 134 µs/call at (B=4, H_Q=16, head_dim=128, S=50) vs ~16 ms
    # per layer for the gather-then-SDPA path. Bit-equivalent output.
    import os as _os
    _fast_disabled = _os.environ.get("TRANSFORMERS_MPS_PAGED_DECODE_DISABLE", "0") != "0"
    _fast_eligible = (not _fast_disabled
            and cache is not None and block_table is not None
            and query.device.type == "mps"
            and _get_mps_paged_decode_op() is not None)
    if _fast_eligible:
        cu_seq_lens_k = kwargs.get("cu_seq_lens_k")
        if isinstance(cu_seq_lens_k, dict):
            layer_type = "sliding_attention" if getattr(module, "sliding_window", False) else "full_attention"
            cu_seq_lens_k = cu_seq_lens_k[layer_type]
        attn_output = _mps_block_table_path(
            module, query, key, value, cache, block_table, cu_seq_lens_k,
        )
        return attn_output, None

    # ---- Varlen path (existing) ----------------------------------------
    if cache is not None:
        # This changes the shape of k and v from [1, num_kv_heads, seqlen_kv, head_dim] to [-1, num_kv_heads, head_dim]
        key, value = cache.update(
            key_states=key,
            value_states=value,
            layer_idx=module.layer_idx,
            read_index=kwargs["read_index"],
            write_index=kwargs["write_index"],
        )
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        # Packed sequence format is used for input, so that it can never be causal.
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
