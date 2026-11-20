# [scratch, bench only] Regional torch.compile of the continuous-batching decode step of GLM-MoE-DSA.
#
# One compiled artifact per *kind* of decoder layer (dense MLP / MoE, with / without its own indexer) instead of one
# compile of the whole `ModelRunner._forward_process_and_sample`:
#   - the layer loop stays eager Python (CB compile level 0) and calls a compiled per-layer function;
#   - that function reads nothing that differs between two layers of a kind as a Python value: no `layer_idx`, no
#     per-layer dict lookup. The cache rows of the layer, the write index and the top-k rows come in as tensors with
#     the same shape for every layer;
#   - the pieces dynamo cannot trace are custom ops with fake implementations, so the layer is a single graph: the
#     cache write (mutation of an input, kept in place), the FlashMLA sparse kernel, the TP all-reduce;
#   - what depends on the batch's total context (read index, per-request bookkeeping) is computed once per step,
#     outside: with `PROTO_INDEXER_SKIP=1` and every context within `index_topk`, the rows each query attends to are
#     the same for all layers.
#
# FUSE_COMPILE: unset / 0 = off; `eager` = same structure, layer function not compiled (isolates the hoisting);
#   `default` = torch.compile mode "default"; `max` = "max-autotune-no-cudagraphs"; `reduce` = "reduce-overhead" is
#   NOT supported (CB captures its own CUDA graphs).
# FUSE_COMPILE_DYNAMIC: 0 = static shapes (one artifact per kind and padded batch size), 1 = dynamic (default 0).
# FUSE_COMPILE_FULLGRAPH: 1 (default) errors on any graph break inside the layer.
# FUSE_COMPILE_TP (tensor parallel only): `twin` (default) = the compiled function runs on a twin of each layer that
#   holds this rank's local parameters and the classes' own forwards, and all-reduces explicitly where the tp_plan
#   does (after `o_proj`, after the MLP block: needs HF_TP_SINGLE_MLP_ALL_REDUCE=1); `trace` = trace through the
#   tp_plan forward wrappers installed on the modules (probe of what dynamo makes of them).
import copy
import os
import time

import torch
import torch.nn.functional as F


MODE = os.environ.get("FUSE_COMPILE", "0")
DYNAMIC = os.environ.get("FUSE_COMPILE_DYNAMIC", "0") == "1"
FULLGRAPH = os.environ.get("FUSE_COMPILE_FULLGRAPH", "1") == "1"
TP_MODE = os.environ.get("FUSE_COMPILE_TP", "twin")

_flash_mla = None
_rotary = None  # `apply_rotary_pos_emb_interleave` of the modeling file, set on first use (no circular import)
_compiled = {}  # (mode, dynamic, fullgraph) -> compiled layer function
_tp_group = None  # process group of the TP mesh, set by `prepare_model` when the model is tensor parallel
STATS = {"steps": 0, "fallbacks": 0}


# --- opaque pieces as custom ops ---------------------------------------------------------------------------------------


def _cache_write(
    cache_rows: torch.Tensor,
    write_index: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rot: torch.Tensor,
    index_key: torch.Tensor | None,
) -> None:
    """Writes the new tokens of a layer to its cache rows [tokens, kv_lora + rope (+ index_head_dim)], in place."""
    kv_lora, rope = kv_latent.shape[-1], k_rot.shape[-1]
    cache_rows[:, :kv_lora].index_copy_(0, write_index, kv_latent)
    cache_rows[:, kv_lora : kv_lora + rope].index_copy_(0, write_index, k_rot)
    if index_key is not None:
        cache_rows[:, kv_lora + rope :].index_copy_(0, write_index, index_key)


def _sparse_mla(q: torch.Tensor, cache_rows: torch.Tensor, rows: torch.Tensor, scale: float, d_qk: int, d_v: int) -> torch.Tensor:
    """FlashMLA sparse on the cache rows: q [M, heads, d_qk] absorbed queries, rows [M, topk] physical rows (-1 = empty
    slot). Returns the latent output [M, heads, d_v]."""
    global _flash_mla
    if _flash_mla is None:
        from vllm.third_party.flashmla import flash_mla_interface

        _flash_mla = flash_mla_interface
    latent_rows = cache_rows.as_strided((cache_rows.shape[0], 1, d_qk), (cache_rows.stride(0), 0, 1))
    return _flash_mla.flash_mla_sparse_fwd(q, latent_rows, rows[:, None, :], scale, d_v=d_v)[0]


def _sparse_mla_fake(q, cache_rows, rows, scale, d_qk, d_v):
    assert q.dim() == 3 and cache_rows.dim() == 2 and rows.dim() == 2
    return q.new_empty((q.shape[0], q.shape[1], d_v))


def _tp_all_reduce(x: torch.Tensor) -> None:
    """Sums this rank's partial result with the other ranks', in place."""
    import torch.distributed as dist

    dist.all_reduce(x, group=_tp_group)


def _localize(module):
    """A twin of `module` for inference on one tensor-parallel rank: same classes, same storage, but the parameters
    are this rank's local tensors instead of DTensors and the forwards are the classes' own (the tp_plan wrappers set
    on the instances are dropped). The original module is left untouched for the regular path."""
    from torch.distributed.tensor import DTensor

    twin = copy.copy(module)
    twin.__dict__ = dict(module.__dict__)
    twin.__dict__.pop("forward", None)
    twin._parameters = {
        name: torch.nn.Parameter(p.to_local(), requires_grad=False) if isinstance(p, DTensor) else p
        for name, p in module._parameters.items()
    }
    twin._modules = {name: None if m is None else _localize(m) for name, m in module._modules.items()}
    return twin


torch.library.custom_op("transformers::fuse_cache_write", _cache_write, mutates_args=("cache_rows",))
torch.library.custom_op("transformers::fuse_sparse_mla", _sparse_mla, mutates_args=())
torch.library.register_fake("transformers::fuse_sparse_mla", _sparse_mla_fake)
torch.library.custom_op("transformers::fuse_tp_all_reduce", _tp_all_reduce, mutates_args=("x",))


# --- the layer, as one traceable function ----------------------------------------------------------------------------


def _decode_layer(layer, hidden_states, cos, sin, cache_rows, write_index, rows, w_kv_b, h_pad: int, tp: bool):
    """`GlmMoeDsaDecoderLayer.forward` on the prototype's decode path (`_proto_sparse_mla_attention` with the indexer
    scoring skipped), same operations in the same order. Nothing here may depend on which layer it is."""
    attn = layer.self_attn
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    batch_size, seq_length = hidden_states.shape[:-1]
    q_resid = attn.q_a_layernorm(attn.q_a_proj(hidden_states))
    q_states = attn.q_b_proj(q_resid).view(batch_size, seq_length, -1, attn.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_rot = torch.split(compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    k_pass = attn.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, attn.kv_lora_rank)
    k_rot = k_rot.view(batch_size, 1, seq_length, attn.qk_rope_head_dim)
    q_rot, k_rot = _rotary(q_rot, k_rot, cos, sin)

    index_key = None
    indexer = attn.indexer
    if indexer is not None:
        # the indexer's key of the new tokens: cached for later, longer contexts (scoring is skipped)
        k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
        ki_rot, ki_pass = torch.split(k, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim=-1)
        _, ki_rot = _rotary(ki_rot, ki_rot, cos, sin, unsqueeze_dim=2)
        index_key = torch.cat([ki_rot, ki_pass], dim=-1).squeeze(2)[0]
    torch.ops.transformers.fuse_cache_write(
        cache_rows, write_index, k_pass.reshape(-1, attn.kv_lora_rank), k_rot.reshape(-1, attn.qk_rope_head_dim),
        index_key,
    )

    num_heads = q_pass.shape[1]
    d_qk = attn.kv_lora_rank + attn.qk_rope_head_dim
    w_uk, w_uv = w_kv_b[:, : attn.qk_nope_head_dim], w_kv_b[:, attn.qk_nope_head_dim :]
    q_abs = q_pass.new_zeros(seq_length, max(h_pad, num_heads), d_qk)
    q_abs[:, :num_heads, : attn.kv_lora_rank] = torch.einsum("hsn,hnc->shc", q_pass[0], w_uk)
    q_abs[:, :num_heads, attn.kv_lora_rank :] = q_rot[0].transpose(0, 1)
    out_latent = torch.ops.transformers.fuse_sparse_mla(q_abs, cache_rows, rows, attn.scaling, d_qk, attn.kv_lora_rank)
    out_latent = out_latent[:, :num_heads]
    attn_output = torch.einsum("shc,hvc->shv", out_latent, w_uv).reshape(batch_size, seq_length, -1)
    attn_output = attn.o_proj(attn_output)
    if tp:
        torch.ops.transformers.fuse_tp_all_reduce(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    if tp:
        torch.ops.transformers.fuse_tp_all_reduce(hidden_states)
    return residual + hidden_states


def _layer_fn():
    if MODE == "eager":
        return _decode_layer
    key = (MODE, DYNAMIC, FULLGRAPH)
    if key not in _compiled:
        mode = {"default": "default", "max": "max-autotune-no-cudagraphs"}[MODE]
        # static shapes: one artifact per layer kind and padded batch size, more than dynamo's default of 8 per frame
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        _compiled[key] = torch.compile(_decode_layer, mode=mode, dynamic=DYNAMIC, fullgraph=FULLGRAPH)
    return _compiled[key]


# --- eager part: once per model, once per step -----------------------------------------------------------------------


def _prepare(model, cache):
    """Once per (model, cache): per-layer cache views that all have the same shape, the dequantized `kv_b_proj`."""
    from .modeling_glm_moe_dsa import _proto_kv_b_dequantized, apply_rotary_pos_emb_interleave

    global _rotary
    _rotary = apply_rotary_pos_emb_interleave
    layers = model.layers[: model.config.num_hidden_layers]
    allocator = cache.layer_to_allocator[0]
    if any(cache.layer_to_allocator[i] is not allocator for i in range(len(layers))):
        raise NotImplementedError("FUSE_COMPILE expects a single cache allocator for all layers")
    page = allocator.tokens_per_page
    position = {layer_idx: i for i, layer_idx in enumerate(allocator.layer_indices)}
    total = allocator.cache_tensor.shape[0]
    common = total - (len(allocator.layer_indices) - 1) * page  # rows every layer can address
    views = [allocator.cache_tensor[position[i] * page : position[i] * page + common] for i in range(len(layers))]
    weights = [_proto_kv_b_dequantized(layer.self_attn) for layer in layers]
    h_pad = 0 if torch.cuda.get_device_capability()[0] == 9 else 64
    tp, run_layers_on = False, list(layers)
    o_proj_weight = layers[0].self_attn.o_proj.weight
    if hasattr(o_proj_weight, "device_mesh"):  # tensor parallel: the parameters are DTensors
        global _tp_group
        mesh = o_proj_weight.device_mesh
        _tp_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        if TP_MODE == "twin":
            if os.environ.get("HF_TP_SINGLE_MLP_ALL_REDUCE", "0") != "1":
                raise NotImplementedError("FUSE_COMPILE under TP expects HF_TP_SINGLE_MLP_ALL_REDUCE=1")
            if any(layer.self_attn.o_proj.bias is not None for layer in layers):
                raise NotImplementedError("FUSE_COMPILE under TP expects bias-free row-parallel projections")
            tp, run_layers_on = True, [_localize(layer) for layer in layers]
    state = {
        "cache": cache, "allocator": allocator, "views": views, "w_kv_b": weights, "h_pad": h_pad, "tp": tp,
        "layers": run_layers_on,
    }
    model._fuse_compile_state = state
    return state


def applies(model, hidden_states, cache, kwargs) -> bool:
    """Decode-shaped CB batches of the prototype path whose indexer scoring is skipped; anything else (prefill batches,
    contexts longer than `index_topk`, other caches) keeps the regular layer loop."""
    from . import modeling_glm_moe_dsa as m

    if not (m._PROTO_SPARSE_MLA and m._PROTO_PER_QUERY_INDEXER and m._PROTO_INDEXER_SKIP):
        return False
    if cache is None or not hasattr(cache, "layer_to_allocator") or "read_index" not in kwargs:
        return False
    if hidden_states.shape[1] > 512 or model.training:
        return False
    allocator = cache.layer_to_allocator[0]
    ok = kwargs["max_length_k"][allocator.layer_type] <= model.config.index_topk
    STATS["fallbacks"] += not ok
    return ok


def run_layers(model, hidden_states, position_embeddings, cache, kwargs):
    state = getattr(model, "_fuse_compile_state", None)
    if state is None or state["cache"] is not cache:
        state = _prepare(model, cache)
    allocator = state["allocator"]
    read_index = kwargs["read_index"][allocator.index]
    write_index = kwargs["write_index"][allocator.index]
    if read_index.numel() == 0:  # no past in the batch: the context is the new tokens
        read_index = write_index
    cu_q = kwargs["cu_seq_lens_q"]
    cu_k = kwargs["cu_seq_lens_k"][allocator.layer_type]
    max_k = kwargs["max_length_k"][allocator.layer_type]

    # Once per step: the physical rows each query attends to (every visible token of its request, in order). Same
    # arithmetic as `_proto_indexer_per_query` with scoring skipped + the mapping of `_proto_sparse_mla_attention`.
    num_tokens, device = hidden_states.shape[1], hidden_states.device
    tok = torch.arange(num_tokens, device=device)
    num_req = cu_q.shape[0] - 1
    req = torch.searchsorted(cu_q[1:].contiguous(), tok, right=True)
    real = req < num_req  # padded query tokens belong to no request
    req = req.clamp(max=num_req - 1)
    q_start, k_start = cu_q[req], cu_k[req]
    past = (cu_k[req + 1] - k_start) - (cu_q[req + 1] - q_start)
    n_visible = torch.where(real, past + (tok - q_start) + 1, 0)
    offs = torch.arange(max_k, device=device)
    pos = (k_start[:, None] + offs[None]).clamp(max=read_index.shape[0] - 1)
    visible = offs[None] < n_visible[:, None]
    rows = torch.where(visible, read_index[pos], -1).to(torch.int32)
    if rows.shape[-1] % 128:  # the kernel takes top-k lists in multiples of 128, -1 marks an empty slot
        rows = F.pad(rows, (0, 128 - rows.shape[-1] % 128), value=-1)

    cos, sin = position_embeddings
    fn = _layer_fn()
    tp = state["tp"]
    timings = STATS.get("layer_seconds")  # set to a list by a benchmark to get the wall time of each layer call
    ranges = STATS.get("ranges")  # set by a benchmark: one profiler range per layer call, for kernel attribution
    for i, layer in enumerate(state["layers"]):
        if timings is not None:
            start = time.perf_counter()
        if ranges:
            with torch.autograd.profiler.record_function(f"M:L{i}:layer"):
                hidden_states = fn(
                    layer, hidden_states, cos, sin, state["views"][i], write_index, rows, state["w_kv_b"][i],
                    state["h_pad"], tp,
                )
            continue
        hidden_states = fn(
            layer, hidden_states, cos, sin, state["views"][i], write_index, rows, state["w_kv_b"][i], state["h_pad"],
            tp,
        )
        if timings is not None:
            timings.append(round(time.perf_counter() - start, 3))
    STATS["steps"] += 1
    return hidden_states
