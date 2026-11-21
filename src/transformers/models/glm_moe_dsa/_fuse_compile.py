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
#
# [gap3 compile] On the stack S (CB_PROTO_MLA_BLOCK_TABLE=1): only block-table decode batches take this path (a varlen
# batch's shapes follow CB's padded (q, kv) intervals: one artifact set per pair, compiled mid-serving). Their rows come
# from CB's block table once per step (same arithmetic as `_proto_block_table_rows`), the top-k list width (CB's
# power-of-2 bucket of the longest context) is a dynamic dim that only the opaque FlashMLA op sees, so one artifact
# per layer kind and padded request count serves every bucket. PROTO_TOPK_LENGTH is followed. The TRT-LLM routed MoE
# (PROTO_EXPERTS_TRTLLM=1) is an opaque op (`integrations/finegrained_fp8.py`), the TP all-reduce goes through the
# inference reduce of the TP styles (NCCL, or flashinfer with HF_TP_ALL_REDUCE=flashinfer on CB's engine thread).
import copy
import os
import time

import torch
import torch.nn.functional as F


try:  # the fused Triton kernels (FUSE_KERNELS=1) register their ops at import
    from . import _fuse_kernels  # noqa: F401
except ImportError:
    _fuse_kernels = None

MODE = os.environ.get("FUSE_COMPILE", "0")
DYNAMIC = os.environ.get("FUSE_COMPILE_DYNAMIC", "0") == "1"
FULLGRAPH = os.environ.get("FUSE_COMPILE_FULLGRAPH", "1") == "1"
TP_MODE = os.environ.get("FUSE_COMPILE_TP", "twin")
# FUSE_KERNELS=1: the per-layer function is `_decode_layer_fused`, whose fixed chains are the Triton kernels of
#   `_fuse_kernels.py` (works with FUSE_COMPILE=eager, and inside the compiled layer with default / max).
# FUSE_TOPK_LENGTH=1: FlashMLA gets the number of valid entries of each top-k list (same as wp2-prefill's
#   PROTO_TOPK_LENGTH on the regular path) and the head-padded query is a zeroed buffer kept per step shape.
KERNELS = os.environ.get("FUSE_KERNELS", "0") == "1"
TOPK_LENGTH = os.environ.get("FUSE_TOPK_LENGTH", "0") == "1"
BLOCK_TABLE_ONLY = os.environ.get("CB_PROTO_MLA_BLOCK_TABLE", "0") == "1"
# [gap3] FUSE_COMPILE_EXACT=1 (default): inductor keeps eager's bf16 rounding between the ops it fuses
#   (`emulate_precision_casts`), so the compiled layer gives the same bits as the eager one. 0: inductor keeps fp32
#   between fused ops (closer to an fp32 reference, but greedy outputs drift from the eager path's).
EXACT = os.environ.get("FUSE_COMPILE_EXACT", "1") == "1"

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


def _sparse_mla(
    q: torch.Tensor, cache_rows: torch.Tensor, rows: torch.Tensor, counts: torch.Tensor | None, scale: float, d_qk: int,
    d_v: int,
) -> torch.Tensor:
    """FlashMLA sparse on the cache rows: q [M, heads, d_qk] absorbed queries, rows [M, topk] physical rows (-1 = empty
    slot), counts [M] int32 valid entries at the start of each list (or None: the kernel walks the whole list).
    Returns the latent output [M, heads, d_v]."""
    global _flash_mla
    if _flash_mla is None:
        from vllm.third_party.flashmla import flash_mla_interface

        _flash_mla = flash_mla_interface
    latent_rows = cache_rows.as_strided((cache_rows.shape[0], 1, d_qk), (cache_rows.stride(0), 0, 1))
    return _flash_mla.flash_mla_sparse_fwd(q, latent_rows, rows[:, None, :], scale, d_v=d_v, topk_length=counts)[0]


def _sparse_mla_fake(q, cache_rows, rows, counts, scale, d_qk, d_v):
    assert q.dim() == 3 and cache_rows.dim() == 2 and rows.dim() == 2
    return q.new_empty((q.shape[0], q.shape[1], d_v))


def _tp_all_reduce(x: torch.Tensor) -> None:
    """Sums this rank's partial result with the other ranks', in place, through the inference reduce of the TP styles
    (NCCL in place; flashinfer returns a new tensor, copied back)."""
    from ...distributed.parallel_utils import _tp_inference_all_reduce

    out = _tp_inference_all_reduce(x, _tp_group)
    if out is not x:
        x.copy_(out)


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


def _decode_layer(layer, hidden_states, cos, sin, cache_rows, write_index, rows, counts, w_kv_b, h_pad: int, tp: bool):
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
    out_latent = torch.ops.transformers.fuse_sparse_mla(
        q_abs, cache_rows, rows, counts, attn.scaling, d_qk, attn.kv_lora_rank
    )
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


def _decode_layer_fused(
    layer, residual, hidden_states, cos, sin, cache_rows, write_index, rows, counts, w_kv_b, q_abs, next_weight,
    next_eps: float, tp: bool, qkv_a=None,
):
    """Same layer with its fixed chains as fused kernels (FUSE_KERNELS=1). `hidden_states` is `residual` already
    normed by this layer's `input_layernorm` (done by the previous layer's last kernel); returns this layer's output
    and the same output normed with `next_weight` (the next layer's `input_layernorm`, or the final norm). `q_abs` is
    the zeroed head-padded query buffer of the step, [M, 64 (or heads), kv_lora + rope]: only its real heads are written."""
    attn = layer.self_attn
    ops = torch.ops.transformers
    seq_length = hidden_states.shape[1]
    indexer = attn.indexer
    index_key = ln_weight = ln_bias = None
    ln_eps = 0.0
    if qkv_a is not None:
        # [gap3] GEMM_FUSE_QKV_A: q_a, kv_a (and the indexer's wk) in one GEMM, the kernel reads the column slices
        projections = F.linear(hidden_states, qkv_a).split(attn._gemm_qkv_a_sizes, dim=-1)
        q_a, kv_a = projections[0], projections[1]
        wk_out = projections[2] if indexer is not None else None
    else:
        q_a, kv_a = attn.q_a_proj(hidden_states), attn.kv_a_proj_with_mqa(hidden_states)
        wk_out = indexer.wk(hidden_states) if indexer is not None else None
    if indexer is not None:  # the indexer's key of the new tokens, cached for later, longer contexts
        index_key, ln_weight, ln_bias, ln_eps = wk_out, indexer.k_norm.weight, indexer.k_norm.bias, indexer.k_norm.eps
    # q_a norm; kv_a norm + key rotary + cache write; indexer key norm + rotary + cache write: one kernel
    q_resid = ops.fuse_norm_rope_write(
        q_a, attn.q_a_layernorm.weight, kv_a, attn.kv_a_layernorm.weight, cos, sin, write_index, cache_rows, index_key,
        ln_weight, ln_bias, attn.q_a_layernorm.variance_epsilon, attn.kv_a_layernorm.variance_epsilon, ln_eps,
    )
    q_b = attn.q_b_proj(q_resid)  # [1, M, heads * (nope + rope)]
    q_pass = q_b.view(seq_length, -1, attn.qk_head_dim)[:, :, : attn.qk_nope_head_dim].transpose(0, 1)  # [heads, M, nope]
    num_heads = q_pass.shape[0]
    w_uk, w_uv = w_kv_b[:, : attn.qk_nope_head_dim], w_kv_b[:, attn.qk_nope_head_dim :]
    q_lat = torch.bmm(q_pass, w_uk)  # [heads, M, kv_lora]: the query absorbs kv_b_proj's key half
    # query rotary + pack [absorbed latent | rotated rope] into the padded buffer: one kernel
    ops.fuse_q_pack(q_b, q_lat, cos, sin, q_abs, attn.qk_nope_head_dim)
    d_qk = attn.kv_lora_rank + attn.qk_rope_head_dim
    out_latent = ops.fuse_sparse_mla(q_abs, cache_rows, rows, counts, attn.scaling, d_qk, attn.kv_lora_rank)
    # value half of kv_b_proj: [heads, M, kv_lora] @ [heads, kv_lora, v]
    attn_output = torch.bmm(out_latent[:, :num_heads].transpose(0, 1), w_uv.transpose(1, 2))
    attn_output = attn.o_proj(attn_output.transpose(0, 1).reshape(1, seq_length, -1))
    if tp:
        ops.fuse_tp_all_reduce(attn_output)
    # residual add + post-attention norm: one kernel
    residual, hidden_states = ops.fuse_add_rms_norm(
        residual, attn_output, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon
    )
    hidden_states = layer.mlp(hidden_states)
    if tp:
        ops.fuse_tp_all_reduce(hidden_states)
    # residual add + the next layer's input norm: one kernel
    return ops.fuse_add_rms_norm(residual, hidden_states, next_weight, next_eps)


def _layer_fn():
    fn = _decode_layer_fused if KERNELS else _decode_layer
    if MODE == "eager":
        return fn
    key = (MODE, DYNAMIC, FULLGRAPH, KERNELS, EXACT)
    if key not in _compiled:
        mode = {"default": "default", "max": "max-autotune-no-cudagraphs"}[MODE]
        # inductor options of the mode, plus eager's rounding with FUSE_COMPILE_EXACT (`mode` and `options` exclude
        # each other in torch.compile)
        options = {**torch._inductor.list_mode_options(mode), "emulate_precision_casts": EXACT}
        # Static shapes: one artifact per layer kind and padded batch size, more than dynamo's default of 8 per frame.
        # The limit is passed per region: `torch._dynamo.config` overrides are thread-local and every CB manager runs
        # its own engine thread, so a value set in one thread is not seen by the next manager's.
        _compiled[key] = torch.compile(
            fn, options=options, dynamic=DYNAMIC, fullgraph=FULLGRAPH, recompile_limit=64
        )
    return _compiled[key]


# --- eager part: once per model, once per step -----------------------------------------------------------------------


def _prepare(model, cache):
    """Once per (model, cache): per-layer cache views that all have the same shape, the dequantized `kv_b_proj`."""
    from .modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave

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
    if allocator.tokens_per_block != len(allocator.layer_indices) * page:
        raise NotImplementedError("FUSE_COMPILE expects one page per layer in a block")
    views = [allocator.cache_tensor[position[i] * page : position[i] * page + common] for i in range(len(layers))]
    for view in views:  # [gap3] the cache size changes with each manager: only the opaque cache ops read this dim
        torch._dynamo.mark_dynamic(view, 0)
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
        "views": views, "h_pad": h_pad, "tp": tp,
        "layers": run_layers_on, "q_abs": {},
    }
    # [gap3] kept on the cache, not on the model: the state holds views of the cache tensor, and a reference from the
    # model kept a finished manager's cache alive while the next manager sized its own (KV blocks 1447 -> 553)
    cache._fuse_compile_state = state
    return state


def applies(model, hidden_states, cache, kwargs) -> bool:
    """Decode-shaped CB batches of the prototype path whose indexer scoring is skipped; anything else (prefill batches,
    contexts longer than `index_topk`, other caches) keeps the regular layer loop."""
    from . import modeling_glm_moe_dsa as m

    if not (m._PROTO_SPARSE_MLA and m._PROTO_PER_QUERY_INDEXER and m._PROTO_INDEXER_SKIP):
        return False
    if cache is None or not hasattr(cache, "layer_to_allocator") or "read_index" not in kwargs:
        return False
    if model.training or torch.is_grad_enabled():
        return False
    if kwargs.get("block_table") is None and (BLOCK_TABLE_ONLY or hidden_states.shape[1] > 512):
        return False  # a block-table batch is a decode batch whatever its size (with CB_MTP=spec, 1 + k tokens each)
    if m._TRAIN_FIX_KV_B == "delta" and any(m._train_kv_b_adapters(layer.self_attn) for layer in model.layers):
        return False  # the adapters' low-rank terms of the absorbed attention are not in the compiled layer
    allocator = cache.layer_to_allocator[0]
    ok = kwargs["max_length_k"][allocator.layer_type] <= model.config.index_topk
    STATS["fallbacks"] += not ok
    return ok


def run_layers(model, hidden_states, position_embeddings, cache, kwargs):
    state = getattr(cache, "_fuse_compile_state", None)
    if state is None:
        state = _prepare(model, cache)
    from . import modeling_glm_moe_dsa as m

    allocator = cache.layer_to_allocator[0]
    # kv_b_proj per step, not cached with the state: training may refresh or rebind it (TRAIN_FIX_KV_B)
    w_kv_b = [m._proto_kv_b_dequantized(layer.self_attn) for layer in model.layers[: model.config.num_hidden_layers]]
    topk_length = TOPK_LENGTH or m._PROTO_TOPK_LENGTH
    if kwargs.get("block_table") is not None:
        write_index, rows, counts = _block_table_rows(model, cache, allocator, hidden_states, kwargs, topk_length)
        STATS["bt_steps"] = STATS.get("bt_steps", 0) + 1
    else:
        write_index, rows, counts = _index_rows(hidden_states, allocator, kwargs, topk_length)
    # the list width is CB's bucket of the longest context: only the opaque attention op reads it, so one artifact
    # serves every bucket (shapes stay static otherwise)
    torch._dynamo.mark_dynamic(rows, 1)

    if KERNELS:
        return _run_layers_fused(model, state, hidden_states, position_embeddings, write_index, rows, counts, w_kv_b)

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
                    layer, hidden_states, cos, sin, state["views"][i], write_index, rows, counts, w_kv_b[i],
                    state["h_pad"], tp,
                )
            continue
        hidden_states = fn(
            layer, hidden_states, cos, sin, state["views"][i], write_index, rows, counts, w_kv_b[i], state["h_pad"], tp
        )
        if timings is not None:
            timings.append(round(time.perf_counter() - start, 3))
    STATS["steps"] += 1
    return hidden_states


def _block_table_rows(model, cache, allocator, hidden_states, kwargs, topk_length):
    """[gap3] Rows of a block-table decode batch, once per step: the row each query token writes its latents to, the
    rows of its context (-1 = empty slot) and their count. One query token per request, or 1 + k with CB_MTP=spec
    (`_proto_block_table_token_state`). Same arithmetic as `_proto_block_table_rows` with the indexer scoring
    skipped. The state is also left on the cache, where the regular path of a later layer of the same step (the MTP
    module's) reads it."""
    from . import modeling_glm_moe_dsa as m

    page = allocator.tokens_per_page
    last_safe_row = allocator.cache_tensor.shape[0] - allocator.tokens_per_block + page - 1
    if m._CB_MTP_SPEC:
        state = m._proto_block_table_token_state(
            kwargs["block_table"][allocator.index],
            kwargs["cu_seq_lens_q"],
            kwargs["cu_seq_lens_k"][allocator.layer_type],
            kwargs["max_length_k"][allocator.layer_type],
            page,
            last_safe_row,
            allocator.write_trash_index,
            hidden_states.shape[1],
        )
    else:
        state = m._proto_block_table_row_state(
            kwargs["block_table"][allocator.index],
            kwargs["cu_seq_lens_k"][allocator.layer_type],
            kwargs["max_length_k"][allocator.layer_type],
            page,
            last_safe_row,
            allocator.write_trash_index,
        )
    cache._proto_bt_state = state
    write_rows, rows_all, visible = state
    if write_rows.shape[0] != hidden_states.shape[1]:
        raise RuntimeError(f"FUSE_COMPILE: {write_rows.shape[0]} write rows for {hidden_states.shape[1]} query tokens")
    rows = torch.where(visible, rows_all, -1).to(torch.int32)
    num_selected = min(model.config.index_topk, rows_all.shape[1])
    counts = visible.sum(-1).clamp(max=num_selected).to(torch.int32) if topk_length else None
    if rows.shape[-1] % 128:  # the kernel takes top-k lists in multiples of 128, -1 marks an empty slot
        rows = F.pad(rows, (0, 128 - rows.shape[-1] % 128), value=-1)
    return write_rows, rows, counts


def _index_rows(hidden_states, allocator, kwargs, topk_length):
    """Rows of a batch on the index path (wp2-fuse, BASE)."""
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
    # the valid rows of a list are its first `n_visible` entries: the kernel stops there instead of walking the -1
    counts = n_visible.clamp(min=1, max=rows.shape[-1]).to(torch.int32) if topk_length else None
    return write_index, rows, counts


def _qkv_a_weights(model):
    """[gap3] Per layer, the fused q_a + kv_a (+ wk) weight of GEMM_FUSE_QKV_A when it may run now (read from the
    model's own modules, not the TP twins, every step: training may unfuse them), else None."""
    from . import modeling_glm_moe_dsa as m

    layers = model.layers[: model.config.num_hidden_layers]
    if not m._GEMM_FUSE_QKV_A:
        return [None] * len(layers)
    weights = []
    for layer in layers:
        a = layer.self_attn
        current = (a.q_a_proj, a.kv_a_proj_with_mqa) + (() if a.indexer is None else (a.indexer.wk,))
        weights.append(m._gemm_fused_weight(a, "_gemm_qkv_a", current))
    return weights


def _run_layers_fused(model, state, hidden_states, position_embeddings, write_index, rows, counts, w_kv_b):
    """Layer loop of FUSE_KERNELS=1: every layer hands the next one its output and that output already normed."""
    cos, sin = position_embeddings
    layers = state["layers"]
    attn = layers[0].self_attn
    num_tokens = hidden_states.shape[1]
    num_heads = w_kv_b[0].shape[0]
    key = (num_tokens, max(state["h_pad"], num_heads))
    q_abs = state["q_abs"].get(key)
    if q_abs is None:  # zeroed once and kept with the cache state: the padded heads stay zero, CUDA graphs keep its address
        q_abs = hidden_states.new_zeros(key[0], key[1], attn.kv_lora_rank + attn.qk_rope_head_dim)
        state["q_abs"][key] = q_abs
    norms = [layer.input_layernorm for layer in layers[1:]] + [model.norm]
    qkv_a = _qkv_a_weights(model)
    fn = _layer_fn()
    tp = state["tp"]
    timings = STATS.get("layer_seconds")
    ranges = STATS.get("ranges")
    residual = hidden_states
    hidden_states = layers[0].input_layernorm(hidden_states)
    for i, layer in enumerate(layers):
        if timings is not None:
            start = time.perf_counter()
        args = (
            layer, residual, hidden_states, cos, sin, state["views"][i], write_index, rows, counts, w_kv_b[i], q_abs,
            norms[i].weight, norms[i].variance_epsilon, tp, qkv_a[i],
        )
        if ranges:
            with torch.autograd.profiler.record_function(f"M:L{i}:layer"):
                residual, hidden_states = fn(*args)
            continue
        residual, hidden_states = fn(*args)
        if timings is not None:
            timings.append(round(time.perf_counter() - start, 3))
    STATS["steps"] += 1
    return residual  # the model applies its final norm itself (the last kernel's normed output is not used)
