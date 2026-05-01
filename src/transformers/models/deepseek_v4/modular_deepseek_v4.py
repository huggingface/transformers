# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_experts_implementation
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm, apply_rotary_pos_emb_interleave
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..gpt_oss.modeling_gpt_oss import GptOssExperts, eager_attention_forward
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel, MixtralTopKRouter
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V4 wraps :func:`~transformers.models.deepseek_v3.modeling_deepseek_v3.apply_rotary_pos_emb_interleave`
    with a permute-back so the rope slice exits in the same interleaved
    ``[a0, b0, a1, b1, …]`` layout it came in with.

    V3's helper restages interleaved pairs into the halves layout
    (``[a0, a1, …, b0, b1, …]``) so it can run llama's half-split RoPE primitive,
    and leaves the result in that layout — fine for V3 because V3 is MLA: V has
    its own ``v_head_dim`` and never carries a rope slice, so the post-rotation
    layout of Q / K only matters for the dot product (which is invariant under a
    consistent permutation of channels on both sides).

    V4 is shared-KV MQA: V is the same tensor as K, so V's rope slice picks up
    the rotation too — and then the attention sum, the per-head ``wo_a``
    grouped projection, and ``wo_b`` all consume that rope slice as part of
    their input. Those weights were trained against the V4-Flash reference
    (``inference/model.py:apply_rotary_emb`` does ``view_as_complex``-style
    rotation in place, preserving the interleaved layout), so we have to put
    the channels back where they were before passing to ``wo_a`` — otherwise the
    grouped projection sees its inputs scrambled and ``wo_b(wo_a(...))`` collapses.
    """
    q, k = apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    def _halves_to_interleave(x: torch.Tensor) -> torch.Tensor:
        # Inverse of V3's ``view(d/2, 2).transpose(-1, -2)``: ``[a0, …, b0, …]`` →
        # ``[a0, b0, a1, b1, …]``.
        b, h, s, d = x.shape
        return x.view(b, h, s, 2, d // 2).transpose(-1, -2).reshape(b, h, s, d)

    return _halves_to_interleave(q), _halves_to_interleave(k)


logger = logging.get_logger(__name__)


DEEPSEEK_V4_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)


_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4-Flash-Base")
@strict
class DeepseekV4Config(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*):
        V3 MLA expert-group count. Kept for config compat;
        unused by V4 (no expert groups).
    first_k_dense_replace (`int`, *optional*):
        V3 field — the first ``k`` MoE layers
        to replace with dense FFNs. Kept for config compat; V4 uses hash routing
        (``num_hash_layers``) instead.
    rope_interleave (`bool`, *optional*):
        V3 flag — whether to interleave rope dims.
        Kept for config compat; V4's RoPE is non-interleaved (rope-first head layout).
    scoring_func (`str`):
        Router activation — ``sqrtsoftplus``, ``softmax``, or ``sigmoid``.
    rope_theta (`float`):
        RoPE base for the main self-attention rotary.
    layer_types (`list[str]`):
        Per-layer attention schedule with values from
        ``{"compressed_sparse_attention", "heavily_compressed_attention"}``.
        V4-Pro default: 2× HCA bootstrap + interleaved CSA / HCA.
    compress_rate_csa (`int`):
        m, the CSA compression rate (default 4).
    compress_rate_hca (`int`):
        m', the HCA compression rate (default 128).
    compress_rope_theta (`float`):
        RoPE base for the compressed branches (paired with
        ``rope_scaling`` for YaRN).
    hc_mult (`int`):
        Manifold-Constrained Hyper-Connection (mHC) expansion factor n_hc
        (always active; Section 2.2).
    hc_sinkhorn_iters (`int`):
        Sinkhorn-Knopp iterations t_max for the mHC residual
        mapping projection onto doubly-stochastic matrices.
    hc_eps (`float`):
        Numerical floor for the Sinkhorn-Knopp normalization.
    num_hash_layers (`int`):
        First N MoE layers route via a frozen ``tid2eid[input_ids]`` lookup.
    swiglu_limit (`float`):
        Clip routed experts' gate/up pre-activations.
    sliding_window (`int`):
        Local window size n_win used in every attention block's
        sliding-window branch.
    o_groups (`int`):
        Number of head-groups g in the grouped output projection
        (paper §2.3.1, "Grouped Output Projection").
    o_lora_rank (`int`):
        Per-group intermediate dim d_g in the grouped output projection.
    index_n_heads (`int`):
        Number of indexer query heads n_h^I (paper §2.3.1, eq. 14).
    index_head_dim (`int`):
        Indexer head dim c^I (paper §2.3.1).
    index_topk (`int`):
        Number of compressed entries per query the Lightning Indexer
        keeps via top-k (paper §2.3.1, eq. 17).
    num_nextn_predict_layers (`int`):
        MTP layer count in the upstream checkpoint
        (not instantiated here).
    partial_rotary_factor (`float`, *optional*):
        Fraction of head_dim that gets RoPE.
        Defaults to ``qk_rope_head_dim / head_dim`` so cos/sin sizes to ``qk_rope_head_dim``.
    """

    model_type = "deepseek_v4"
    attribute_map = {"num_local_experts": "n_routed_experts"}

    base_model_tp_plan = {
        "layers.*.self_attn.wq_a": "colwise",
        "layers.*.self_attn.wq_b": "colwise",
        "layers.*.self_attn.wkv": "colwise",
        "layers.*.self_attn.wo_a": "rowwise",
        "layers.*.self_attn.wo_b": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 1048576
    rope_theta: float | int = 10000.0

    layer_types: list[str] | None = None
    compress_rate_csa: int = 4
    compress_rate_hca: int = 128
    compress_rope_theta: float | int = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1

    # V3 fields kept ``None`` so the V3-style MLA paths in inherited configs never fire
    # (V4 doesn't use MLA — it uses shared-KV MQA via ``wkv`` directly).
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    n_group: int | None = None
    topk_group: int | None = None
    first_k_dense_replace: int | None = None
    rope_interleave: bool | None = True

    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    rope_parameters: RopeParameters | dict | None = None
    partial_rotary_factor: float | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    def validate_layer_type(self):
        """V4 narrows the global ``ALLOWED_LAYER_TYPES`` to the two block types it actually
        ships with, on top of the standard length / type-membership checks.
        """
        if self.layer_types is None or self.num_hidden_layers is None:
            return
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must equal "
                f"`len(layer_types)` ({len(self.layer_types)})."
            )
        bad = [layer_type for layer_type in self.layer_types if layer_type not in DEEPSEEK_V4_LAYER_TYPES]
        if bad:
            raise ValueError(
                f"`layer_types` entries must be one of {DEEPSEEK_V4_LAYER_TYPES} for DeepSeek-V4; got {bad}."
            )

    def __post_init__(self, **kwargs):
        compress_ratios = kwargs.pop("compress_ratios", None)
        PreTrainedConfig.__post_init__(self, **kwargs)
        n = self.num_hidden_layers
        if self.layer_types is None and compress_ratios is not None:
            # Translate the V4 checkpoint's per-layer integer ``compress_ratios`` into the
            # named ``layer_types`` schedule (0 = sliding-only, 4 = CSA, 128 = HCA).
            self.layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in compress_ratios]
        if self.layer_types is None:
            # V4-Pro default: two HCA bootstrap layers, then CSA / HCA interleaved.
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(n - 2, 0))
            ]
            head = ["heavily_compressed_attention"] * min(n, 2)
            self.layer_types = head + interleave
        self.layer_types = list(self.layer_types[:n])
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = self.qk_rope_head_dim / self.head_dim
        # Normalize rope_parameters into a per-rope-type dict ``{"main": {...}, "compress": {...}}``
        # (Gemma3 pattern, keys are *rope-type* labels — unrelated to ``layer_types``).
        # Idempotent across save/load: round-tripping preserves structure.
        #
        # By the time we get here :class:`PreTrainedConfig` has already run
        # :meth:`RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`, which folds the
        # checkpoint's legacy top-level ``rope_scaling`` block into ``self.rope_parameters``
        # as a flat dict (``rope_type``, ``factor``, ``beta_fast``, ``beta_slow``,
        # ``original_max_position_embeddings``, …). The block ships under
        # ``rope_scaling`` in :attr:`config.json` and never appears as a top-level kwarg
        # for us to intercept before the mixin runs — the mixin always wins. We just
        # split that flat dict into the two rope-type buckets.
        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
        else:
            # Build the per-rope-type dict ``{"main", "compress"}``. The flat ``rp``
            # already carries any YaRN params the checkpoint shipped under top-level
            # ``rope_scaling`` (folded in by ``RotaryEmbeddingConfigMixin``). We propagate
            # them into both buckets — the difference between the two is just the
            # ``rope_theta`` base (the model's main attention uses ``rope_theta=10000``,
            # the compressor / indexer uses ``compress_rope_theta=160000``).
            base = {k: v for k, v in rp.items() if k not in ("main", "compress")}
            base.setdefault("rope_theta", self.rope_theta)
            base["partial_rotary_factor"] = self.partial_rotary_factor
            base.setdefault("rope_type", "default")
            main = dict(base)
            compress = {**base, "rope_theta": self.compress_rope_theta}
            self.rope_parameters = {"main": main, "compress": compress}


class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV4RotaryEmbedding(Gemma3RotaryEmbedding):
    """Multi-layer-type rotary embedding (Gemma3 pattern). Holds two ``inv_freq``
    buffers — ``"main"`` for self-attention (``rope_theta``) and ``"compress"`` for
    the Compressor / Indexer (``compress_rope_theta``). Both honour
    ``partial_rotary_factor`` so cos/sin is sized to ``qk_rope_head_dim`` rather than
    the full ``head_dim``. ``forward(x, position_ids, layer_type=...)`` (inherited
    from :class:`Gemma3RotaryEmbedding`) picks one.

    The ``layer_types`` here are the *rope* layer types (``"main"`` / ``"compress"``),
    keys of ``config.rope_parameters``. They are unrelated to ``config.layer_types``,
    which lists the per-decoder-block attention type.
    """

    layer_types = ("main", "compress")

    def __init__(self, config: "DeepseekV4Config", device=None):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = {}
        for layer_type in self.layer_types:
            params = config.rope_parameters.get(layer_type)
            if params is None:
                continue
            self.rope_type[layer_type] = params.get("rope_type", "default")
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, scaling = rope_init_fn(config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", scaling)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        # V4 honours ``partial_rotary_factor`` so cos/sin sizes to ``qk_rope_head_dim``.
        params = config.rope_parameters[layer_type]
        base = params["rope_theta"]
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        factor = params.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0


def _sliding_kv_update(
    cache_layer: "DynamicSlidingWindowLayer", key_states: torch.Tensor, value_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared sliding-window K=V update body. V4 uses shared-KV MQA, so ``keys`` and
    ``values`` point to the same storage on every layer; both V4 cache layer types
    (HCA / CSA) call this from their ``update``."""
    if not cache_layer.is_initialized:
        cache_layer.lazy_initialization(key_states, value_states)
        cache_layer.values = cache_layer.keys
    cache_layer.cumulative_length += key_states.shape[-2]
    full = torch.cat([cache_layer.keys, key_states], dim=-2)
    cache_layer.keys = full[:, :, -cache_layer.sliding_window + 1 :, :]
    cache_layer.values = cache_layer.keys
    return full, full


def _update_window_buffer(
    buffer_kv: torch.Tensor | None,
    buffer_gate: torch.Tensor | None,
    kv: torch.Tensor,
    gate: torch.Tensor,
    compress_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge a still-buffered tail with freshly projected ``(kv, gate)`` and split off
    the longest window-aligned chunk. Used by both the compressor- and indexer-side
    window buffers; tokens past the last full window stay in the buffer until the
    next call rounds them out to a multiple of ``compress_rate``."""
    if buffer_kv is not None and buffer_kv.shape[1]:
        kv = torch.cat([buffer_kv, kv], dim=1)
        gate = torch.cat([buffer_gate, gate], dim=1)
    usable = (kv.shape[1] // compress_rate) * compress_rate
    return kv[:, :usable], gate[:, :usable], kv[:, usable:], gate[:, usable:]


def _append_to_pool(pool: torch.Tensor | None, new_pooled: torch.Tensor) -> torch.Tensor:
    """Append freshly emitted compressed entries to a running pool, returning the
    full pool (or an empty tensor if nothing has been pooled yet)."""
    if new_pooled.shape[1] > 0:
        return new_pooled if pool is None else torch.cat([pool, new_pooled], dim=1)
    if pool is None:
        return new_pooled.new_zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]))
    return pool


class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    """Cache layer for HCA blocks (paper §2.3.2). Holds the long-range compressor's
    buffer / pool / count on top of the sliding-window K=V branch. HCA uses
    *non-overlapping* windows, so there is **no** overlap state, and HCA has **no**
    indexer either.

    Fields on top of :class:`DynamicSlidingWindowLayer`:

      * ``compressor_pool`` — the running list of compressed KV entries emitted so
        far (one per ``compress_rate_hca`` source tokens; the long-range KVs the
        attention concatenates onto its sliding-window keys / values).
      * ``compressor_buffer_kv`` / ``compressor_buffer_gate`` — source tokens that
        arrived between two full windows; once the buffer hits ``compress_rate_hca``
        tokens the compressor closes a window, emits one pooled entry, and drains
        the buffer.
      * ``compressor_pool_count`` — number of compressed entries emitted so far,
        so ``compressor_pool_count * compress_rate_hca`` is the absolute position
        of the *next* window's first source token.

    The class-level ``layer_type`` auto-registers this class with
    :data:`LAYER_TYPE_CACHE_MAPPING` so :class:`DynamicCache` builds it on its own
    when ``config.layer_types[i] == "heavily_compressed_attention"``.
    """

    layer_type = "heavily_compressed_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rate_hca
        self.compressor_buffer_kv: torch.Tensor | None = None
        self.compressor_buffer_gate: torch.Tensor | None = None
        self.compressor_pool: torch.Tensor | None = None
        self.compressor_pool_count = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        return _sliding_kv_update(self, key_states, value_states)

    def update_compressor(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Merge the freshly projected ``(kv, gate)`` (paper §2.3.2 eqs. 20–21:
        ``C = H·W^{KV}``, ``Z = H·W^Z``) with the buffered tail from prior calls and
        return the longest window-aligned chunk that's ready to pool, plus the
        absolute source-token position of that chunk's first window. The returned
        chunk is softmax-pooled by the compressor with ``position_bias`` to emit one
        compressed entry per window of ``compress_rate_hca`` tokens (eqs. 22–23)."""
        first_pool_position = self.compressor_pool_count * self.compress_rate
        chunk_kv, chunk_gate, self.compressor_buffer_kv, self.compressor_buffer_gate = _update_window_buffer(
            self.compressor_buffer_kv, self.compressor_buffer_gate, kv, gate, self.compress_rate
        )
        return chunk_kv, chunk_gate, first_pool_position

    def update_compressor_pool(self, new_pooled: torch.Tensor) -> torch.Tensor:
        """Append freshly emitted compressed entries to ``compressor_pool``
        (``C^{Comp}``, paper §2.3.2 eq. 23) and return the full pool. Bumps
        ``compressor_pool_count`` so the next ``update_compressor`` call knows the
        absolute source-token position of its first window."""
        self.compressor_pool = _append_to_pool(self.compressor_pool, new_pooled)
        self.compressor_pool_count += new_pooled.shape[1]
        return self.compressor_pool


class DeepseekV4CSACache(DynamicSlidingWindowLayer):
    """Cache layer for CSA blocks (paper §2.3.1). Holds two parallel sets of
    buffer / pool / count / overlap state on top of the sliding-window K=V branch:

      * **compressor side** — the main-branch ``head_dim`` pool (the long-range KVs
        the attention concatenates after top-k indexer selection).
      * **indexer side** — the Lightning Indexer's smaller ``index_head_dim`` pool
        (the keys ``K^{IComp}`` that queries score against to pick the top-k blocks,
        eqs. 14–17). Kept separate from the compressor pool because the head dim
        differs.

    Both sides use **overlapping** windows of stride ``compress_rate_csa`` and width
    ``2 * compress_rate_csa`` (paper §2.3.1), so each side also keeps an
    ``*_overlap_kv`` / ``*_overlap_gate`` pair holding the last full window's
    projected ``(kv, gate)`` so the next forward call's first window can stitch in
    its low-channel slice as the prior contribution.

    The class-level ``layer_type`` auto-registers this class with
    :data:`LAYER_TYPE_CACHE_MAPPING` so :class:`DynamicCache` builds it on its own
    when ``config.layer_types[i] == "compressed_sparse_attention"``.
    """

    layer_type = "compressed_sparse_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rate_csa
        # Compressor side
        self.compressor_buffer_kv: torch.Tensor | None = None
        self.compressor_buffer_gate: torch.Tensor | None = None
        self.compressor_pool: torch.Tensor | None = None
        self.compressor_pool_count = 0
        self.compressor_overlap_kv: torch.Tensor | None = None
        self.compressor_overlap_gate: torch.Tensor | None = None
        # Indexer side (parallel state at ``index_head_dim``)
        self.indexer_buffer_kv: torch.Tensor | None = None
        self.indexer_buffer_gate: torch.Tensor | None = None
        self.indexer_pool: torch.Tensor | None = None
        self.indexer_pool_count = 0
        self.indexer_overlap_kv: torch.Tensor | None = None
        self.indexer_overlap_gate: torch.Tensor | None = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        return _sliding_kv_update(self, key_states, value_states)

    def update_compressor(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compressor-side window buffer (paper §2.3.1 main-branch pool, eqs. 9–12).
        Same window-aligned tail-buffering as HCA, but at the CSA cadence
        (``compress_rate_csa``)."""
        first_pool_position = self.compressor_pool_count * self.compress_rate
        chunk_kv, chunk_gate, self.compressor_buffer_kv, self.compressor_buffer_gate = _update_window_buffer(
            self.compressor_buffer_kv, self.compressor_buffer_gate, kv, gate, self.compress_rate
        )
        return chunk_kv, chunk_gate, first_pool_position

    def update_compressor_pool(self, new_pooled: torch.Tensor) -> torch.Tensor:
        """Append freshly emitted entries to the CSA compressor pool (the
        ``C^{Comp}`` running list at ``head_dim``, eqs. 11–12)."""
        self.compressor_pool = _append_to_pool(self.compressor_pool, new_pooled)
        self.compressor_pool_count += new_pooled.shape[1]
        return self.compressor_pool

    def get_compressor_overlap(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.compressor_overlap_kv, self.compressor_overlap_gate

    def set_compressor_overlap(self, kv: torch.Tensor, gate: torch.Tensor) -> None:
        self.compressor_overlap_kv = kv
        self.compressor_overlap_gate = gate

    def update_indexer(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Indexer-side mirror of :meth:`update_compressor` (paper §2.3.1, "Lightning
        Indexer for Sparse Selection"). Same logic at the smaller ``index_head_dim``
        — the small-head pool keys ``K^{IComp}`` (eq. 14's ``W^{IUQ}`` complement on
        the key side) that the indexer scores queries against to pick the top-k
        blocks (eqs. 15–17). Buffer / pool / count are kept separate from the
        compressor's state because the head dim differs."""
        first_pool_position = self.indexer_pool_count * self.compress_rate
        chunk_kv, chunk_gate, self.indexer_buffer_kv, self.indexer_buffer_gate = _update_window_buffer(
            self.indexer_buffer_kv, self.indexer_buffer_gate, kv, gate, self.compress_rate
        )
        return chunk_kv, chunk_gate, first_pool_position

    def update_indexer_pool(self, new_pooled: torch.Tensor) -> torch.Tensor:
        """Append freshly emitted entries to the indexer pool ``K^{IComp}`` (paper
        §2.3.1 eq. 16: the keys against which the ``q^I_t`` queries score for top-k
        selection). Same cadence as the compressor pool — one entry per
        ``compress_rate_csa`` source tokens — but at ``index_head_dim``."""
        self.indexer_pool = _append_to_pool(self.indexer_pool, new_pooled)
        self.indexer_pool_count += new_pooled.shape[1]
        return self.indexer_pool

    def get_indexer_overlap(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.indexer_overlap_kv, self.indexer_overlap_gate

    def set_indexer_overlap(self, kv: torch.Tensor, gate: torch.Tensor) -> None:
        self.indexer_overlap_kv = kv
        self.indexer_overlap_gate = gate


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear used by the V4 grouped output projection
    (paper §2.3.1, "Grouped Output Projection"; HCA reuses the same scheme,
    §2.3.2). With ``num_attention_heads = n_h`` and per-head dim ``c``, the core
    attention's stacked output is ``c·n_h``-dim, which is *very* large for V4
    (V4-Flash: c=512, n_h=64 → 32768; V4-Pro: c=512, n_h=128 → 65536). A direct
    ``c·n_h → hidden_size`` projection would dominate the per-token cost.

    The paper sidesteps that by splitting the n_h heads into ``g`` groups, projecting
    each ``c·n_h/g``-dim group independently to a ``d_g``-dim intermediate output
    (with ``d_g < c·n_h/g``), and then mixing the resulting ``g·d_g`` vector to
    ``hidden_size`` through a single follow-up linear (``self_attn.wo_b``). This
    module owns the per-group block (``self_attn.wo_a``).

    The ``weight`` parameter is shaped like a standard ``nn.Linear``
    (``[out_features, in_features_per_group]``) so quantizers keyed on
    ``nn.Linear.weight`` still pick it up; ``forward`` does the per-group ``bmm``.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_features_per_group]
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        x = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(x, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


class DeepseekV4Indexer(nn.Module):
    """Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
    Attention (CSA) to pick the top-k compressed KV blocks per query. The indexer
    runs its own scaled-down compressor at ``index_head_dim`` over the same windows
    as the outer CSA compressor, then scores queries against the pooled keys with
    ``∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`` and keeps the top ``index_topk``
    indices.

    The indexer has its own rotary because it applies RoPE to two sets of tensors:

      * **pool keys** at deterministic positions ``i * compress_rate + first_pool_position``,
      * **queries** at the model's current ``position_ids`` (variable per forward).

    Both must use the same theta as the outer compressor (``compress_rope_theta``) so
    query/key inner products are translation-invariant in the standard rope sense — if
    they used different thetas the score ``q · k`` would carry a residual position-
    dependent skew. We can't precompute cos/sin once at init because the query
    positions vary per call, so the indexer owns a rotary embedding and calls it with
    ``layer_type="compress"`` twice per forward (once for pool keys, once for queries).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rate_csa
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        # The indexer always pools with the CSA cadence (``compress_rate=4``), so its
        # inner pool runs the same overlapping-window scheme as :class:`DeepseekV4CSACompressor`
        # (paper §2.3.1) — ``coff = 2`` everywhere on the pool branch.
        self.coff = 2
        self.wkv = nn.Linear(config.hidden_size, self.coff * self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.coff * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.coff * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None

        # --- Pool side: same overlapping windows as the outer CSA compressor, at index_head_dim ---
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_pool_position = kv[:, :usable], gate[:, :usable], 0
            prior_kv, prior_gate = None, None
        else:
            chunk_kv, chunk_gate, first_pool_position = cache_layer.update_indexer(kv, gate)
            prior_kv, prior_gate = cache_layer.get_indexer_overlap()
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, self.coff * self.head_dim)
            chunk_gate = chunk_gate.view(
                batch, n_windows, self.compress_rate, self.coff * self.head_dim
            ) + self.position_bias.to(chunk_gate.dtype)
            if cache_layer is not None:
                cache_layer.set_indexer_overlap(chunk_kv[:, -1].clone(), chunk_gate[:, -1].clone())
            chunk_kv, chunk_gate = _overlap_pool(chunk_kv, chunk_gate, prior_kv, prior_gate, self.head_dim)
            new_pooled = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_rate + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            cos, sin = self.rotary_emb(new_pooled, position_ids=positions, layer_type="compress")
            # V4-Flash places the rotary slice at the *end* of each head (matches the
            # reference's ``x[..., -rd:]`` indexing) — wkv weight is laid out [nope|rope]
            # so the rotary half is the trailing ``rope_head_dim`` channels.
            pool_nope, pool_rope = new_pooled[..., : -self.rope_head_dim], new_pooled[..., -self.rope_head_dim :]
            pool_rope, _ = apply_rotary_pos_emb(
                pool_rope.unsqueeze(1), torch.zeros_like(pool_rope.unsqueeze(1)), cos, sin
            )
            new_pooled = torch.cat([pool_nope, pool_rope.squeeze(1)], dim=-1)
        else:
            new_pooled = chunk_kv.new_zeros((batch, 0, self.head_dim))
        pooled_kv = new_pooled if cache_layer is None else cache_layer.update_indexer_pool(new_pooled)

        # --- Query side ---
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress")
        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_nope, q_rope = q[..., : -self.rope_head_dim], q[..., -self.rope_head_dim :]
        q_rope, _ = apply_rotary_pos_emb(q_rope, torch.zeros_like(q_rope), cos_q, sin_q)
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)

        # --- Score: ReLU(q·kᵀ) * weights, then top-k ---
        scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        topk = min(self.index_topk, pooled_kv.shape[1])
        return index_scores.topk(topk, dim=-1).indices


# -----------------------------------------------------------------------------
# Compressors — :class:`DeepseekV4HCACompressor` and :class:`DeepseekV4CSACompressor`
# are independent. They share the same softmax-gated window-pool primitive but differ
# in three ways that we keep on each class explicitly: HCA pools non-overlapping
# windows with ``coff = 1`` and has no indexer, CSA pools overlapping windows with
# ``coff = 2`` and runs a Lightning Indexer on top of the pool.
# -----------------------------------------------------------------------------


def _overlap_pool(
    chunk_kv: torch.Tensor,
    chunk_gate: torch.Tensor,
    prior_kv: torch.Tensor | None,
    prior_gate: torch.Tensor | None,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand ``[B, n_win, ratio, 2*head_dim]`` chunks into ``[B, n_win, 2*ratio, head_dim]``
    by stitching each window's *low-channel* slice onto the *high-channel* slice of the
    prior window — matching the V4-Flash reference (``Compressor.overlap_transform``).

    Each pooled output thus mixes ``ratio`` *current* source tokens (high half of the
    learned 2d split) with ``ratio`` *previous* source tokens (low half), so windows
    have width ``2*ratio`` but stride ``ratio`` (paper §2.3.1). For window 0, the prior
    half is filled with zero (kv) / ``-inf`` (gate, so its softmax weight is exactly 0),
    unless ``prior_kv`` / ``prior_gate`` carry the last full window from a previous
    forward call — in which case its low-channel slice slots into row ``[0, :ratio]``.
    """
    batch, n_windows, ratio, _ = chunk_kv.shape
    new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, head_dim))
    new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, head_dim), float("-inf"))
    new_kv[:, :, ratio:] = chunk_kv[..., head_dim:]
    new_gate[:, :, ratio:] = chunk_gate[..., head_dim:]
    if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :head_dim]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :head_dim]
    if prior_kv is not None and prior_gate is not None:
        new_kv[:, 0, :ratio] = prior_kv[..., :head_dim].to(new_kv.dtype)
        new_gate[:, 0, :ratio] = prior_gate[..., :head_dim].to(new_gate.dtype)
    return new_kv, new_gate


def _rope_pool(
    pooled: torch.Tensor, rotary_emb: nn.Module, positions: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    """Apply RoPE to the trailing ``rope_head_dim`` slice of each pooled entry at its
    deterministic absolute position. V4-Flash lays out each head as
    ``[nope | rope]`` (matches the reference's ``x[..., -rd:]`` indexing) so the
    rotary half is the trailing channels."""
    cos, sin = rotary_emb(pooled, position_ids=positions, layer_type="compress")
    pool_nope, pool_rope = pooled[..., :-rope_head_dim], pooled[..., -rope_head_dim:]
    pool_rope, _ = apply_rotary_pos_emb(pool_rope.unsqueeze(1), torch.zeros_like(pool_rope.unsqueeze(1)), cos, sin)
    return torch.cat([pool_nope, pool_rope.squeeze(1)], dim=-1)


class DeepseekV4HCACompressor(nn.Module):
    """Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20–23). Pools
    every ``compress_rate_hca`` (m'=128) source tokens into a single compressed KV
    entry with **non-overlapping** windows — no overlap state, no indexer.

    The three building blocks (paper notation in parentheses):

      * **kv** = ``wkv(hidden_states)`` — head-dim KV projection ``C ∈ R^{n×c}``
        (eq. 20). Doubles as both key and value (shared-KV MQA).
      * **gate** = ``wgate(hidden_states)`` — head-dim compression weights
        ``Z ∈ R^{n×c}`` (eq. 21). Combined with ``position_bias`` and softmaxed per
        window to produce the convex combination that mixes ``compress_rate_hca``
        source KVs into one pooled entry.
      * **pool** — running list of compressed KV entries (``C^{Comp}``, eq. 23).
        Lives on :class:`DeepseekV4HCACache`; the in-flight buffer of tokens that
        haven't yet filled a window lives there too.

    Each closed window of m' tokens produces one pooled entry:
    ``C^{Comp}_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j``. RoPE on the trailing
    ``rope_head_dim`` slice is applied at the deterministic absolute position
    ``i * compress_rate_hca + first_pool_position`` so cross-call concatenation stays
    causality-correct. Returns the running pool ``[B, 1, T, head_dim]``.

    When ``past_key_values is None`` (a checkpoint replay zeroes the cache to break
    the grad-cache loop), runs in stateless single-shot mode: pool every complete
    window from ``hidden_states`` and discard the remainder.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rate_hca
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        # ``q_residual`` / ``position_ids`` are unused — the uniform forward signature
        # lets :class:`DeepseekV4Attention` call either compressor without branching.
        batch, _, _ = hidden_states.shape
        cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_pool_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_pool_position = cache_layer.update_compressor(kv, gate)
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, self.head_dim)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, self.head_dim) + self.position_bias.to(
                chunk_gate.dtype
            )
            new_pooled = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_rate + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            new_pooled = _rope_pool(new_pooled, self.rotary_emb, positions, self.rope_head_dim)
        else:
            new_pooled = chunk_kv.new_zeros((batch, 0, self.head_dim))
        if cache_layer is None:
            return new_pooled.unsqueeze(1)
        return cache_layer.update_compressor_pool(new_pooled).unsqueeze(1)


class DeepseekV4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9–17). Pools every
    ``compress_rate_csa`` (m=4) source tokens with **overlapping** windows — stride
    ``compress_rate_csa`` and effective width ``2 * compress_rate_csa`` — and runs a
    Lightning Indexer on top of the pool that scores queries with
    ``∑_h w_{t,h} · ReLU(q_{t,h} · K^{IComp}_s)`` to gather the top ``index_topk``
    entries per query before they reach core attention.

    Compared to :class:`DeepseekV4HCACompressor` the differences are explicit:

      * ``wkv`` / ``wgate`` / ``position_bias`` project to **2 × head_dim** (the
        learned channel split — high half pools into the current window, low half
        pools into the next window's overlap with this one, see :func:`_overlap_pool`).
      * The cache layer's ``compressor_overlap_*`` state carries the last full
        window across forward calls.
      * A :class:`DeepseekV4Indexer` sub-module gathers the top-``index_topk`` pool
        entries per query (paper §2.3.1, "Lightning Indexer for Sparse Selection").
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rate_csa
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        # ``2 * head_dim`` because windows overlap: each pooled entry is a softmax-gated
        # convex combination of ``compress_rate_csa`` *current* tokens (high-channel half)
        # mixed with ``compress_rate_csa`` *previous* tokens (low-channel half). The
        # learned channel split happens in :func:`_overlap_pool`.
        self.wkv = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_pool_position = kv[:, :usable], gate[:, :usable], 0
            prior_kv, prior_gate = None, None
        else:
            chunk_kv, chunk_gate, first_pool_position = cache_layer.update_compressor(kv, gate)
            prior_kv, prior_gate = cache_layer.get_compressor_overlap()
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, 2 * self.head_dim)
            chunk_gate = chunk_gate.view(
                batch, n_windows, self.compress_rate, 2 * self.head_dim
            ) + self.position_bias.to(chunk_gate.dtype)
            if cache_layer is not None:
                # Persist the *raw* last full window (gate already biased) so the next
                # forward call's first window can read its low-channel slice as prior.
                cache_layer.set_compressor_overlap(chunk_kv[:, -1].clone(), chunk_gate[:, -1].clone())
            chunk_kv, chunk_gate = _overlap_pool(chunk_kv, chunk_gate, prior_kv, prior_gate, self.head_dim)
            new_pooled = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_rate + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            new_pooled = _rope_pool(new_pooled, self.rotary_emb, positions, self.rope_head_dim)
        else:
            new_pooled = chunk_kv.new_zeros((batch, 0, self.head_dim))
        pooled = (
            new_pooled.unsqueeze(1)
            if cache_layer is None
            else cache_layer.update_compressor_pool(new_pooled).unsqueeze(1)
        )
        # Lightning Indexer: gather top-``index_topk`` pool entries per query.
        topk = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)  # [B, S, k]
        expanded = pooled.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
        return torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


# -----------------------------------------------------------------------------
# Attention with sink.
# -----------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """V4 attention block (paper §2.3). Single class for all three layer types — the
    only thing that varies is the long-range branch (the ``compressor`` sub-module);
    the surrounding QKV / RoPE / sink / sliding-window / output projection is
    identical. The three layer types are dispatched by ``COMPRESSOR_CLASSES``:

      * ``sliding_attention``: ``compressor = None``; only the local sliding-window
        K=V branch ("Full Attention").
      * ``compressed_sparse_attention``: :class:`DeepseekV4CSACompressor` —
        low-compression overlapping-window pool plus a Lightning Indexer that keeps
        the top-``index_topk`` pool entries per query (paper §2.3.1).
      * ``heavily_compressed_attention``: :class:`DeepseekV4HCACompressor` —
        high-compression non-overlapping-window pool, no indexer (paper §2.3.2).

    Block components (paper §2.3.3):

      * Shared-KV Multi-Query Attention: ``num_key_value_heads = 1``; ``wkv`` projects
        directly to that single KV head and the same tensor is read as both key and
        value.
      * Partial RoPE on the first ``rope_head_dim`` of each head ("Partial Rotary
        Positional Embedding"). RoPE is also applied with position ``-i`` to the
        attention output's rope slice, so the contribution of each KV entry stays a
        function of the *relative* distance to the query.
      * RMSNorm on the queries (``q_norm``) and the compressed KV head (``kv_norm``)
        right before the core attention, to keep logits bounded.
      * Per-head learnable attention sink (eq. 27).
      * Grouped low-rank output projection (§2.3.1, "Grouped Output Projection"):
        ``g`` head-groups → ``d_g``-dim intermediate outputs through a block-diagonal
        :class:`DeepseekV4GroupedLinear`, then mixed back to ``hidden_size`` by ``wo_b``.
      * A supplementary uncompressed sliding-window KV branch of size
        ``sliding_window`` ("Additional Branch of Sliding Window Attention") that
        preserves local fine-grained dependencies, concatenated with the
        long-range compressor's output before core attention.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        # V4 doesn't reuse V3's MLA projections (q_a/q_b/kv_a_proj_with_mqa/kv_b_proj/
        # o_proj) — every V4 block is shared-KV MQA with a single ``wkv`` and a grouped
        # output projection — so inheriting from ``DeepseekV3Attention`` only to delete
        # half of what its ``__init__`` builds is not worth it. We init from
        # ``nn.Module`` directly and set up V4-specific projections inline.
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        # Long-range branch dispatched by ``layer_type`` (see ``COMPRESSOR_CLASSES``
        # above). ``None`` means full-attention / sliding-only — no compressor is
        # built and the layer keeps just the local sliding-window K=V branch.
        compressor_cls = COMPRESSOR_CLASSES[self.layer_type]
        self.compressor = compressor_cls(config) if compressor_cls is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = hidden_states.shape[:2]
        cos, sin = position_embeddings

        # --- Q + KV projections + partial RoPE on the *trailing* qk_rope_head_dim of
        # each head (matches the V4-Flash reference's ``[..., -rd:]`` indexing — wkv
        # weights are laid out [nope|rope] in the checkpoint, so the trailing slice is
        # what gets rotated).
        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Per-head RMSNorm-style rescale (no learned weight) — the V4-Flash reference
        # (``inference/model.py:498``) does ``q *= rsqrt(mean(q**2) + eps)`` on each
        # head after wq_b, before RoPE. Skipping it leaves attention scores at the
        # wrong scale and the model collapses to a single repeated token within a
        # handful of layers.
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.config.rms_norm_eps).to(q.dtype)
        kv = self.kv_norm(self.wkv(hidden_states)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        q_nope, q_rope = q[..., : -self.qk_rope_head_dim], q[..., -self.qk_rope_head_dim :]
        kv_nope, kv_rope = kv[..., : -self.qk_rope_head_dim], kv[..., -self.qk_rope_head_dim :]
        q_rope, kv_rope = apply_rotary_pos_emb(q_rope, kv_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        # --- Sliding-window K=V branch goes through the standard cache update ---
        if past_key_values is not None:
            kv, _ = past_key_values.update(kv, kv, self.layer_idx)

        # Sliding-only layers skip the long-range branch (no compressor was built).
        # For HCA / CSA, ``DynamicCache(config=...)`` builds the right cache layer per
        # ``config.layer_types[i]`` via ``LAYER_TYPE_CACHE_MAPPING``, so the compressor
        # reads its layer state from ``past_key_values.layers[layer_idx]``.
        # ``past_key_values`` is ``None`` only when ``GradientCheckpointingLayer`` zeroes
        # it during a checkpoint replay — the compressor handles that as a single-shot
        # window pool with no persistent state.
        if self.compressor is None:
            full_kv = kv
        else:
            compressed_kv = self.compressor(hidden_states, q_residual, position_ids, past_key_values, self.layer_idx)
            full_kv = torch.cat([kv, compressed_kv], dim=2)

        if attention_mask is not None and full_kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(attention_mask, (0, full_kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            full_kv,
            full_kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # De-rotate the output's rope slice. V4 shares K and V (``wkv`` projects to a
        # single tensor), so V's rope slice carries the same per-token rotation as K.
        # Attention sums V-rotated values across attended positions, so the output's
        # rope slice is a position-mixed content; conjugate rotation at the query
        # position pulls it back into a position-independent frame before the output
        # projection mixes heads.
        out_nope, out_rope = attn_output[..., : -self.qk_rope_head_dim], attn_output[..., -self.qk_rope_head_dim :]
        out_rope = out_rope.transpose(1, 2)
        out_rope, _ = apply_rotary_pos_emb(out_rope, torch.zeros_like(out_rope), cos, -sin)
        attn_output = torch.cat([out_nope, out_rope.transpose(1, 2)], dim=-1)

        grouped = attn_output.reshape(batch, seq_len, -1).view(batch, seq_len, self.config.o_groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(2)), attn_weights


class DeepseekV4HyperConnection(nn.Module):
    r"""
    Manifold-Constrained Hyper-Connections
    (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent
    Transformer blocks

    Owns the learned (``fn``, ``base``, ``scale``)
    parameters that turn the incoming ``hc_mult`` residual streams into collapse / expand
    weights. The decoder layer instantiates two of these (one for the attention site,
    one for the mlp site).

    ASCII shape guide — ``B`` = batch, ``S`` = seq, ``H`` = hc_mult, ``D`` = hidden_size::

              hidden_streams        flatten(2)        RMSNorm-rescale + F.linear(fn)
         [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────────────►
                                                             mix-logits
                                                             [B, S, (2+H)*H]
                                                                    │
                            ┌───────────────────────────────────────┴──────────────────────────────┐
                            ▼                          ▼                                           ▼
                        pre logits                post logits                               comb logits
                        [B, S, H]                 [B, S, H]                                 [B, S, H, H]
                        × scale[0]                × scale[1]                                × scale[2]
                        + base[:H]                + base[H:2H]                              + base[2H:]
                        σ() + eps                 σ() + eps                                 σ() + eps
                        │                         │                                         │
                        pre                        post                                     Sinkhorn(iters)
                        (stream collapse weights)  (block-output placement)                 row/col normalise
                                                                                            │
                                                                                            comb
                                                                                            (stream mixer)
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        project it onto the manifold of doubly stochastic matrices M.
        This is achieved by the Sinkhorn-Knopp algorithm, which first applies an exponential function
        ˜
        to
        𝐵𝑙 to ensure positivity, getting 𝑀(0) = exp(˜
        𝐵𝑙), and then iteratively performs column and row
        normalization:
        𝑀(𝑡) = T𝑟(T𝑐(𝑀(𝑡−1))), (8)
        where T𝑟 and T𝑐 denote row and column normalization, respectively.
        """
        flat = hidden_streams.flatten(start_dim=2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mix = F.linear(flat, self.fn.float()) * rsqrt  # [B, S, (2+H)*H]
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc]) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2 * hc :].view(hc, hc)
            )
            + self.hc_eps
        )
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse; used by ``DeepseekV4Model`` before the shared RMSNorm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class DeepseekV4MLP(Qwen2MoeMLP):
    """Shared expert — plain SwiGLU MLP, ``moe_intermediate_size`` hidden."""

    def __init__(self, config: DeepseekV4Config, intermediate_size: int | None = None):
        super().__init__(config, intermediate_size or config.moe_intermediate_size)


@use_experts_implementation
class DeepseekV4Experts(GptOssExperts):
    """Routed experts: per-expert iteration + ``_apply_gate`` hook from GPT-OSS, but
    using the Mixtral weight layout (no biases, ``[num_experts, 2*intermediate, hidden]``
    for ``gate_up_proj`` and ``[num_experts, hidden, intermediate]`` for ``down_proj``).
    Activation is SiLU and gate/up are clamped to ``swiglu_limit`` before mixing.
    """

    def __init__(self, config: DeepseekV4Config):
        nn.Module.__init__(self)
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.limit = config.swiglu_limit
        self.act_fn = ACT2FN[config.hidden_act]

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            current = self._apply_gate(gate_up)
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class DeepseekV4TopKRouter(MixtralTopKRouter):
    """DeepSeekMoE top-k router (paper §2.1, "Mixture-of-Experts"). Two changes from
    the V3 router:

      * The expert affinity activation is ``Sqrt(Softplus(·))`` instead of the V3
        Sigmoid (paper §2.1: "we change the activation function that computes the
        affinity scores from Sigmoid(·) into Sqrt(Softplus(·))"). The ``scoring_func``
        config field selects this for V4 checkpoints.
      * The constraint on the number of routing target nodes used in V3 is dropped,
        and the V3 ``n_group`` / ``topk_group`` machinery is removed entirely (paper
        §2.1: "we remove the constraint on the number of routing target nodes").

    The auxiliary-loss-free strategy is preserved via the per-expert ``bias`` buffer
    that biases the top-k argmax without flowing gradients (same ``noaux_tc`` idea
    as DeepSeek-V3).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        # The correction bias biases the argmax only — never gradient-carrying — so it's
        # a buffer (same convention as DeepseekV3's ``e_score_correction_bias``).
        self.register_buffer("bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(MixtralTopKRouter):
    """Hash routing for the first ``num_hash_layers`` MoE layers (paper §2.1, "Mixture-
    of-Experts"). The first three blocks of V4 replace the dense FFN of V3 with an MoE
    where the expert selection is determined by a fixed hash of the input token id —
    a frozen ``tid2eid`` (token id to expert id) lookup — instead of a learned gate.
    The learned gate ``weight`` still produces the per-expert scoring values used to
    weight the selected experts' activations; only the *which-experts* selection is
    static.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = layer_idx < config.num_hash_layers
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None, **_) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            if input_ids is None:
                raise ValueError(
                    "DeepseekV4's hash-routing layers need `input_ids` to look up expert indices. "
                    "The `inputs_embeds`-only inference path is not supported for models with "
                    "`num_hash_layers > 0`."
                )
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    r"""DeepSeek-V4 decoder block (paper §2). Differs from a classic residual block in
    two places:

      * The residual is a stack of ``hc_mult`` parallel streams kept in shape
        ``[B, S, hc_mult, D]`` throughout the block, mixed in and out via two
        :class:`DeepseekV4HyperConnection` modules (Manifold-Constrained Hyper-
        Connections / mHC, paper §2.2; Xie et al., 2026). The mHC mappings constrain
        the residual transform to the manifold of doubly-stochastic matrices via the
        Sinkhorn-Knopp projection — making signal propagation non-expansive across
        deep stacks.
      * ``self_attn`` is :class:`DeepseekV4Attention` for every layer. Its compressor
        sub-module is the only thing that varies by layer type
        (:class:`DeepseekV4HCACompressor` for HCA layers,
        :class:`DeepseekV4CSACompressor` for CSA, picked via
        ``config.layer_types[layer_idx]``); the CSA compressor also owns the
        Lightning Indexer at ``self_attn.compressor.indexer``.

    Classic residual decoder layer::

        h ──► norm ──► self_attn ──► + ──► norm ──► mlp ──► +
        └──────── residual ────────┘   └─────── residual ───┘

    Deepseek V4 decoder layer (``H = hc_mult`` parallel residual streams throughout)::

                attention site                                    mlp site
        ┌────────────────────────────────────────┐    ┌────────────────────────────────────────┐
        │  hidden_streams [B, S, H, D]           │    │  hidden_streams [B, S, H, D]           │
        │        │                               │    │        │                               │
        │  attn_hc(streams) ─► (pre, post, comb) │    │  ffn_hc(streams) ─► (pre, post, comb)  │
        │        │                               │    │        │                               │
        │   Σ pre·streams  (collapse)            │    │   Σ pre·streams  (collapse)            │
        │        │                               │    │        │                               │
        │   input_layernorm                      │    │   post_attention_layernorm             │
        │        │                               │    │        │                               │
        │   self_attn                            │    │   mlp  (MoE routed + shared)           │
        │        │                               │    │        │                               │
        │   post·output + comb·streams  (expand) │    │   post·output + comb·streams  (expand) │
        │        │                               │    │        │                               │
        │        ▼                               │    │        ▼                               │
        │  new hidden_streams  ──────────────────┘    │  new hidden_streams                    │
        └────────────────────────────────────────┘    └────────────────────────────────────────┘


    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        # hidden_states throughout: [B, S, hc_mult, hidden].

        # --- Attention site: collapse → norm → attn → expand ---
        pre, post, comb = self.attn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        dtype = hidden_states.dtype
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )

        # --- MLP site: collapse → norm → mlp → expand ---
        pre, post, comb = self.ffn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=kwargs.get("input_ids"))
        dtype = hidden_states.dtype
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.to(dtype), hidden_states)


# -----------------------------------------------------------------------------
# Pre-trained base + Model + ForCausalLM.
# -----------------------------------------------------------------------------


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    # V4 ships eager-only: the compressor / indexer paths weren't validated against
    # SDPA / FlashAttention / FlexAttention kernels — leaving these ``False`` makes
    # ``set_attn_implementation`` reject those backends instead of silently routing
    # through them.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    # The compressor's rolling-window buffer / pool / overlap state lives on the
    # per-layer cache (:class:`DeepseekV4HCACache` / :class:`DeepseekV4CSACache`)
    # and isn't compatible with :class:`StaticCache` — that path would hand the
    # compressor a :class:`StaticSlidingWindowLayer` with no ``update_compressor``
    # method. Disabling fullgraph compile keeps generation tests on the dynamic
    # cache build that does dispatch to V4's own cache layers.
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = ["attn_hc", "ffn_hc"]
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    # ``_is_stateful`` opts out of generation modes that need to roll the cache
    # back across drafts (assisted generation, prompt lookup, contrastive search).
    # The compressor's running-window state isn't rewindable, so ``generate``
    # raises a clear error early instead of failing deep in the compressor with
    # a missing-method ``AttributeError``.
    _is_stateful = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
        "hidden_states": DeepseekV4DecoderLayer,
        "attentions": DeepseekV4Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, DeepseekV4TopKRouter):
                init.zeros_(module.bias)  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                init.zeros_(module.tid2eid)  # buffer; real values come from the checkpoint
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = DeepseekV4HyperHead(config)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.rotary_emb_compress = DeepseekV4RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # V4's compressor reads / writes per-layer buffer state on the cache, so we
        # always build a ``DynamicCache(config=...)`` internally — even when
        # ``use_cache=False`` we need a forward-scoped cache to thread the compressor's
        # buffer through the window pooling. ``LAYER_TYPE_CACHE_MAPPING`` populates the
        # right :class:`DeepseekV4HCACache` / :class:`DeepseekV4CSACache` per layer.
        # When ``use_cache=False`` we still hand the layers a real cache; we just don't
        # surface it back to the caller so the user-facing semantics match other models.
        return_cache = past_key_values if use_cache else None
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length()
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
        # ``generate()`` may pass a per-layer-type mask dict already built by
        # ``create_masks_for_generate``; all V4 layer types use the same sliding-window
        # mask, so use the prebuilt one directly. Otherwise build it here.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        cos_sin = self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main")

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=cos_sin,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=return_cache)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
]
