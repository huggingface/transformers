# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


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


DEEPSEEK_V4_MLP_LAYER_TYPES = ("hash_moe", "moe")


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4-Flash-Base")
@strict
class DeepseekV4Config(PreTrainedConfig):
    r"""
    DeepSeek-V4's hybrid attention follows the paper (Section 2.3): every block is one
    of three attention types — *Full Attention* (sliding-window only), *Compressed
    Sparse Attention* (CSA, Section 2.3.1) and *Heavily Compressed Attention* (HCA,
    Section 2.3.2). CSA compresses the KV cache by ``compress_rate_csa`` (m=4 in V4-
    Flash/Pro) and selects ``index_topk`` blocks per query via the Lightning Indexer;
    HCA applies a much heavier compression of ``compress_rate_hca`` (m'=128) and
    skips sparse selection. Both branches add a small uncompressed sliding-window
    branch for fine-grained locality.

    layer_types (`list[str]`): Per-layer attention schedule with values from
        ``{"compressed_sparse_attention", "heavily_compressed_attention"}``.
        V4-Pro default: 2× HCA bootstrap + interleaved CSA / HCA.
    compress_rates (`dict[str, int]`): Per-layer-type compression rate. Default
        ``{"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}``
        (m=4 for CSA, m'=128 for HCA, paper §2.3.1 / §2.3.2). BC: configs that ship
        ``compress_rate_csa`` / ``compress_rate_hca`` as top-level kwargs are folded
        in at ``__post_init__`` time.
    rope_theta (`float`): RoPE base for the main self-attention rotary.
    compress_rope_theta (`float`): RoPE base for the compressed branches (paired with
        ``rope_scaling`` for YaRN).
    partial_rotary_factor (`float`, *optional*): Fraction of head_dim that gets RoPE.
        Defaults to ``qk_rope_head_dim / head_dim`` so cos/sin sizes to ``qk_rope_head_dim``.
    hc_mult (`int`): Manifold-Constrained Hyper-Connection (mHC) expansion factor n_hc
        (always active; Section 2.2).
    hc_sinkhorn_iters (`int`): Sinkhorn-Knopp iterations t_max for the mHC residual
        mapping projection onto doubly-stochastic matrices.
    hc_eps (`float`): Numerical floor for the Sinkhorn-Knopp normalization.
    mlp_layer_types (`list[str]`): Per-layer MoE schedule with values from
        ``{"hash_moe", "moe"}``. ``hash_moe`` routes via a frozen
        ``tid2eid[input_ids]`` lookup (paper §2.1, "Hash-MoE bootstrap"); ``moe``
        is the standard top-k routed MoE. Default: 3× ``hash_moe`` then ``moe``
        for the rest. BC: legacy configs that ship ``num_hash_layers`` as a
        top-level kwarg are folded in at ``__post_init__`` time.
    scoring_func (`str`): Router activation — ``sqrtsoftplus``, ``softmax``, or ``sigmoid``.
    swiglu_limit (`float`): Clip routed experts' gate/up pre-activations.
    sliding_window (`int`): Local window size n_win used in every attention block's
        sliding-window branch.
    o_groups (`int`): Number of head-groups g in the grouped output projection
        (paper §2.3.1, "Grouped Output Projection").
    o_lora_rank (`int`): Per-group intermediate dim d_g in the grouped output projection.
    index_n_heads (`int`): Number of indexer query heads n_h^I (paper §2.3.1, eq. 14).
    index_head_dim (`int`): Indexer head dim c^I (paper §2.3.1).
    index_topk (`int`): Number of compressed entries per query the Lightning Indexer
        keeps via top-k (paper §2.3.1, eq. 17).
    num_nextn_predict_layers (`int`): MTP layer count in the upstream checkpoint
        (not instantiated here).
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    # ``num_local_experts`` is the standard MoE attr name (read by FP8 / TP integrations);
    # ``intermediate_size`` is what :class:`LlamaMLP` reads for the shared expert width
    # — V4 only ships ``moe_intermediate_size`` so we route the read through.
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        # q_a_proj / kv_proj outputs feed RMSNorms (q_norm / kv_norm) that normalise
        # across the full output dim — sharding the output would break the norm. Only
        # q_b_proj is colwise-sharded (per-head split is safe: q_head_norm is per-head),
        # and o_b_proj is rowwise (input-dim sharded). o_a_proj is a GroupedLinear
        # whose forward uses ``torch.bmm``; the standard TP wrappers don't handle bmm.
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.o_b_proj": "rowwise",
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
    q_lora_rank: int = 1024
    default_partial_rotary_factor = 64 / 512  # ``qk_rope_head_dim`` (64) / ``head_dim`` (512)
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 1048576
    rope_theta: float | int = 10000.0

    layer_types: list[str] | None = None
    compress_rates: dict | None = None
    default_compress_rates = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
    compress_rope_theta: float | int = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    mlp_layer_types: list[str] | None = None
    default_num_hash_layers = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1

    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    partial_rotary_factor: float | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0

    def validate_layer_type(self):
        """V4 narrows the global ``ALLOWED_LAYER_TYPES`` to the three attention-block
        types and two MLP-block types it actually ships with, on top of the standard
        length / type-membership checks.
        """
        if self.num_hidden_layers is None:
            return
        for name, types, allowed in (
            ("layer_types", self.layer_types, DEEPSEEK_V4_LAYER_TYPES),
            ("mlp_layer_types", self.mlp_layer_types, DEEPSEEK_V4_MLP_LAYER_TYPES),
        ):
            if types is None:
                continue
            if len(types) != self.num_hidden_layers:
                raise ValueError(
                    f"`num_hidden_layers` ({self.num_hidden_layers}) must equal `len({name})` ({len(types)})."
                )
            bad = [t for t in types if t not in allowed]
            if bad:
                raise ValueError(f"`{name}` entries must be one of {allowed} for DeepSeek-V4; got {bad}.")

    def _apply_legacy_kwargs(self, kwargs: dict) -> dict:
        """Strip and stash legacy V4 kwargs that older checkpoints / configs ship under
        their original V3-flavoured names. The values are kept on the instance so
        ``__post_init__`` can fold them into the new fields after the parent's init has
        run, and the kwargs dict is returned cleaned for ``PreTrainedConfig.__post_init__``.
        """
        self._legacy_compress_ratios = kwargs.pop("compress_ratios", None)
        self._legacy_compress_rate_csa = kwargs.pop("compress_rate_csa", None)
        self._legacy_compress_rate_hca = kwargs.pop("compress_rate_hca", None)
        self._legacy_num_hash_layers = kwargs.pop("num_hash_layers", None)
        # ``qk_rope_head_dim`` isn't a config field — it's derived from
        # ``partial_rotary_factor * head_dim`` and only set as a runtime attribute.
        self._legacy_qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)
        return kwargs

    def _resolve_compress_rates(self) -> None:
        if self.compress_rates is None:
            self.compress_rates = dict(self.default_compress_rates)
        if self._legacy_compress_rate_csa is not None:
            self.compress_rates["compressed_sparse_attention"] = self._legacy_compress_rate_csa
        if self._legacy_compress_rate_hca is not None:
            self.compress_rates["heavily_compressed_attention"] = self._legacy_compress_rate_hca

    def _resolve_layer_types(self) -> None:
        n = self.num_hidden_layers
        if self.layer_types is None and self._legacy_compress_ratios is not None:
            # Translate the V4 checkpoint's per-layer integer ``compress_ratios`` into the
            # named ``layer_types`` schedule (0 = sliding-only, 4 = CSA, 128 = HCA).
            self.layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in self._legacy_compress_ratios]
        if self.layer_types is None:
            # V4-Pro default: two HCA bootstrap layers, then CSA / HCA interleaved.
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(n - 2, 0))
            ]
            head = ["heavily_compressed_attention"] * min(n, 2)
            self.layer_types = head + interleave
        self.layer_types = list(self.layer_types[:n])

    def _resolve_mlp_layer_types(self) -> None:
        n = self.num_hidden_layers
        if self.mlp_layer_types is None:
            n_hash = (
                self._legacy_num_hash_layers
                if self._legacy_num_hash_layers is not None
                else self.default_num_hash_layers
            )
            self.mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
        self.mlp_layer_types = list(self.mlp_layer_types[:n])

    def _resolve_partial_rotary_factor(self) -> None:
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = (
                self._legacy_qk_rope_head_dim / self.head_dim
                if self._legacy_qk_rope_head_dim is not None
                else self.default_partial_rotary_factor
            )
        # Runtime-only attribute; never declared as a dataclass field.
        self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

    def _resolve_rope_parameters(self) -> None:
        """Normalize ``rope_parameters`` into a per-rope-type dict
        ``{"main": {...}, "compress": {...}}`` (Gemma3 pattern; keys are *rope-type*
        labels, unrelated to ``layer_types``). Idempotent across save/load.

        By the time we get here :class:`PreTrainedConfig` has already run
        :meth:`RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`, which folds the
        checkpoint's legacy top-level ``rope_scaling`` block into ``self.rope_parameters``
        as a flat dict (``rope_type``, ``factor``, YaRN params, …). We just split that
        flat dict into the two rope-type buckets — the only difference between the two
        is the ``rope_theta`` base (main attention uses ``rope_theta=10000``; the
        compressor / indexer uses ``compress_rope_theta=160000``).
        """
        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
            return
        base = {k: v for k, v in rp.items() if k not in ("main", "compress")}
        base.setdefault("rope_theta", self.rope_theta)
        base["partial_rotary_factor"] = self.partial_rotary_factor
        base.setdefault("rope_type", "default")
        main = dict(base)
        compress = {**base, "rope_theta": self.compress_rope_theta}
        self.rope_parameters = {"main": main, "compress": compress}

    def __post_init__(self, **kwargs):
        kwargs = self._apply_legacy_kwargs(kwargs)
        PreTrainedConfig.__post_init__(self, **kwargs)
        self._resolve_compress_rates()
        self._resolve_layer_types()
        self._resolve_mlp_layer_types()
        self._resolve_partial_rotary_factor()
        self._resolve_rope_parameters()


__all__ = ["DeepseekV4Config"]
