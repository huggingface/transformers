# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    scoring_func (`str`):
        Router activation — `sqrtsoftplus`, `softmax`, or `sigmoid`.
    rope_theta (`float`):
        RoPE base for the main self-attention rotary.
    layer_types (`list[str]`):
        Per-layer attention schedule with values from
        `{"compressed_sparse_attention", "heavily_compressed_attention"}`.
        V4-Pro default: 2× HCA bootstrap + interleaved CSA / HCA.
    compress_rates (`dict[str, int]`):
        Per-layer-type compression rate. Default
        `{"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}`
        (m=4 for CSA, m'=128 for HCA, paper §2.3.1 / §2.3.2). BC: configs that ship
        `compress_rate_csa` / `compress_rate_hca` as top-level kwargs are folded
        in at `__post_init__` time.
    compress_rope_theta (`float`):
        RoPE base for the compressed branches (paired with
        `rope_scaling` for YaRN).
    hc_mult (`int`):
        Manifold-Constrained Hyper-Connection (mHC) expansion factor n_hc
        (always active; Section 2.2).
    hc_sinkhorn_iters (`int`):
        Sinkhorn-Knopp iterations t_max for the mHC residual
        mapping projection onto doubly-stochastic matrices.
    hc_eps (`float`):
        Numerical floor for the Sinkhorn-Knopp normalization.
    mlp_layer_types (`list[str]`):
        Per-layer MoE schedule with values from
        `{"hash_moe", "moe"}`. `hash_moe` routes via a frozen
        `tid2eid[input_ids]` lookup (paper §2.1, "Hash-MoE bootstrap"); `moe`
        is the standard top-k routed MoE. Default: 3× `hash_moe` then `moe`
        for the rest. BC: legacy configs that ship `num_hash_layers` as a
        top-level kwarg are folded in at `__post_init__` time.
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
        Defaults to `qk_rope_head_dim / head_dim` so cos/sin sizes to `qk_rope_head_dim`.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    # `num_local_experts` is the standard MoE attr name (read by FP8 / TP integrations);
    # `intermediate_size` is what :class:`LlamaMLP` reads for the shared expert width
    # — V4 only ships `moe_intermediate_size` so we route the read through.
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        # EP-only by default, same shape as gpt-oss: route on the gate, run the
        # routed experts as a grouped-GEMM kernel sharded along the expert axis,
        # and wrap the experts module with `moe_tp_experts` so its output gets
        # all-reduced across ranks. Attention stays replicated (V4 is shared-KV
        # MQA + a CSA / HCA compressor branch — both broadcast a single KV head
        # across all attention heads via `repeat_kv`, so colwise-sharding
        # `q_b_proj` would leave KV replicated and `repeat_kv` would no longer
        # match the rank-local query head count). The shared MLP also stays
        # replicated — it's small and not worth TP-ing. There's deliberately
        # no `base_model_tp_plan` for V4: we don't ship a pure-TP plan, only EP.
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    default_partial_rotary_factor = 64 / 512  # `qk_rope_head_dim` (64) / `head_dim` (512)
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

    # V4's `rope_parameters` is keyed by *rope-type* labels (`main` / `compress`) — not
    # by `layer_types`. The base `validate_rope` checks `keys ⊆ layer_types` and falls
    # back to wrapping the whole dict as a single set of params when the subset check
    # fails, which then warns about `main` / `compress` as unrecognized keys. Override
    # to iterate the rope-type-keyed sub-dicts directly.
    _rope_type_labels = ("main", "compress")

    def validate_rope(self):
        rope_parameters_dict = getattr(self, "rope_parameters", None) or {}
        ignore_keys = self.ignore_keys_at_rope_validation
        # The yarn / longrope / llama3 validators in
        # :class:`RotaryEmbeddingConfigMixin` read `self.rope_parameters[<key>]`
        # directly (e.g. `original_max_position_embeddings`). With V4's
        # rope-type-keyed nesting, the top-level dict only has `main` / `compress`,
        # so those reads fail. Temporarily point `self.rope_parameters` at the
        # rope-type-specific sub-dict for the duration of the validation call,
        # then restore it.
        for rope_type_label in self._rope_type_labels:
            rope_parameters = rope_parameters_dict.get(rope_type_label)
            if not isinstance(rope_parameters, dict):
                continue
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            rope_parameters["rope_type"] = rope_type
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            if validation_fn is None:
                continue
            self.rope_parameters = rope_parameters
            try:
                validation_fn(rope_parameters, ignore_keys=ignore_keys)
            finally:
                self.rope_parameters = rope_parameters_dict

    def validate_layer_type(self):
        """V4 narrows the global `ALLOWED_LAYER_TYPES` to the three attention-block
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

    def __post_init__(self, **kwargs):
        # Strip legacy V4 kwargs (V3-flavoured names that older checkpoints still ship)
        # before the parent's strict-validated init sees them, then fold each into the
        # modern field below.
        legacy_compress_ratios = kwargs.pop("compress_ratios", None)
        legacy_compress_rate_csa = kwargs.pop("compress_rate_csa", None)
        legacy_compress_rate_hca = kwargs.pop("compress_rate_hca", None)
        legacy_num_hash_layers = kwargs.pop("num_hash_layers", None)
        legacy_qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)
        PreTrainedConfig.__post_init__(self, **kwargs)
        n = self.num_hidden_layers

        # `compress_rates`: dict, default per attention type. Legacy scalar overrides fold in.
        if self.compress_rates is None:
            self.compress_rates = dict(self.default_compress_rates)
        if legacy_compress_rate_csa is not None:
            self.compress_rates["compressed_sparse_attention"] = legacy_compress_rate_csa
        if legacy_compress_rate_hca is not None:
            self.compress_rates["heavily_compressed_attention"] = legacy_compress_rate_hca

        # `layer_types`: explicit > legacy `compress_ratios` per-layer ints (0/4/128) >
        # V4-Pro default (2× HCA bootstrap + CSA/HCA interleave).
        if self.layer_types is None and legacy_compress_ratios is not None:
            self.layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in legacy_compress_ratios]
        if self.layer_types is None:
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(n - 2, 0))
            ]
            self.layer_types = ["heavily_compressed_attention"] * min(n, 2) + interleave
        self.layer_types = list(self.layer_types[:n])

        # `mlp_layer_types`: first `num_hash_layers` hash_moe, rest moe.
        if self.mlp_layer_types is None:
            n_hash = legacy_num_hash_layers if legacy_num_hash_layers is not None else self.default_num_hash_layers
            self.mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
        self.mlp_layer_types = list(self.mlp_layer_types[:n])

        # `partial_rotary_factor` = legacy `qk_rope_head_dim / head_dim` if given, else default.
        # `qk_rope_head_dim` is a runtime-only attr (never a dataclass field).
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = (
                legacy_qk_rope_head_dim / self.head_dim
                if legacy_qk_rope_head_dim is not None
                else self.default_partial_rotary_factor
            )
        self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # yarn is applied ONLY to layers with a
        # compressor (CSA/HCA); pure sliding-window layers use plain RoPE with
        # `theta=rope_theta` (10000) and no scaling. Compress layers use
        # `theta=compress_rope_theta` (160000) with yarn factor=16, and the reference
        # does NOT multiply cos/sin by the yarn mscale — force `attention_factor=1.0`
        # so transformers' `_compute_yarn_parameters` doesn't apply `0.1·log(16)+1`.
        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            # Already nested — drop any leftover top-level keys.
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
        else:
            yarn = {k: v for k, v in rp.items() if k not in ("main", "compress")}
            main = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress = {
                **yarn,
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
                "attention_factor": 1.0,
            }
            compress.setdefault("rope_type", "default")
            self.rope_parameters = {"main": main, "compress": compress}


__all__ = ["DeepseekV4Config"]
