# Copyright 2026 the HuggingFace Team. All rights reserved.
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


@auto_docstring(checkpoint="zai-org/GLM-5-Next")
@strict
class Glm5NextConfig(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of routed expert groups.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Clamp limit applied to SwiGLU gate/up projections.
    linear_attn_config (`dict`, *optional*):
        KDA linear attention layout and dimensions. Layers listed in
        `kda_layers` use KDA; layers listed in `full_attn_layers` use MLA.
    mhc (`bool`, *optional*, defaults to `False`):
        Enables MHC residual streams. Older checkpoints without this field use
        the standard single-stream residual path.
    hc_mult (`int`, *optional*, defaults to 4):
        Number of MHC residual streams.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical floor used by MHC Sinkhorn normalization.
    hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
        Number of Sinkhorn iterations used by MHC routing.
    index_head_dim (`int`, *optional*, defaults to 128):
        DSA indexer projection head dimension.
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of DSA indexer heads.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of sparse-attention positions selected by the DSA indexer.
    index_kpool (`int`, *optional*, defaults to 1):
        DSA serving-cache key pooling factor. Values greater than 1 enable
        checkpoint-compatible index-pool compression parameters.
    index_kpool_compress (`bool`, *optional*, defaults to `False`):
        Whether DSA index-pool compression parameters are present.
    index_kpool_always_select_tail (`bool`, *optional*, defaults to `False`):
        Whether the incomplete KPool tail is always included in sparse attention.
    indexer_rope_interleave (`bool`, *optional*, defaults to `False`):
        Whether DSA indexer RoPE uses interleaved pairs instead of NeoX half rotation.
    index_dsa_use_layernorm (`bool`, *optional*):
        Whether DSA indexer keys include `indexer.k_norm.*`. If this field is
        absent, GLM5-Next keeps the legacy no-indexer path.
    index_skip_topk_offset (`int`, *optional*, defaults to 1):
        Offset used when deriving the default DSA indexer shared/full pattern.
    index_topk_freq (`int`, *optional*, defaults to 1):
        Frequency used when deriving the default DSA indexer shared/full pattern.
    index_topk_pattern (`str` or `list[str]`, *optional*):
        Explicit DSA indexer shared/full pattern.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer mode. Values are `"full"` or `"shared"`.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer feed-forward schedule. Values are `"dense"` or `"sparse"`.
    layer_types (`list[str]`, *optional*):
        Per-layer attention cache schedule. Values are `"linear_attention"` for
        KDA layers and `"full_attention"` for MLA layers.
    """

    model_type = "glm5_next"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_allreduce",
        "layers.*.mlp.experts": "moe_experts_allreduce",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise_allreduce",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    max_position_embeddings: int = 1104096
    initializer_range: float = 0.02
    hidden_act = "silu"
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    q_lora_rank: int | None = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    swiglu_limit: float | None = None
    linear_attn_config: dict | None = None
    mhc: bool = False
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int | None = 2048
    index_kpool: int = 1
    index_kpool_compress: bool = False
    index_kpool_always_select_tail: bool = False
    indexer_rope_interleave: bool = False
    index_dsa_use_layernorm: bool | None = None
    index_skip_topk_offset: int | None = 1
    index_topk_freq: int | None = 1
    index_topk_pattern: str | list[str] | None = None
    indexer_types: list[str] | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = 154820
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            }

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )

        if self.linear_attn_config is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            full_attn_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 == 3]
            self.linear_attn_config = {
                "full_attn_layers": full_attn_layers,
                "head_dim": 128,
                "kda_layers": kda_layers,
                "num_heads": 64,
                "short_conv_kernel_size": 4,
                "lower_bound": None,
                "safe_gate": False,
            }

        if self.layer_types is None:
            kda_layers = set(self.linear_attn_config.get("kda_layers", []))
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]

        if self.indexer_types is None:
            pattern = self.index_topk_pattern
            freq = self.index_topk_freq
            offset = self.index_skip_topk_offset
            if isinstance(pattern, str):
                self.indexer_types = [{"F": "full", "S": "shared"}[char] for char in pattern]
            elif pattern is not None:
                self.indexer_types = list(pattern)
            else:
                self.indexer_types = [
                    "full" if (max(layer_idx - offset, 0) % freq) == 0 else "shared"
                    for layer_idx in range(self.num_hidden_layers)
                ]

        super().__post_init__(**kwargs)

        # TODO: Proper alias and checking/validating
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )

        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}.")
        if self.index_kpool > 1 and self.index_kpool_compress:
            if self.index_topk is None or self.index_topk <= 0:
                raise ValueError("Active KPool requires index_topk to be a positive integer.")
            if self.index_topk % self.index_kpool != 0:
                raise ValueError(
                    f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool})."
                )

        if self.q_lora_rank is None:
            raise ValueError("For DSA usage in hte attention layers, the `q_lora_rank` is strictly required!")


__all__ = ["Glm5NextConfig"]
