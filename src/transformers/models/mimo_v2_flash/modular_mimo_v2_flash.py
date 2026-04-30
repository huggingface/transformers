# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
    DeepseekV3NaiveMoe,
    DeepseekV3PreTrainedModel,
    DeepseekV3TopkRouter,
)
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import Glm4MoeMLP, apply_rotary_pos_emb, repeat_kv
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer
from ..laguna.modeling_laguna import LagunaModel
from ..mixtral.modeling_mixtral import MixtralRMSNorm
from ..qwen2.modeling_qwen2 import Qwen2Attention


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2-Flash")
@strict
class MiMoV2FlashConfig(Glm4MoeConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for group-based top-k routing.
    topk_group (`int`, *optional*, defaults to 1):
        Number of groups selected per token in group-based top-k routing.
    head_dim (`int`, *optional*, defaults to 192):
        Dimension of query and key heads.
    v_head_dim (`int`, *optional*, defaults to 128):
        Dimension of value heads (special case because MiMo uses a smaller v head dim than (qk) head dim )
    mlp_layer_types (`list`, *optional*):
        MLP pattern for each layer (`"dense"` or `"sparse"`). Defaults to 1 dense + rest sparse.
    attention_value_scale (`float`, *optional*, defaults to 0.707 (which is the decimal approximation
        of `sqrt(hidden_size / (num_attention_heads * v_head_dim))`):
        Constant multiplier applied to rescale the attention values.
    """

    model_type = "mimo_v2_flash"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "colwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    # Overrides from Glm4MoeConfig
    vocab_size: int = 152576
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    n_routed_experts: int = 256
    bos_token_id: int | None = 1
    routed_scaling_factor: float | None = 1.0
    # MiMo-V2-Flash specific
    head_dim: int = 192
    v_head_dim: int = 128
    sliding_window: int = 128
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    attention_value_scale: float | None = 0.707
    # Remove unused attributes inherited from Glm4MoeConfig
    first_k_dense_replace = AttributeError()
    n_shared_experts = AttributeError()
    use_qk_norm = AttributeError()

    def __post_init__(self, **kwargs):
        # Full attention for the first layer and every 6th layer; SWA for the rest.
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i == 0 or not ((i + 1) % 6)) else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        # First layer is a dense MLP, the rest are MoE.
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        # Per-layer rope defaults matching the XiaomiMiMo/MiMo-V2-Flash pretrained thetas.
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 5_000_000.0, "partial_rotary_factor": 0.334},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.334},
            }
        # BC: The hub config.json stores `routed_scaling_factor` as null
        if self.routed_scaling_factor is None:
            self.routed_scaling_factor = 1.0

        PreTrainedConfig.__post_init__(self, **kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        return kwargs


class MiMoV2FlashRMSNorm(MixtralRMSNorm):
    pass


class MiMoV2FlashRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: MiMoV2FlashConfig, device=None):
        super().__init__(config, device=device)

    @staticmethod
    def compute_default_rope_parameters(
        config: MiMoV2FlashConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters[layer_type]["rope_theta"]
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 0.334)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class MiMoV2FlashTopkRouter(DeepseekV3TopkRouter):
    pass


class MiMoV2FlashNaiveMoe(DeepseekV3NaiveMoe):
    pass


class MiMoV2FlashMoE(DeepseekV3MoE):
    """
    Only difference from `DeepseekV3MoE` is that we have no shared experts.
    So we drop it and override the forward to skip the shared expert (residual like) add.
    """

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        del self.shared_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        return self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)


class MiMoV2FlashMLP(Glm4MoeMLP):
    pass


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Key difference from `eager_attention_forward`: optional attention sinks.
    if module.sinks is not None:
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        attn_weights = torch.cat([attn_weights, sinks], dim=-1)

    # Subtract max for BF16/FP16 numerical stability
    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)

    if module.sinks is not None:
        probs = probs[..., :-1]  # drop the sink

    attn_weights = nn.functional.dropout(probs, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MiMoV2FlashAttention(Qwen2Attention):
    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        # SWA layers double the kv heads vs full-attention and have attention sinks.
        is_swa = config.layer_types[layer_idx] == "sliding_attention"
        num_key_value_heads = config.num_key_value_heads * 2 if is_swa else config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        super().__init__(config, layer_idx)
        self.v_head_dim = config.v_head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(
            config.hidden_size, num_key_value_heads * config.v_head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(num_attention_heads * config.v_head_dim, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(num_attention_heads)) if is_swa else None
        self.v_scale = config.attention_value_scale if config.attention_value_scale is not None else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        # Different head dims compared to other attentions
        qk_hidden_shape = (*input_shape, -1, self.head_dim)
        v_hidden_shape = (*input_shape, -1, self.v_head_dim)

        query_states = self.q_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        # Additional scaling on values.
        value_states = self.v_proj(hidden_states).view(v_hidden_shape).transpose(1, 2) * self.v_scale

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # Optional sinks, only when in SWA layer
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiMoV2FlashDecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


@auto_docstring
class MiMoV2FlashPreTrainedModel(DeepseekV3PreTrainedModel):
    _supports_sdpa = False  # disabling SDPA as it has no sink API atm (same as gpt-oss)
    _supports_flash_attn = True  # not compatible because of asymmetric qk/v head dims and/or sinks (for most FAs)
    _supports_flex_attn = False  # asymmetric head dim + not being a power of 2
    _compatible_flash_implementations = ["flash_attention_4"]
    _keys_to_ignore_on_load_unexpected = [r"^model\.mtp\."]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, MiMoV2FlashAttention) and module.sinks is not None:
            init.normal_(module.sinks, mean=0.0, std=std)
        elif isinstance(module, MiMoV2FlashRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class MiMoV2FlashModel(LagunaModel):
    pass


class MiMoV2FlashForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "MiMoV2FlashConfig",
    "MiMoV2FlashForCausalLM",
    "MiMoV2FlashModel",
    "MiMoV2FlashPreTrainedModel",
]
