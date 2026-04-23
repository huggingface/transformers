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
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_experts_implementation, use_kernelized_func
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import apply_rotary_pos_emb  # noqa: F401
from ..llama.modeling_llama import LlamaDecoderLayer, repeat_kv
from ..mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from ..qwen2.modeling_qwen2 import Qwen2Attention
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2-Flash")
@strict
class MiMoV2FlashConfig(Glm4MoeConfig):
    r"""
    head_dim (`int`, *optional*, defaults to 192):
        Dimension of query and key heads.
    v_head_dim (`int`, *optional*, defaults to 128):
        Dimension of value heads (special case because MiMo uses a smaller v head dim than (qk) head dim )
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for group-based top-k routing.
    topk_group (`int`, *optional*, defaults to 1):
        Number of groups selected per token in group-based top-k routing.
    mlp_layer_types (`list`, *optional*):
        MLP pattern for each layer (`"dense"` or `"sparse"`). Defaults to 1 dense + rest sparse.
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
    router_jitter_noise: float = 0.0
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
        # The hub config.json stores `routed_scaling_factor` as null
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


# NOTE @casinca: Concerning this TopKRouter:
# This is the "fixed" TopKRouter from the remote DSV3 implementation with correct masked_fill=-inf and not 0.0 like the
# old native transformers DSV3 implementation currently in the repo. see the update:
# https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/commit/e9b33add76883f293d6bf61f6bd89b497e80e335#d2h-632685
#
# OR HF fixes the DSV3 masking and then I can try to inherit from DSV3, so that I don't override the forward.
#
# On top of that, I made this class as it is, so that it's a direct drop-in replacement for the MixtralSparseMoeBlock:
# I can just do inheritance from MixtralSparseMoeBlock and override the self.gate = MiMoV2FlashTopKRouter(config)
#
# tldr: this MiMo class is refactored (with DSV3 fix + internals) to be compatible with the MixtralSparseMoeBlock,
# following newer style like minimax M2 in the repo, for fused experts etc... I think this is what Vasqu prefers
class MiMoV2FlashTopKRouter(nn.Module):
    """MiMo gating with sigmoid scoring and group-based top-k selection."""

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )

    def forward(self, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]

        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return logits, topk_weight, topk_idx


@use_experts_implementation
class MiMoV2FlashExperts(MixtralExperts):
    """Fused experts (V5). Checkpoint layout: `mlp.experts.gate_up_proj`, `mlp.experts.down_proj`.
    Original reference used per-expert `nn.ModuleList` MLPs (`experts.{i}.gate_proj` etc.)
    """

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        # MoE experts use moe_intermediate_size, not the dense MLP's intermediate_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))


class MiMoV2FlashSparseMoeBlock(MixtralSparseMoeBlock):
    pass


class MiMoV2FlashMLP(Qwen2MoeMLP):
    pass


# Eager attention forward function with optional attention sinks.
# Same as the remote MiMo `eager_attention_forward` but with mask preparation removed (not needed post transformers V5)
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


@use_kernelized_func(apply_rotary_pos_emb)
class MiMoV2FlashAttention(Qwen2Attention):
    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        # SWA layers double the kv heads vs full-attention and have attention sinks.
        is_swa = config.layer_types[layer_idx] == "sliding_attention"
        num_kv_heads = config.num_key_value_heads * 2 if is_swa else config.num_key_value_heads
        num_attn_heads = config.num_attention_heads
        super().__init__(config, layer_idx)
        self.v_head_dim = config.v_head_dim
        self.num_key_value_groups = num_attn_heads // num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, num_attn_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * config.v_head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(num_attn_heads * config.v_head_dim, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(num_attn_heads), requires_grad=False) if is_swa else None

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
        value_states = self.v_proj(hidden_states).view(v_hidden_shape).transpose(1, 2)

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


class MiMoV2FlashDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MiMoV2FlashSparseMoeBlock(config)


@auto_docstring
class MiMoV2FlashPreTrainedModel(MixtralPreTrainedModel):
    config: MiMoV2FlashConfig
    _no_split_modules = ["MiMoV2FlashDecoderLayer"]
    _supports_sdpa = False  # disabling SDPA as it has no sink API atm (same as gpt-oss)
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"^model\.mtp\."]
    _can_record_outputs = {
        "hidden_states": MiMoV2FlashDecoderLayer,
        "attentions": MiMoV2FlashAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, MiMoV2FlashAttention) and module.sinks is not None:
            init.normal_(module.sinks, mean=0.0, std=std)
        elif isinstance(module, MiMoV2FlashTopKRouter):
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MiMoV2FlashRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class MiMoV2FlashModel(MixtralModel):
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
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Build causal mask mapping: full attention + sliding window
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MiMoV2FlashForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "MiMoV2FlashConfig",
    "MiMoV2FlashForCausalLM",
    "MiMoV2FlashModel",
    "MiMoV2FlashPreTrainedModel",
]
