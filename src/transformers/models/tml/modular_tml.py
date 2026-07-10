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
# coding=utf-8

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, LinearAttentionCacheLayerMixin
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...integrations import use_experts_implementation, use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
    torch_compilable_check,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import is_causal_conv1d_available
from ...utils.output_capturing import capture_outputs
from ..gemma3.modeling_gemma3 import Gemma3ModelOutputWithPast, Gemma3CausalLMOutputWithPast
from ..llama.modeling_llama import LlamaRMSNorm, repeat_kv


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

logger = logging.get_logger(__name__)


class TmlTextConfig(PreTrainedConfig):
    model_type = "tml_text"
    base_config_key = "text_config"
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    attribute_map = {
        "embedding_multiplier": "logits_mup_width_multiplier",
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 201024
    hidden_size: int = 6144
    num_hidden_layers: int = 66
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 16
    swa_head_dim: int = 128
    sliding_window_size: int = 512
    d_rel: int = 16
    rel_extent: int = 1024
    local_layer_ids: list[int] | None = None
    layer_types: list[str] | None = None
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    mlp_layer_types: list[int] | None = None
    intermediate_size: int = 24576
    hidden_act: str = "silu"
    # MoE
    moe_intermediate_size: int = 3072
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    shared_expert_sink: bool = True
    route_scale: float = 8.0

    logits_mup_width_multiplier: float = 24.0
    rms_norm_eps_moe_gate: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            if self.local_layer_ids is not None:
                local_layer_ids = set(self.local_layer_ids)
            else:
                local_layer_ids = {i for i in range(self.num_hidden_layers) if (i + 1) % 6}
            self.layer_types = [
                "sliding_attention" if i in local_layer_ids else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            dense_mlp_idx = kwargs.pop("dense_mlp_idx", 0)
            self.mlp_layer_types = ["dense" if i < dense_mlp_idx else "sparse" for i in range(self.num_hidden_layers)]

        if kwargs.get("dense_intermediate_size") is not None:
            self.intermediate_size = kwargs.pop("dense_intermediate_size")

        super().__post_init__(**kwargs)


class TmlAudioConfig(PreTrainedConfig):
    model_type = "tml_audio"
    base_config_key = "audio_config"

    n_mel_bins: int = 80
    mel_vocab_size: int = 256
    text_hidden_size: int = 6144
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02


class TmlVisionConfig(PreTrainedConfig):
    model_type = "tml_vision"
    base_config_key = "vision_config"
    attribute_map = {"num_hidden_layers": "n_layers"}

    text_hidden_size: int = 6144
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02


class TmlConfig(PreTrainedConfig):
    """Top-level multimodal config (`TmlMMConfig` in the SGLang source)."""

    model_type = "tml"
    sub_configs = {
        "text_config": TmlTextConfig,
        "audio_config": TmlAudioConfig,
        "vision_config": TmlVisionConfig,
    }

    text_config: TmlTextConfig | dict | None = None
    audio_config: TmlAudioConfig | dict | None = None
    vision_config: TmlVisionConfig | dict | None = None
    # `<|content_image|>` / `<|content_audio_input|>` in the tml tokenizer
    image_token_id: int = 200005
    audio_token_id: int = 200020

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config = self.sub_configs["audio_config"](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = self.sub_configs["audio_config"]()

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.vision_config.text_hidden_size = self.text_config.hidden_size
        self.audio_config.text_hidden_size = self.text_config.hidden_size
        super().__post_init__(**kwargs)


class TmlShortConvolutionsLayer(LinearAttentionCacheLayerMixin):
    is_compileable = False
    layer_type = "tml_short_conv"

    def __init__(self, **kwargs):
        self.conv_states = dict.fromkeys(range(4))
        self.is_conv_states_initialized = dict.fromkeys(range(4), False)
        self.has_previous_state = dict.fromkeys(range(4), False)

    def lazy_initialization(self, conv_states: torch.Tensor | None = None, conv_idx: int = 0) -> None:
        if conv_states is not None:
            self.dtype, self.device = conv_states.dtype, conv_states.device
            # Even if prefill is larfer/shorter than the conv_size, the tensor is always either padded or truncated
            self.batch_size, self.conv_kernel_size = conv_states.shape[0], conv_states.shape[-1]
            # The shape is always static, so we init as such
            self.conv_states[conv_idx] = torch.zeros_like(conv_states, dtype=self.dtype, device=self.device)
            # Mark as static address to be able to use cudagraphs
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.conv_states[conv_idx])
            self.is_conv_states_initialized[conv_idx] = True

    def update_conv_state(self, conv_states: torch.Tensor, conv_idx: int, **kwargs) -> torch.Tensor:
        """
        Update the linear attention cache in-place, and return the necessary conv states.

        Args:
            conv_states (`torch.Tensor`): The new conv states to cache.
            conv_idx (`int`): The layer idx of conv layer ot update.

        Returns:
            `torch.Tensor`: The updated conv states.
        """
        if conv_idx is None:
            raise ValueError("`conv_idx` has to be provided!")

        if conv_idx not in self.conv_states:
            raise ValueError(f"`conv_idx`={conv_idx} is not initialized!")

        # Lazy initialization
        if not self.is_conv_states_initialized[conv_idx]:
            self.lazy_initialization(conv_states=conv_states, conv_idx=conv_idx)

        if not self.has_previous_state[conv_idx]:
            # Note that we copy instead of assigning, to preserve the static address for cudagraphs
            self.conv_states[conv_idx].copy_(conv_states)
            self.has_previous_state[conv_idx] = True
        # Technically, this update is not logically correct if the prefill is smaller than `conv_kernel_size`,
        # as it will `roll` anyway in the first decoding step, even though it should `roll` ONLY if the cache is already full.
        # But since `conv_kernel_size=4` in practice, it's almost impossible to have a smaller prefill so it's mostly fine for now
        else:
            # Note that we copy instead of assigning, to preserve the static address for cudagraphs
            num_new_tokens = conv_states.shape[-1]
            if num_new_tokens >= self.conv_kernel_size:
                self.conv_states[conv_idx].copy_(conv_states[..., -self.conv_kernel_size :])
            else:
                new_conv_states = self.conv_states[conv_idx].roll(shifts=-num_new_tokens, dims=-1)
                new_conv_states[:, :, -num_new_tokens:] = conv_states
                self.conv_states[conv_idx].copy_(new_conv_states)

        return self.conv_states[conv_idx]

    def update_recurrent_state(self, *args, **kwargs):
        raise NotImplementedError("Model does not use any recurrent cache!")


class TmlModelOutputWithPast(Gemma3ModelOutputWithPast):
    pass


class TmlCausalLMOutputWithPast(Gemma3CausalLMOutputWithPast):
    pass


class TmlRMSNorm(LlamaRMSNorm):
    pass


class TmlRelativeLogits(nn.Module):
    """hidden states conditioned relative position bias. `proj` is a trained bank of bias-vs-distance profiles; each token's
    `relative_states` mixes them into one bias value per backward distance
    (`sglang RelLogitsProj` + the FA4 `score_mod`, materialized densely). The bias is zero
    outside `0 <= distance < rel_extent`; causality and padding stay in the attention mask.
    """

    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent))

    def forward(
        self,
        relative_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        # relative_states: [batch, q_len, num_heads, d_rel] -> bias: [batch, num_heads, q_len, kv_len]
        rel_logits = (relative_states @ self.proj).transpose(1, 2)
        distance = (query_positions[:, None] - key_positions[None, :])[None, None, :, :]
        gather_index = distance.clamp(0, self.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        position_bias = rel_logits.gather(-1, gather_index)
        return position_bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    position_bias: torch.Tensor | None = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if position_bias is not None:
        attn_weights = attn_weights + position_bias
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class TmlAttention(nn.Module):
    def __init__(self, config: TmlTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_local_attn = config.layer_types[self.layer_idx] == "sliding_attention"
        self.head_dim = config.swa_head_dim if self.is_local_attn else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_local_attn else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if self.is_local_attn else config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window_size if self.is_local_attn else None
        self.rel_extent = config.sliding_window_size if self.is_local_attn else config.rel_extent
        # q/k are RMS-normalized per head, hence 1/d rather than 1/sqrt(d)
        self.scaling = 1.0 / self.head_dim
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.r_proj = nn.Linear(config.hidden_size, self.num_heads * config.d_rel, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.k_sconv = TmlShortConvolution(
            self.num_key_value_heads * self.head_dim, config.sconv_kernel_size, layer_idx, conv_idx=0
        )
        self.v_sconv = TmlShortConvolution(
            self.num_key_value_heads * self.head_dim, config.sconv_kernel_size, layer_idx, conv_idx=1
        )
        self.q_norm = TmlRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = TmlRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rel_logits_proj = TmlRelativeLogits(config.d_rel, self.rel_extent)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_params: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_sconv(self.k_proj(hidden_states), cache_params=cache_params)
        value_states = self.v_sconv(self.v_proj(hidden_states), cache_params=cache_params)
        relative_states = self.r_proj(hidden_states)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        q_length = query_states.shape[2]
        if past_key_values is not None:
            # Important to get those values before updating the cache to be correct
            kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, self.layer_idx)
            q_offset = past_key_values.get_seq_length()
            # Update the cache
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        else:
            kv_length = key_states.shape[2]
            q_offset, kv_offset = 0, 0

        kv_positions = torch.arange(kv_length, device=hidden_states.device) + kv_offset
        q_positions = torch.arange(q_length, device=hidden_states.device) + q_offset
        relative_states = relative_states.view(*input_shape, self.num_heads, -1)
        position_bias = self.rel_logits_proj(relative_states, q_positions, kv_positions)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
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
            position_bias=position_bias,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TmlMLP(nn.Module):
    def __init__(self, config: TmlTextConfig, intermediate_size=None):
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.global_scale = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up_states = self.up_proj(hidden_states)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states) * self.global_scale


# Same as DeepSeekV3, should copy!
@use_experts_implementation
class TmlExperts(nn.Module):
    def __init__(self, config: TmlConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class TmlTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_total_experts = self.num_experts + self.n_shared_experts
        self.hidden_dim = config.hidden_size
        self.route_scale = config.route_scale
        self.top_k = config.num_experts_per_tok

        self.weight = nn.Parameter(torch.empty(self.n_total_experts, config.hidden_size))
        self.global_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(flat.type(torch.float32), self.weight.type(torch.float32))

        # same as `self.route_tokens_to_experts` from before, prob same as our MoE and can be copied
        scores = router_logits.sigmoid()
        routed_scores = scores[..., : -self.n_shared_experts]
        scores_for_choice = routed_scores + self.e_score_correction_bias
        topk_indices = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)[1]

        routed_logits = router_logits[..., : -self.n_shared_experts]
        shared_logits = router_logits[..., -self.n_shared_experts :]
        topk_logits = torch.cat([routed_logits.gather(-1, topk_indices), shared_logits], dim=-1)
        topk_log_probs = F.logsigmoid(topk_logits)
        topk_weights = torch.exp(topk_log_probs - torch.logsumexp(topk_log_probs, dim=-1, keepdim=True))

        topk_weights = topk_weights * self.route_scale
        topk_weights = topk_weights * self.global_scale

        shared_gammas = topk_weights[..., -self.n_shared_experts :].contiguous()
        topk_weights = topk_weights[..., : self.top_k].contiguous()

        return routed_logits, topk_weights, topk_indices, shared_gammas


# TODO: should we make this as normal MLP with linear layers?
class TmlSharedExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_shared_experts = config.n_shared_experts
        shared_d_mlp = config.moe_intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(config.n_shared_experts, shared_d_mlp, config.hidden_size))
        self.up_proj = nn.Parameter(torch.empty(config.n_shared_experts, shared_d_mlp, config.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(config.n_shared_experts, config.hidden_size, shared_d_mlp))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, gammas):
        input_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, input_shape[-1])
        gammas = gammas.reshape(-1, self.n_shared_experts).transpose(0, 1)  # (S, T)

        expanded = hidden_states.unsqueeze(0).expand(self.n_shared_experts, -1, -1)  # (S, T, D)
        gate = torch.bmm(expanded, self.gate_proj.mT)  # (S, T, f)
        up = torch.bmm(expanded, self.up_proj.mT)  # (S, T, f)
        activated = (self.act_fn(gate) * up).float() * gammas.unsqueeze(-1)  # (S, T, f)
        down = torch.bmm(activated.to(gate.dtype), self.down_proj.mT)  # (S, T, D)

        out = down.float().sum(dim=0).to(hidden_states.dtype)  # (T, D)
        return out.view(input_shape)


class TmlMoE(nn.Module):
    """Gate -> routed experts (+ shared experts), TML flavour."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = TmlTopkRouter(config)
        self.experts = TmlExperts(config)
        self.shared_experts = TmlSharedExperts(config)

    def forward(self, hidden_states) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        routed_logits, topk_weights, topk_indices, shared_gammas = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights.type_as(hidden_states)).view(
            *orig_shape
        )

        hidden_states = hidden_states + self.shared_experts(residuals, gammas=shared_gammas)
        return hidden_states


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


class TmlShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, conv_kernel_size: int, layer_idx: int, conv_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx
        self.conv_kernel_size = conv_kernel_size
        self.activation = None  # just hardcode for now

        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=conv_kernel_size,
            groups=hidden_size,
            padding=conv_kernel_size - 1,
            bias=False,
        )

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update

        if not (causal_conv1d_fn is not None and causal_conv1d_update is not None):
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Keep the computation in fp32
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        residual = hidden_states
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)

        # We have cached `conv_state` to continue from. The two cached modes
        # (single-token decode and chunk-tokens continuation) share the state read here; they only
        # diverge in how the conv input is assembled and which kernel consumes the states below
        use_precomputed_states = (
            cache_params is not None and cache_params.layers[self.layer_idx].has_previous_state[self.conv_idx]
        )

        # getting projected states from cache if it exists
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[self.conv_idx]

        if use_precomputed_states and seq_len == 1:
            # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
            hidden_states = self.causal_conv1d_update(
                hidden_states,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # Multi-token forward (prefill, or chunked-tokens decode when the cache has prior state).
            if use_precomputed_states:
                # Cached chunked-tokens decode: prepend the cached conv context so the causal conv
                # sees the correct left-context rather than zero-padding. Dropped from the output
                # at the end of this branch.
                hidden_states = torch.cat([conv_state, hidden_states], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx, conv_idx=self.conv_idx)
            if self.causal_conv1d_fn is not None:
                hidden_states = self.causal_conv1d_fn(
                    x=hidden_states,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=kwargs.get("seq_idx"),
                )
            else:
                hidden_states = self.conv1d(hidden_states)[:, :, : hidden_states.shape[-1]]
            if use_precomputed_states:
                hidden_states = hidden_states[:, :, -seq_len:]

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = (hidden_states + residual).to(dtype=orig_dtype)
        return hidden_states


class TmlDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TmlTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TmlAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = TmlMoE(config)
        else:
            self.mlp = TmlMLP(config)

        self.input_layernorm = TmlRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TmlRMSNorm(config.hidden_size, config.rms_norm_eps)
        # Maybe use_conv is always `True`, check it!
        self.layer_type = config.layer_types[layer_idx]
        self.attn_sconv = TmlShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=2
        )
        self.mlp_sconv = TmlShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=3
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_params: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_params=cache_params,
            **kwargs,
        )
        # TODO: the short convolutions are stateless for now, so cached decoding is wrong for
        # sequences continued token-by-token; they need their conv context carried in the cache
        # TOFIX as well ig
        hidden_states = self.attn_sconv(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_sconv(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class TmlPreTrainedModel(PreTrainedModel):
    config_class = TmlConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TmlDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # The relative position bias flows through the attention interface as a `position_bias` (duh)
    # kwarg that only the eager path consumes; other backends need a score_mod/kernel
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]
    _keep_in_fp32_modules_strict = ["attn_sconv", "mlp_sconv", "k_sconv", "v_sconv"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, TmlRelativeLogits):
            init.normal_(module.proj, mean=0.0, std=std)
        elif isinstance(module, TmlMLP):
            init.ones_(module.global_scale)
        elif isinstance(module, TmlExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, TmlTopkRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.ones_(module.global_scale)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, TmlSharedExperts):
            init.normal_(module.gate_proj, mean=0.0, std=std)
            init.normal_(module.up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)


@auto_docstring
class TmlTextModel(TmlPreTrainedModel):
    config: TmlTextConfig

    def __init__(self, config: TmlTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TmlDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TmlRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = TmlRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_params: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            if cache_params is None:
                layers = [TmlShortConvolutionsLayer() for _ in range(self.config.num_hidden_layers)]
                cache_params = Cache(layers=layers, offloading=False)  # hardcode for now

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
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
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.layer_type],
                past_key_values=past_key_values,
                cache_params=cache_params,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class TmlAudioModel(TmlPreTrainedModel):
    def __init__(self, config: TmlAudioConfig):
        super().__init__(config)
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.encoder = nn.Embedding(config.n_mel_bins * config.mel_vocab_size, config.text_hidden_size)
        self.final_norm = TmlRMSNorm(config.text_hidden_size, eps=1e-6)

        embedding_indices = torch.arange(self.n_mel_bins) * self.mel_vocab_size
        self.register_buffer("embedding_indices", embedding_indices.unsqueeze(0), persistent=False)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        if input_features.shape[1] != self.n_mel_bins:
            raise ValueError("`input_features` have to have exactly `num_mel_bin` length!")

        input_features = input_features.to(torch.int32)
        embedding_indices = self.embedding_indices.to(input_features.device) + input_features

        hidden_states = (
            self.encoder(embedding_indices.reshape(-1))
            .reshape(input_features.shape[0], self.n_mel_bins - 1)
            .sum(axis=1)
        )

        hidden_states = self.final_norm(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states,
        )


class TmlVisionEncoderLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, t_fold: int, hw_fold: int, add_norm: bool):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        if add_norm:
            self.layer_norm = TmlRMSNorm(output_dim)
        self.hw_fold = hw_fold
        self.t_fold = t_fold
        self.add_norm = add_norm

    def fold_timespace_to_depth(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of shape (B, T, H, W, C) to a tensor of shape (B, T // t, H // hw, W //  hw, C * (t * hw**2))
        """
        B, T, H, W, C = hidden_states.shape

        t_new = T // self.t_fold
        h_new = H // self.hw_fold
        w_new = W // self.hw_fold

        hidden_states = hidden_states.reshape(B, t_new, self.t_fold, h_new, self.hw_fold, w_new, self.hw_fold, C)

        hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6, 7)
        hidden_states = hidden_states.reshape(B, t_new, h_new, w_new, self.t_fold * self.hw_fold * self.hw_fold * C)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.hw_fold > 1 or self.t_fold > 1:
            hidden_states = self.fold_timespace_to_depth(hidden_states)

        hidden_states = self.projection(hidden_states)
        if self.add_norm:
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = F.gelu(hidden_states)
        return hidden_states


def prime_factors(number: int) -> list[int]:
    factors = []

    while number % 2 == 0:
        factors.append(2)
        number //= 2

    for p in range(3, math.isqrt(number) + 1, 2):
        while number % p == 0:
            factors.append(p)
            number //= p

    if number > 1:
        factors.append(number)
    return factors


def plan_out_scales(
    temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int, device="cpu"
) -> torch.LongTensor:
    """
    Plan out the dimensions for each layer in the HMLP encoder.

    This function determines the progression of dimensions (temporal, height, width, channels)
    for a multi-layer perceptual model that processes image/video patches. It follows these
    principles:
    1. Start with small dimensions and increase to full size
    2. Expand spatial dimensions (height/width) first, then temporal
    3. Increase channel count to avoid information bottlenecks
    4. Round channel dimensions to multiples of 64 for hardware efficiency

    The function computes optimal assignments of scale configurations to layers using either:
    - For n_layers >= len(scales): Individual best matching scales for each layer (allowing duplicates)
    - For n_layers < len(scales): Global optimal assignment via linear_sum_assignment

    The first and last scales are always fixed to ensure the proper input and output dimensions.

    Args:
        temporal_patch_size: Temporal dimension of input patches
        patch_size: Spatial dimension (height/width) of input patches
        n_layers: Number of layers in the encoder
        n_channels: Number of input channels (default: 3 for RGB)

    Returns:
        torch.LongTensor of shape `(n_layers + 1, 4)` where the last dim holds values for (t, h, w, c) grids.
    """
    h = torch.cumprod(torch.tensor(prime_factors(patch_size)[::-1], device=device), dim=0)
    t = torch.cumprod(torch.tensor(prime_factors(temporal_patch_size)[::-1], device=device), dim=0)

    h_ch = torch.ceil(h**2 * n_channels / 64).int() * 64
    t_ch = torch.ceil(h[-1] ** 2 * n_channels * t).int() * 64

    base = torch.tensor([[1, 1, 1, n_channels]], device=device)
    spatial = torch.stack([torch.ones_like(h), h, h, h_ch], dim=1)
    temporal = torch.stack([t, torch.full_like(t, h[-1]), torch.full_like(t, h[-1]), t_ch], dim=1)
    scales = torch.cat([base, spatial, temporal], dim=0)

    size_reduction = torch.prod(scales[:, :-1], dim=1).float()

    total_elements = patch_size * patch_size * temporal_patch_size * n_channels
    log_ideal_scales = torch.linspace(
        0, torch.log(torch.tensor(total_elements, device=device)), n_layers + 1, device=device
    )
    cost_matrix = torch.abs(log_ideal_scales.unsqueeze(1) - torch.log(size_reduction).unsqueeze(0))

    if n_layers >= scales.shape[0]:
        idxs = torch.argmin(cost_matrix, dim=1)
    else:
        from scipy.optimize import linear_sum_assignment

        _, idxs_np = linear_sum_assignment(cost_matrix.cpu().numpy())
        idxs = torch.tensor(idxs_np, device=device)
        # idxs = torch.softmax(-cost_matrix * 10, dim=1).argmax(dim=1)

    idxs[0] = 0
    idxs[-1] = scales.shape[0] - 1
    return scales[idxs]


class TmlVisionModel(TmlPreTrainedModel):
    def __init__(self, config: TmlVisionConfig):
        super().__init__(config)
        self.scales = plan_out_scales(
            config.temporal_patch_size,
            config.patch_size,
            config.num_hidden_layers,
            config.num_channels,
        )

        # num_hidden_layers - 1 to encoder and the last to proj to text hidden dim
        self.encoder_layers = nn.ModuleList()
        for i, (start_scale, end_scale) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            shuffle_mult = (
                (end_scale[0] // start_scale[0]) * (end_scale[1] // start_scale[1]) * (end_scale[2] // start_scale[2])
            )
            output_dim = config.text_hidden_size if i == config.num_hidden_layers - 1 else end_scale[3]
            hw_fold = end_scale[1] // start_scale[1]
            t_fold = end_scale[0] // start_scale[0]
            self.encoder_layers.append(
                TmlVisionEncoderLayer(
                    input_dim=start_scale[3] * shuffle_mult,
                    output_dim=output_dim,
                    hw_fold=hw_fold,
                    t_fold=t_fold,
                    add_norm=i != config.num_hidden_layers - 1,
                )
            )

        self.final_norm = TmlRMSNorm(config.text_hidden_size)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        num_patches = pixel_values.shape[0]
        hidden_states = pixel_values
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states=hidden_states)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = hidden_states.reshape(num_patches, -1)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The Base Tml model which consists of a vision backbone and a language model without language modeling head.,
    """
)
class TmlModel(TmlPreTrainedModel):
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    def __init__(self, config: TmlConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = TmlTextModel(config.text_config)
        self.audio_tower = TmlAudioModel(config.audio_config)
        self.vision_tower = TmlVisionModel(config.vision_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the vision model into language model space.")
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        return self.vision_tower(pixel_values=pixel_values, **kwargs)

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        torch_compilable_check(
            n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {n_image_features}",
        )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_params: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **lm_kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TmlModelOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, TmlForConditionalGeneration

        >>> model = TmlForConditionalGeneration.from_pretrained("google/tml2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/tml2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_norm(self.get_input_embeddings()(input_ids))

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config.get_text_config(),
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **lm_kwargs,
        )

        return TmlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


@auto_docstring(
    custom_intro="""
    The Base Tml model which consists of a vision backbone and a language model without language modeling head.,
    """
)
class TmlForConditionalGeneration(TmlPreTrainedModel, GenerationMixin):
    # `embed` and `unembed` are separate tensors in the checkpoints, never tied
    _tied_weights_keys = {}
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    # Fix: https://github.com/huggingface/transformers/issues/40564
    accepts_loss_kwargs = False

    def __init__(self, config: TmlConfig):
        super().__init__(config)
        self.model = TmlModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @auto_docstring
    def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]):
        return self.model.get_image_features(pixel_values, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_params: Cache | None = None,
        input_features: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TmlCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, TmlForConditionalGeneration

        >>> model = TmlForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
        >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {"type": "text", "text": "You are a helpful assistant."}
        ...         ]
        ...     },
        ...     {
        ...         "role": "user", "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Where is the cat standing?"},
        ...         ]
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ...     add_generation_prompt=True
        ... )
        >>> # Generate
        >>> generate_ids = model.generate(**inputs)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            **kwargs,
        )

        hidden_states = outputs[0] / self.config.text_config.logits_mup_width_multiplier
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return TmlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        input_features=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- custom `pixel_values/input_features` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["input_features"] = input_features

        return model_inputs

    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        generation_mode,
        batch_size,
        max_cache_length,
        **kwargs,
    ):
        super()._prepare_cache_for_generation(
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            generation_mode=generation_mode,
            batch_size=batch_size,
            max_cache_length=max_cache_length,
            **kwargs,
        )
        text_config = self.config.get_text_config(decoder=True)
        layers = [TmlShortConvolutionsLayer(config=text_config) for I in range(text_config.num_hidden_layers)]
        model_kwargs["cache_params"] = Cache(layers=layers, offloading=False)  # hardcode for now


__all__ = [
    "TmlConfig",
    "TmlTextConfig",
    "TmlAudioConfig",
    "TmlVisionConfig",
    "TmlPreTrainedModel",
    "TmlTextModel",
    "TmlAudioModel",
    "TmlVisionModel",
    "TmlModel",
    "TmlForConditionalGeneration",
]
