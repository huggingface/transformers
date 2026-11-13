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

import copy
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, ImageInput, ImageType, SizeDict, get_image_type
from ...integrations import use_experts_implementation, use_kernelized_func
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import ImagesKwargs, ProcessorMixin, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ...vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
from ..clip.modeling_clip import CLIPMLP
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter, DeepseekV4UnweightedRMSNorm
from ..inkling.configuration_inkling import InklingConfig, InklingTextConfig
from ..inkling.modeling_inkling import (
    InklingAttention,
    InklingCausalLMOutputWithPast,
    InklingExperts,
    InklingForCausalLM,
    InklingForConditionalGeneration,
    InklingModel,
    InklingModelOutputWithPast,
    InklingPreTrainedModel,
    InklingRelativeLogits,
    InklingShortConvolution,
    InklingTextModel,
)
from ..llama.modeling_llama import repeat_kv
from ..llava.modeling_llava import LlavaMultiModalProjector
from ..olmo2.modeling_olmo2 import Olmo2RMSNorm
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from ..qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states, causal_conv1d_fn, causal_conv1d_update
from ..sam3.modeling_sam3 import Sam3ViTRotaryEmbedding
from ..step3p7.configuration_step3p7 import Step3p7VisionConfig
from ..step3p7.modeling_step3p7 import Step3p7VisionEncoderLayer
from ..video_llama_3.modeling_video_llama_3 import VideoLlama3VisionAttention


@auto_docstring(checkpoint="thinkingmachines/Inkling-2-Small-MXFP8")
@strict
class Inkling2TextConfig(InklingTextConfig):
    r"""
    unpadded_vocab_size (`int`, *optional*, defaults to 200058):
        Number of rows of the unembedding matrix that hold real tokens. Logits beyond it are dropped.
    swa_num_attention_heads (`int`, *optional*, defaults to 48):
        Number of attention heads in the sliding-window layers.
    swa_num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of key/value heads in the sliding-window layers.
    swa_head_dim (`int`, *optional*, defaults to 128):
        Dimension of query and key heads in the sliding-window layers.
    sliding_window_size (`int`, *optional*, defaults to 1024):
        Size of the sliding attention window used by layers whose `layer_types` entry is `"hybrid_sliding"`.
    d_rel (`int`, *optional*, defaults to 16):
        Per-head dimension of the relative states that are mixed into the relative position bias.
    rel_extent (`int`, *optional*, defaults to 1024):
        Backward distance, in tokens, over which the relative position bias is applied in the full-attention layers.
    log_scaling_n_floor (`int`, *optional*, defaults to 128000):
        Position from which queries and relative states are scaled up logarithmically in the full-attention layers.
    log_scaling_alpha (`float`, *optional*, defaults to 0.1):
        Strength of the logarithmic scaling controlled by `log_scaling_n_floor`.
    qk_l2_clip_threshold (`float`, *optional*, defaults to 22.627416997969522):
        Maximum L2 norm of each query and key head vector. Longer vectors are scaled down to this norm.
    rel_l2_clip_threshold (`float`, *optional*, defaults to 8.0):
        Maximum L2 norm of each relative-state head vector.
    n_layers_per_attn_res_block (`int`, *optional*, defaults to 6):
        Number of decoder layers summed into one block before the block becomes a source of the attention residuals.
    mlp_gate_softcap (`float`, *optional*, defaults to 4.0):
        Softcap applied to the linear part of the experts' gate projection.
    mlp_up_softcap (`float`, *optional*, defaults to 25.0):
        Softcap applied to the experts' up projection.
    final_logit_softcapping (`float`, *optional*, defaults to 30.0):
        Softcap applied to the output logits.
    scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
        Activation turning the router logits into expert scores.
    """

    model_type = "inkling2_text"
    base_model_tp_plan = {
        "embed_tokens": "replicate",
        "layers.*.mlp.experts.gate_up_proj": "colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = AttributeError()
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }
    attribute_map = {
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
        "sconv_kernel_size": "conv_kernel_size",
        "model_max_length": "max_position_embeddings",
        "intermediate_size": "moe_intermediate_size",
        "routed_scaling_factor": "route_scale",
    }

    unpadded_vocab_size: int = 200058
    hidden_size: int = 4608
    num_hidden_layers: int = 48
    num_attention_heads: int = 48
    swa_num_attention_heads: int = 48
    swa_num_key_value_heads: int = 8
    sliding_window_size: int = 1024
    log_scaling_n_floor: int = 128000
    max_position_embeddings: int = 1048576
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    num_experts_per_tok: int = 8
    bos_token_id: int | None = None
    eos_token_id: int | None = 200006
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    scoring_func: str = "sigmoid"
    use_cache: bool = True

    intermediate_size = AttributeError()
    n_shared_experts = AttributeError()
    mlp_layer_types = AttributeError()
    shared_expert_sink = AttributeError()
    logits_mup_width_multiplier = AttributeError()
    rms_norm_eps_moe_gate = AttributeError()
    num_mtp_layers = AttributeError()
    chain_hidden_post_norm = AttributeError()
    mtp_hidden_states_first = AttributeError()
    local_layer_ids = AttributeError()
    mtp_local_layer_ids = AttributeError()
    number_of_conv_states = 4

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(self.num_hidden_layers)
            ]
        PreTrainedConfig.__post_init__(self, **kwargs)

    def mtp_layer_types(self):
        raise AttributeError("Not needed for Inkling2")

    def mtp_mlp_layer_types(self):
        raise AttributeError("Not needed for Inkling2")


@auto_docstring(checkpoint="thinkingmachines/Inkling-2-Small-MXFP8")
@strict
class Inkling2VisionConfig(Step3p7VisionConfig):
    r"""
    spatial_merge_size (`int`, *optional*, defaults to 2):
        Side, in native patches, of the square of patch features concatenated into one language model token.
    posemb_grid_size (`int`, *optional*, defaults to 32):
        Side of the learned position embedding grid, interpolated to each image's native patch grid.
    projection_hidden_dim (`int`, *optional*, defaults to 4096):
        Hidden size of the MLP projecting merged patch features to the language model.
    projector_bias (`bool`, *optional*, defaults to `False`):
        Whether the projection MLP uses biases.
    text_hidden_size (`int`, *optional*, defaults to 4608):
        Hidden size of the language model the image features are projected to.
    """

    model_type = "inkling2_vision"
    attribute_map = {"native_patch_size": "patch_size", "adapter_merge_size": "spatial_merge_size"}

    hidden_size: int = 1024
    intermediate_size: int | None = None
    num_hidden_layers: int = 23
    patch_size: int = 20
    hidden_act: str = "gelu"
    mlp_ratio: float = 4.0
    spatial_merge_size: int = 2
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    text_hidden_size: int = 4608
    initializer_range: float = 0.02

    image_size = AttributeError()
    max_position_embeddings = AttributeError()

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * self.mlp_ratio)
        PreTrainedConfig.__post_init__(self, **kwargs)


# Missing from the hub config: the routed experts are MXFP8 (e4m3, a ue8m0 scale per row and 32 values), the rest bf16
INKLING2_QUANTIZATION_CONFIG = {
    "quant_method": "fp8",
    "weight_block_size": [1, 32],
    "scale_fmt": "ue8m0",
    "modules_to_not_convert": ["lm_head", "model.vision_tower", "model.language_model.layers.*.self_attn"],
}


@auto_docstring(checkpoint="thinkingmachines/Inkling-2-Small-MXFP8")
@strict
class Inkling2Config(InklingConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 200054):
        Token index of the image placeholders, each replaced by one image feature.
    """

    model_type = "inkling2"
    sub_configs_defaults = {"text_config": Inkling2TextConfig, "vision_config": Inkling2VisionConfig}

    audio_config = AttributeError()
    audio_token_id = AttributeError()
    audio_bos_token_id = AttributeError()
    image_bos_token_id = AttributeError()

    def __post_init__(self, **kwargs):
        raise AttributeError("Not needed for Inkling2")

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        # Quantization config is missing from the hub config: only added to an original checkpoint's config, not to
        # one saved by Transformers (which has its own, or none once dequantized)
        if "quantization_config" not in config_dict and "transformers_version" not in config_dict:
            config_dict = {**config_dict, "quantization_config": copy.deepcopy(INKLING2_QUANTIZATION_CONFIG)}
        return PreTrainedConfig.from_dict(config_dict, **kwargs)

    def validate_architecture(self):
        if self.vision_config.text_hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"`vision_config.text_hidden_size` ({self.vision_config.text_hidden_size}) must match "
                f"`text_config.hidden_size` ({self.text_config.hidden_size})."
            )


class Inkling2ModelOutputWithPast(InklingModelOutputWithPast):
    pass


class Inkling2CausalLMOutputWithPast(InklingCausalLMOutputWithPast):
    pass


# Different rounding order than original Inkling
class Inkling2RMSNorm(Olmo2RMSNorm):
    pass


class Inkling2UnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    pass


class Inkling2L2Clip(nn.Module):
    """Scales each head vector down to `threshold` when its L2 norm is larger; shorter vectors are unchanged."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


class Inkling2ScaledL2Clip(nn.Module):
    """`Inkling2L2Clip` applied after a learned scale."""

    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float() * self.scale.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


class Inkling2RelativeLogits(InklingRelativeLogits):
    pass


@use_kernelized_func([causal_conv1d_update, causal_conv1d_fn])
class Inkling2ShortConvolution(InklingShortConvolution):
    @force_accelerate_hooks("conv1d")
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Cache | None = None,
        conv_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        residual = hidden_states
        hidden_states = apply_mask_to_padding_states(hidden_states, conv_mask)
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)

        conv_state = None if cache is None else cache.get_conv_state(self.layer_idx, state_idx=self.conv_idx)

        if conv_state is not None and seq_len == 1 and not cache.is_recording_past(self.layer_idx):
            hidden_states = causal_conv1d_update(
                hidden_states, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias
            )
        else:
            if cache is not None:
                hidden_states = cache.update_conv_state(
                    hidden_states, self.layer_idx, state_idx=self.conv_idx, conv_kernel_size=self.conv_kernel_size
                )

            hidden_states = causal_conv1d_fn(
                hidden_states, self.conv1d.weight.squeeze(1), self.conv1d.bias, seq_idx=kwargs.get("seq_idx")
            )

            if cache is not None:
                hidden_states = hidden_states[:, :, -seq_len:]

        hidden_states = hidden_states.transpose(1, 2)
        # Different from Inkling which has no output scale
        hidden_states = (hidden_states + residual) / math.sqrt(self.conv_kernel_size + 1)
        return hidden_states.to(dtype=input_dtype)


class Inkling2DynamicShortConvolution(nn.Module):
    """
    Short causal convolution whose taps are the sum of static per-channel weights and per-token weights projected from
    `dynamic_input`, shared within each key/value head. Used for keys and values in the full-attention layers.
    """

    def __init__(
        self, hidden_size: int, input_size: int, num_heads: int, conv_kernel_size: int, layer_idx: int, conv_idx: int
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        # tap index is the backward distance: `static_weight[0]` weights the current token
        self.static_weight = nn.Parameter(torch.zeros(conv_kernel_size, hidden_size))
        self.weight_proj = nn.Linear(input_size, num_heads * conv_kernel_size, bias=False)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        dynamic_input: torch.Tensor,
        cache: Cache | None = None,
        conv_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_dtype = hidden_states.dtype
        batch_size, seq_len, hidden_size = hidden_states.shape
        dynamic_weights = self.weight_proj(dynamic_input).float()
        dynamic_weights = dynamic_weights.view(batch_size, seq_len, self.num_heads, 1, self.conv_kernel_size)
        weights = self.static_weight.float().T.view(self.num_heads, -1, self.conv_kernel_size) + dynamic_weights

        hidden_states = hidden_states.float()
        residual = hidden_states
        context = apply_mask_to_padding_states(hidden_states, conv_mask).transpose(1, 2)
        if cache is not None:
            context = cache.update_conv_state(
                context, self.layer_idx, state_idx=self.conv_idx, conv_kernel_size=self.conv_kernel_size
            )
        context = F.pad(context, (self.conv_kernel_size - 1, 0))[..., -(seq_len + self.conv_kernel_size - 1) :]
        # windows[..., lag] is the token `lag` steps back
        windows = context.unfold(-1, self.conv_kernel_size, 1).flip(-1).transpose(1, 2)
        windows = windows.reshape(batch_size, seq_len, self.num_heads, -1, self.conv_kernel_size)
        hidden_states = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
        hidden_states = (hidden_states + residual) / math.sqrt(2 * self.conv_kernel_size + 1)
        return hidden_states.to(input_dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    position_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if position_bias is not None:
        attn_weights = attn_weights + position_bias
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if sinks is not None:
        # The sink is one extra logit per query with a zero value: it only absorbs probability mass
        combined_logits = torch.cat([attn_weights, sinks], dim=-1).float()
        attn_weights = F.softmax(combined_logits, dim=-1)[..., :-1].to(query.dtype)
    else:
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Inkling2SlidingAttention(InklingAttention):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.head_dim**-0.5
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.q_l2_clip = Inkling2ScaledL2Clip(self.num_heads, self.head_dim, config.qk_l2_clip_threshold)
        self.k_l2_clip = Inkling2L2Clip(config.qk_l2_clip_threshold)
        self.rel_l2_clip = Inkling2ScaledL2Clip(self.num_heads, config.d_rel, config.rel_l2_clip_threshold)
        self.v_norm = Inkling2UnweightedRMSNorm(eps=config.rms_norm_eps)
        del self.q_norm
        del self.k_norm

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_sconv(self.k_proj(hidden_states), cache=cache, conv_mask=conv_mask)
        value_states = self.v_sconv(self.v_proj(hidden_states), cache=cache, conv_mask=conv_mask)
        relative_states = self.r_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)

        q_length = query_states.shape[1]
        if cache is not None:
            kv_length, kv_offset = cache.get_mask_sizes(q_length, self.layer_idx)
            q_offset = cache.get_query_offset(self.layer_idx)
        else:
            kv_length = q_length
            q_offset, kv_offset = 0, 0
        kv_positions = torch.arange(kv_length, device=hidden_states.device) + kv_offset
        q_positions = torch.arange(q_length, device=hidden_states.device) + q_offset

        query_states = self.q_l2_clip(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_l2_clip(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape)
        value_states = self.v_norm(value_states.float()).to(value_states.dtype)
        value_states = value_states.transpose(1, 2)
        relative_states = self.rel_l2_clip(relative_states.view(*input_shape, self.num_heads, -1))

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)

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
            sinks=sinks.transpose(1, 2).unsqueeze(-1),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Inkling2FullAttention(InklingAttention):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.head_dim**-0.5
        self.k_sconv = Inkling2DynamicShortConvolution(
            self.num_key_value_heads * self.head_dim,
            config.hidden_size,
            self.num_key_value_heads,
            config.sconv_kernel_size,
            layer_idx,
            conv_idx=0,
        )
        self.v_sconv = Inkling2DynamicShortConvolution(
            self.num_key_value_heads * self.head_dim,
            config.hidden_size,
            self.num_key_value_heads,
            config.sconv_kernel_size,
            layer_idx,
            conv_idx=1,
        )
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.q_l2_clip = Inkling2ScaledL2Clip(self.num_heads, self.head_dim, config.qk_l2_clip_threshold)
        self.k_l2_clip = Inkling2L2Clip(config.qk_l2_clip_threshold)
        self.rel_l2_clip = Inkling2ScaledL2Clip(self.num_heads, config.d_rel, config.rel_l2_clip_threshold)
        self.v_norm = Inkling2UnweightedRMSNorm(eps=config.rms_norm_eps)
        del self.q_norm
        del self.k_norm

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_hidden_states: torch.Tensor,
        value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        key_hidden_states (`torch.Tensor`):
            Input of the key projection and of the key convolution weights.
        value_hidden_states (`torch.Tensor`):
            Input of the value projection and of the value convolution weights.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_sconv(self.k_proj(key_hidden_states), key_hidden_states, cache=cache, conv_mask=conv_mask)
        value_states = self.v_sconv(
            self.v_proj(value_hidden_states),
            value_hidden_states,
            cache=cache,
            conv_mask=conv_mask,
        )
        relative_states = self.r_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)

        q_length = query_states.shape[1]
        if cache is not None:
            kv_length, kv_offset = cache.get_mask_sizes(q_length, self.layer_idx)
            q_offset = cache.get_query_offset(self.layer_idx)
        else:
            kv_length = q_length
            q_offset, kv_offset = 0, 0
        kv_positions = torch.arange(kv_length, device=hidden_states.device) + kv_offset
        q_positions = torch.arange(q_length, device=hidden_states.device) + q_offset

        # log scaling of the queries and relative states, in fp32 as in the original
        tau = 1.0 + self.config.log_scaling_alpha * torch.log(
            ((kwargs["position_ids"] + 1).float() / self.config.log_scaling_n_floor).clamp(min=1.0)
        )
        tau = tau[..., None, None]

        query_states = self.q_l2_clip(query_states.view(hidden_shape))
        query_states = (query_states.float() * tau).to(query_states.dtype).transpose(1, 2)
        key_states = self.k_l2_clip(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape)
        value_states = self.v_norm(value_states.float()).to(value_states.dtype)
        value_states = value_states.transpose(1, 2)
        relative_states = self.rel_l2_clip(relative_states.view(*input_shape, self.num_heads, -1))
        relative_states = (relative_states.float() * tau).to(relative_states.dtype)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)

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
            sinks=sinks.transpose(1, 2).unsqueeze(-1),
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Inkling2Softcap(nn.Module):
    def __init__(self, softcap: float):
        super().__init__()
        self.softcap = softcap

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(hidden_states / self.softcap) * self.softcap


class Inkling2SoftcappedSwiGLU(nn.Module):
    """SwiGLU where the sigmoid sees the raw gate and the softcaps only bound the linear gate factor and `up`."""

    def __init__(self, config: Inkling2TextConfig):
        super().__init__()
        self.gate_softcap = Inkling2Softcap(config.mlp_gate_softcap)
        self.up_softcap = Inkling2Softcap(config.mlp_up_softcap)

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        input_dtype = gate.dtype
        gate, up = gate.float(), up.float()
        return (self.gate_softcap(gate) * torch.sigmoid(gate) * self.up_softcap(up)).to(input_dtype)


@use_experts_implementation
class Inkling2Experts(InklingExperts):
    def __init__(self, config: Inkling2TextConfig):
        super().__init__(config)
        self.act_fn = Inkling2SoftcappedSwiGLU(config)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate, up)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # accumulate the top-k expert outputs in fp32, as in the original
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self._apply_gate(F.linear(current_state, self.gate_up_proj[expert_idx]))
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


class Inkling2TopKRouter(DeepseekV4TopKRouter):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # routing runs in fp32, as in the original
        hidden_states = hidden_states.float()
        return super().forward(hidden_states)


class Inkling2SparseMoeBlock(Qwen3MoeSparseMoeBlock):
    pass


class Inkling2BlockAttentionResidual(nn.Module):
    def __init__(self, config: Inkling2TextConfig, num_queries: int = 1):
        super().__init__()
        self.query = nn.Parameter(
            torch.zeros(config.hidden_size) if num_queries == 1 else torch.zeros(num_queries, config.hidden_size)
        )
        self.norm = Inkling2UnweightedRMSNorm()

    def forward(self, block_states: torch.Tensor, partial_block_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
        stacked_states = self.norm(stacked_states)

        hidden_size = stacked_states.shape[-1]
        # (batch, seq_len, num_queries, hidden_size), the same queries at every position
        query_states = self.query.float().view(-1, hidden_size).expand(*stacked_states.shape[:-2], -1, -1)
        attn_output = F.scaled_dot_product_attention(
            query_states, stacked_states, stacked_states, scale=hidden_size**-0.5
        )
        return attn_output.unbind(-2)


class Inkling2SlidingDecoderLayer(GradientCheckpointingLayer):
    attention_type = "sliding_attention"

    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.attention_input = Inkling2BlockAttentionResidual(config) if layer_idx > 0 else None
        self.mlp_input = Inkling2BlockAttentionResidual(config)
        self.self_attn = Inkling2SlidingAttention(config, layer_idx)
        self.mlp = Inkling2SparseMoeBlock(config)
        self.input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_convolution = Inkling2ShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=2
        )
        self.post_mlp_convolution = Inkling2ShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=3
        )

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is not None:
            hidden_states = self.attention_input(block_states, hidden_states)[0]

        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states).to(dtype),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            cache=cache,
            **kwargs,
        )
        hidden_states = self.post_attention_convolution(hidden_states, cache=cache, conv_mask=conv_mask)

        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.post_mlp_convolution(hidden_states, cache=cache, conv_mask=conv_mask)
        return (residual + hidden_states).to(dtype)


class Inkling2FullDecoderLayer(GradientCheckpointingLayer):
    attention_type = "full_attention"

    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.attention_input = Inkling2BlockAttentionResidual(config, num_queries=3) if layer_idx > 0 else None
        self.mlp_input = Inkling2BlockAttentionResidual(config)
        self.self_attn = Inkling2FullAttention(config, layer_idx)
        self.mlp = Inkling2SparseMoeBlock(config)
        self.q_input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.k_input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.v_input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_convolution = Inkling2ShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=2
        )
        self.post_mlp_convolution = Inkling2ShortConvolution(
            config.hidden_size, config.conv_kernel_size, layer_idx=layer_idx, conv_idx=3
        )

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is None:
            query_hidden_states = key_hidden_states = value_hidden_states = hidden_states
        else:
            query_hidden_states, key_hidden_states, value_hidden_states = self.attention_input(
                block_states, hidden_states
            )

        hidden_states, _ = self.self_attn(
            hidden_states=self.q_input_layernorm(query_hidden_states).to(dtype),
            key_hidden_states=self.k_input_layernorm(key_hidden_states).to(dtype),
            value_hidden_states=self.v_input_layernorm(value_hidden_states).to(dtype),
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            cache=cache,
            **kwargs,
        )
        hidden_states = self.post_attention_convolution(hidden_states, cache=cache, conv_mask=conv_mask)

        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.post_mlp_convolution(hidden_states, cache=cache, conv_mask=conv_mask)
        return (residual + hidden_states).to(dtype)


class Inkling2VisionRotaryEmbedding(Sam3ViTRotaryEmbedding):
    def __init__(self, config: Inkling2VisionConfig):
        super().__init__(config)

    @staticmethod
    def compute_axial_rope_parameters(
        config: Inkling2VisionConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        return super().compute_axial_rope_parameters(config, device, **kwargs)

    def recomposition_frequencies(self, freq):
        """Concatenates width then height frequencies, each repeated for one interleaved channel pair."""
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        return torch.cat([freq_w, freq_h], dim=-1).repeat_interleave(2, dim=-1)


def rotate_pairwise(x):
    """Rotates each interleaved channel pair `(x1, x2)` to `(-x2, x1)`."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_pairwise(q) * sin)
    k_embed = (k * cos) + (rotate_pairwise(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Inkling2VisionAttention(VideoLlama3VisionAttention):
    pass


class Inkling2VisionMLP(CLIPMLP):
    pass


class Inkling2VisionEncoderLayer(Step3p7VisionEncoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
        )
        hidden_states = residual + self.lambda_1 * hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.layernorm_after(hidden_states))
        return residual + self.lambda_2 * hidden_states


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Inkling2VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.merge_size = config.spatial_merge_size
        self.grid_size = config.posemb_grid_size
        self.patch_embedding = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size, bias=False)
        self.position_embedding = nn.Embedding(config.posemb_grid_size**2, config.hidden_size)
        self.class_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.cls_position_embedding = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_grid_thw: torch.Tensor,
        patch_index: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        merge, patch = self.merge_size, self.patch_size
        patches = pixel_values.reshape(-1, merge, patch, merge, patch, pixel_values.shape[-1]).transpose(2, 3)
        patch_embeddings = self.patch_embedding(patches.reshape(-1, self.patch_embedding.in_features))

        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            patch_grid_thw,
            num_grid_per_side=self.grid_size,
            mode="bilinear",
            align_corners=False,
            spatial_merge_size=merge,
            kwargs=kwargs,
        )
        interp_indices = interp_indices.to(pixel_values.device, non_blocking=True)
        interp_weights = interp_weights.to(pixel_values.device, non_blocking=True)
        position_embeddings = (self.position_embedding(interp_indices) * interp_weights[:, :, None]).sum(1)
        patch_embeddings = patch_embeddings + position_embeddings.to(patch_embeddings.dtype)

        # each image starts with a class token, followed by its patches
        cls_embedding = (self.class_embedding + self.cls_position_embedding).to(patch_embeddings.dtype)
        embeddings = cls_embedding.repeat(patch_embeddings.shape[0] + patch_grid_thw.shape[0], 1)
        embeddings[patch_index] = patch_embeddings
        return embeddings


class Inkling2VisionMerger(LlavaMultiModalProjector):
    def __init__(self, config: Inkling2VisionConfig):
        nn.Module.__init__(self)
        self.linear_1 = nn.Linear(
            config.spatial_merge_size**2 * config.hidden_size, config.projection_hidden_dim, bias=config.projector_bias
        )
        self.act = ACT2FN[config.hidden_act]
        self.linear_2 = nn.Linear(config.projection_hidden_dim, config.text_hidden_size, bias=config.projector_bias)


@auto_docstring
class Inkling2PreTrainedModel(InklingPreTrainedModel):
    config: Inkling2Config
    _no_split_modules = ["Inkling2SlidingDecoderLayer", "Inkling2FullDecoderLayer", "Inkling2VisionEncoderLayer"]
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_record_outputs = {
        "hidden_states": [Inkling2SlidingDecoderLayer, Inkling2FullDecoderLayer],
        "attentions": [Inkling2SlidingAttention, Inkling2FullAttention],
        "router_logits": OutputRecorder(Inkling2TopKRouter, index=0),
    }
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "mlp.gate.weight",
        "post_attention_convolution",
        "post_mlp_convolution",
        "k_sconv.conv1d",
        "v_sconv.conv1d",
        "attention_input",
        "mlp_input",
        "final_input",
    ]
    _keys_to_ignore_on_load_unexpected = AttributeError()

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, Inkling2RelativeLogits):
            init.normal_(module.proj, mean=0.0, std=std)
        elif isinstance(module, Inkling2Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Inkling2TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Inkling2ScaledL2Clip):
            init.ones_(module.scale)
        elif isinstance(module, Inkling2DynamicShortConvolution):
            init.zeros_(module.static_weight)
        elif isinstance(module, Inkling2BlockAttentionResidual):
            init.zeros_(module.query)
        elif isinstance(module, Inkling2VisionEmbeddings):
            init.normal_(module.class_embedding, mean=0.0, std=std)
            init.normal_(module.cls_position_embedding, mean=0.0, std=std)
        elif isinstance(module, Inkling2VisionEncoderLayer):
            init.constant_(module.lambda_1, module.config.layer_scale_init_value)
            init.constant_(module.lambda_2, module.config.layer_scale_init_value)


@auto_docstring
class Inkling2TextModel(InklingTextModel):
    def __init__(self, config: Inkling2TextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                Inkling2SlidingDecoderLayer(config, layer_idx)
                if config.layer_types[layer_idx] == "hybrid_sliding"
                else Inkling2FullDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_input = Inkling2BlockAttentionResidual(config)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])
        for decoder_layer in self.layers:
            new_hidden_states = decoder_layer(
                hidden_states,
                block_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                cache=cache,
                position_ids=position_ids,
                **kwargs,
            )
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_hidden_states

        hidden_states = self.final_input(block_states, hidden_states)[0]
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache,
        )


class Inkling2ForCausalLM(InklingForCausalLM):
    def __init__(self, config: Inkling2TextConfig):
        super().__init__(config)
        self.logit_softcap = Inkling2Softcap(config.final_logit_softcapping)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Inkling2CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, Inkling2ForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Small-MXFP8")
        >>> model = Inkling2ForCausalLM.from_pretrained(
        ...     "thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
        >>> inputs = tokenizer.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     add_generation_prompt=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device)
        >>> input_len = inputs["input_ids"].shape[-1]

        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False))
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states).float()
        logits = self.logit_softcap(logits)
        logits = logits[..., : self.config.unpadded_vocab_size]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs)

        return Inkling2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


@auto_docstring
class Inkling2VisionModel(Inkling2PreTrainedModel):
    config: Inkling2VisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": Inkling2VisionEncoderLayer,
        "attentions": Inkling2VisionAttention,
    }

    def __init__(self, config: Inkling2VisionConfig):
        super().__init__(config)
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Inkling2VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([Inkling2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.merger = Inkling2VisionMerger(config)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_image_tokens, patch_size * spatial_merge_size, patch_size * spatial_merge_size, num_channels)`):
            Image patches in `[-1, 1]`, one per language model image token, in raster order within each image.
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
            Temporal, height and width of each image, counted in language model image tokens.
        """
        merge = self.config.spatial_merge_size
        device = pixel_values.device
        patch_grid_thw = image_grid_thw.cpu() * torch.tensor([1, merge, merge])
        patches_per_image = patch_grid_thw.prod(-1)
        num_patches = pixel_values.shape[0] * merge**2
        if (cu_seqlens := kwargs.pop("cu_seqlens", None)) is None:
            cu_seqlens = F.pad((patches_per_image + 1).cumsum(0), (1, 0)).to(torch.int32)
        # sequence index of every patch, skipping the class token in front of each image
        image_ids = torch.arange(1, patch_grid_thw.shape[0] + 1)
        patch_index = torch.arange(num_patches) + torch.repeat_interleave(image_ids, patches_per_image)
        patch_index = patch_index.to(device, non_blocking=True)

        pixel_values = pixel_values.to(self.embeddings.patch_embedding.weight.dtype)
        hidden_states = self.embeddings(pixel_values, patch_grid_thw, patch_index, **kwargs)
        hidden_states = self.pre_layernorm(hidden_states)

        # patches are 1-indexed (row, col) and class tokens sit at (0, 0)
        position_ids = get_vision_position_ids(patch_grid_thw, merge, kwargs=kwargs).to(device, non_blocking=True)
        position_ids = position_ids.new_zeros(hidden_states.shape[0], 2).index_copy(0, patch_index, position_ids + 1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
            )

        # with a device map, the last layers can live on another device than the embeddings
        patch_states = hidden_states[patch_index.to(hidden_states.device)]
        pooler_output = self.merger(patch_states.reshape(-1, merge**2 * hidden_states.shape[-1]))
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooler_output)


class Inkling2Model(InklingModel):
    input_modalities = ("image", "text")

    def __init__(self, config: Inkling2Config):
        super().__init__(config)
        del self.audio_tower
        del self.vocab_size

    @can_return_tuple
    @auto_docstring(custom_intro="Encodes images into language model embeddings, one per image token.")
    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        image_grid_thw = image_grid_thw.cpu()
        image_outputs = self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)
        image_outputs.pooler_output = image_outputs.pooler_output.split(image_grid_thw.prod(-1).tolist())
        return image_outputs

    def get_audio_features(self):
        raise AttributeError("Not needed for Inkling2")

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Inkling2ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width of each image, counted in language model image tokens.

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, Inkling2Model

        >>> processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling-2-Small-MXFP8")
        >>> model = Inkling2Model.from_pretrained(
        ...     "thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        ...             },
        ...             {"type": "text", "text": "What animal is in this image, and where is it?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     add_generation_prompt=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device)

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)

        image_embeds = None
        if mm_encoder_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_embeds, self.config.image_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_embeds)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Inkling2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeds,
            router_logits=outputs.router_logits,
        )


class Inkling2ForConditionalGeneration(InklingForConditionalGeneration):
    input_modalities = ("image", "text")

    def __init__(self, config: Inkling2Config):
        super().__init__(config)
        self.logit_softcap = Inkling2Softcap(config.text_config.final_logit_softcapping)

    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Inkling2CausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width of each image, counted in language model image tokens.

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, Inkling2ForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling-2-Small-MXFP8")
        >>> model = Inkling2ForConditionalGeneration.from_pretrained(
        ...     "thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        ...             },
        ...             {"type": "text", "text": "What animal is in this image, and where is it?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     add_generation_prompt=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device)
        >>> input_len = inputs["input_ids"].shape[-1]

        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        text_config = self.config.text_config
        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states).float()
        logits = self.logit_softcap(logits)
        logits = logits[..., : text_config.unpadded_vocab_size]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=logits.shape[-1], **kwargs)

        return Inkling2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(self):
        raise AttributeError("Not needed for Inkling2")


@auto_docstring(custom_intro="Constructs an Inkling2 image processor.")
class Inkling2ImageProcessor(TorchvisionBackend):
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    do_rescale = True
    do_normalize = True
    size = {"height": 40, "width": 40}
    model_input_names = ["pixel_values", "image_grid_thw"]

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: "torch.device | None" = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> "torch.Tensor":
        # keep the alpha channel of PIL images so that `convert_to_rgb` composites it onto the checkerboard
        if do_convert_rgb and get_image_type(image) == ImageType.PIL:
            image = tvF.pil_to_tensor(image.convert("RGBA" if image.has_transparency_data else "RGB"))
            input_data_format = ChannelDimension.FIRST
        return super().process_image(
            image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device, **kwargs
        )

    def convert_to_rgb(
        self, image: "torch.Tensor", input_data_format: str | ChannelDimension | None = None
    ) -> "torch.Tensor":
        if image.shape[-3] != 4:
            return super().convert_to_rgb(image, input_data_format=input_data_format)
        height, width = image.shape[-2:]
        rows = torch.arange(height, device=image.device)[:, None] // 8
        columns = torch.arange(width, device=image.device)[None, :] // 8
        background = torch.where((rows + columns) % 2 == 0, 204, 153)
        rgba = image.to(torch.int32)
        alpha = rgba[3:4]
        return ((rgba[:3] * alpha + background * (255 - alpha) + 127) // 255).to(torch.uint8)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | None,
        **kwargs,
    ) -> BatchFeature:
        patches, image_grid_thw = [], []
        for image in images:
            channels, height, width = image.shape
            rows, columns = math.ceil(height / size.height), math.ceil(width / size.width)
            # pad to the grid with -1 on the 0-255 scale, then cut it into one patch per image token
            pad_size = SizeDict(height=rows * size.height, width=columns * size.width)
            image = self.pad([image.float()], pad_size=pad_size, fill_value=-1.0)[0]
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            image = image.reshape(channels, rows, size.height, columns, size.width).permute(1, 3, 2, 4, 0)
            patches.append(image.flatten(0, 1))
            image_grid_thw.append([1, rows, columns])
        return BatchFeature(
            data={"pixel_values": torch.cat(patches), "image_grid_thw": torch.tensor(image_grid_thw)},
            tensor_type=return_tensors,
        )


@auto_docstring
class Inkling2Processor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, image_token="<|unused_200054|>"):
        r"""
        image_token (`str`, *optional*, defaults to `"<|unused_200054|>"`):
            Placeholder token for each image soft-token slot, replaced by image features.
        """
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        return self.image_token * int(image_inputs["image_grid_thw"][image_idx].prod())


__all__ = [
    "Inkling2Config",
    "Inkling2TextConfig",
    "Inkling2VisionConfig",
    "Inkling2PreTrainedModel",
    "Inkling2TextModel",
    "Inkling2ForCausalLM",
    "Inkling2VisionModel",
    "Inkling2Model",
    "Inkling2ForConditionalGeneration",
    "Inkling2ImageProcessor",
    "Inkling2Processor",
]
