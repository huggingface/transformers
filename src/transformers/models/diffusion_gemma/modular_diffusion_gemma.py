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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import (
    bidirectional_mask_function,
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
    create_causal_mask,
    create_masks_for_generate,
    create_sliding_window_causal_mask,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from ..gemma4.modeling_gemma4 import (
    Gemma4ClippableLinear,
    Gemma4Model,
    Gemma4MultimodalEmbedder,
    Gemma4RMSNorm,
    Gemma4TextDecoderLayer,
    Gemma4TextExperts,
    Gemma4TextMLP,
    Gemma4TextRotaryEmbedding,
    Gemma4TextRouter,
    Gemma4TextScaledWordEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    get_block_sequence_ids_for_mask,
)
from ..t5gemma2.modeling_t5gemma2 import T5Gemma2Model, T5Gemma2PreTrainedModel
from .generation_diffusion_gemma import DiffusionGemmaGenerationConfig, DiffusionGemmaGenerationMixin


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/diffusiongemma-26B-A4B-it")
@strict
class DiffusionGemmaTextConfig(Gemma4TextConfig):
    r"""
    use_bidirectional_attention (`str`, *optional*):
        Controls bidirectional attention behavior. When set to `"vision"`, vision tokens
        attend bidirectionally while text tokens use causal attention. When set to `"all"`,
        all tokens use bidirectional attention.
    num_global_key_value_heads (`int`, *optional*):
        Number of key-value heads for global (full) attention layers. If `None`, defaults
        to `num_key_value_heads`.
    global_head_dim (`int`, defaults to 512):
        Dimension of each attention head in global (full) attention layers.
    top_k_experts (`int`, *optional*):
        Number of experts activated per token in MoE layers.
    moe_intermediate_size (`int`, *optional*):
        Intermediate (hidden) size of each expert's feed-forward network in MoE layers.
    """

    model_type = "diffusion_gemma_text"
    final_logit_softcapping = 30.0

    base_model_pp_plan = {
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    enable_moe_block = AttributeError()
    attention_k_eq_v = AttributeError()
    use_double_wide_mlp = AttributeError()
    num_kv_shared_layers = AttributeError()
    vocab_size_per_layer_input = AttributeError()
    hidden_size_per_layer_input = AttributeError()
    use_cache = AttributeError()


@auto_docstring(checkpoint="google/diffusiongemma-26B-A4B-it")
@strict
class DiffusionGemmaConfig(Gemma4Config):
    r"""
    boi_token_id (`int`, *optional*, defaults to 255999):
        The begin-of-image token index to wrap the image prompt.
    eoi_token_id (`int`, *optional*, defaults to 258882):
        The end-of-image token index to wrap the image prompt.
    canvas_length (`int`, *optional*, defaults to 256):
        The size of the canvas or, in other words, the block length in block diffusion. Used to initialize an empty
        canvas.

    Example:

    ```python
    >>> from transformers import (
    >>>     DiffusionGemmaConfig,
    >>>     DiffusionGemmaModel,
    >>>     DiffusionGemmaTextConfig,
    >>>     Gemma4VisionConfig,
    >>> )

    >>> # Initializing a DiffusionGemma Text config.
    >>> text_config = DiffusionGemmaTextConfig()

    >>> # Initializing a Gemma 4 vision config (DiffusionGemma uses Gemma 4's vision block).
    >>> vision_config = Gemma4VisionConfig()

    >>> # Initializing a DiffusionGemma text config
    >>> configuration = DiffusionGemmaConfig(text_config, vision_config)

    >>> # Initializing a model from the configuration
    >>> model = DiffusionGemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "diffusion_gemma"
    sub_configs = {
        "text_config": DiffusionGemmaTextConfig,
        "vision_config": AutoConfig,
    }

    text_config: DiffusionGemmaTextConfig | dict[str, Any] | None = None
    vision_config: PreTrainedConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    initializer_range: float | None = 0.02
    canvas_length: int | None = 256
    # Important: this model also ties the text encoder with the decoder. Setting this to `False` undoes all ties.
    tie_word_embeddings: bool = True

    audio_config = AttributeError()
    boa_token_id = AttributeError()
    eoa_token_index = AttributeError()
    video_token_id = AttributeError()
    audio_token_id = AttributeError()

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = DiffusionGemmaTextConfig()
            logger.info("text_config is None. Using default DiffusionGemmaTextConfig.")
        elif isinstance(self.text_config, dict):
            self.text_config = DiffusionGemmaTextConfig(**self.text_config)

        if self.vision_config is None:
            logger.info("vision_config is None. DiffusionGemmaEncoderModel.vision_tower will not be initialized.")
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "gemma4_vision")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)

        PreTrainedConfig.__post_init__(**kwargs)


class DiffusionGemmaTextRotaryEmbedding(Gemma4TextRotaryEmbedding):
    pass


class DiffusionGemmaRMSNorm(Gemma4RMSNorm):
    pass


class DiffusionGemmaClippableLinear(Gemma4ClippableLinear):
    def __init__(
        self,
        config: PreTrainedConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__(config, in_features, out_features)


class DiffusionGemmaEncoderTextAttention(nn.Module):
    """Attention layer for the diffusion model.

    This layer is just like `Gemma4TextAttention`, with one key differences:
    1. Removes shared KV cache logic, as it is unused in DiffusionGemma.
    """

    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int):
        super().__init__()
        self.is_causal = config.use_bidirectional_attention != "all"

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        num_key_value_heads = config.num_global_key_value_heads if not self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = (
            nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
            if self.is_sliding
            else None
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # The code in this function is adapted from Gemma4TextAttention. ** The modified parts are clearly indicated **
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CHANGED: removed `if self.is_kv_shared_layer` branch, kept the `else`
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        # CHANGED: removed the `if self.store_full_length_kv` branch

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiffusionGemmaDecoderTextAttention(nn.Module):
    """Attention layer for the diffusion model.

    This layer is just like `Gemma4TextAttention`, with three key differences:
    1. Removes shared KV cache logic, as it is unused in DiffusionGemma.
    2. It doesn't update the KV cache in the forward pass. The KV cache here corresponds to the
       encoder's KV cache, which is passed in via `past_key_values` -- from the decoder's perspective, it can be seen
       as a read-only encoder KV cache.
    3. `self.is_causal` is set to `False`. `config.use_bidirectional_attention` only controls the
       encoder, not the decoder attention.
    """

    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int):
        super().__init__()
        self.is_causal = False  # In the decoder, attention is bidirectional!

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        num_key_value_heads = config.num_global_key_value_heads if not self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = (
            nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
            if self.is_sliding
            else None
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # The code in this function is adapted from Gemma4TextAttention. ** The modified parts are clearly indicated **
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CHANGED: removed `if self.is_kv_shared_layer` branch, kept the `else`
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.append(key_states, value_states, self.layer_idx)

        # CHANGED: removed the `if self.store_full_length_kv` branch
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiffusionGemmaText4MLP(Gemma4TextMLP):
    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int):
        nn.Module.__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]


class DiffusionGemmaTextRouter(Gemma4TextRouter):
    pass


class DiffusionGemmaTextExperts(Gemma4TextExperts):
    pass


class DiffusionGemmaEncoderTextLayer(GradientCheckpointingLayer):
    """Encoder layer for the diffusion encoder.

    Identical to `Gemma4TextDecoderLayer` except that:
    1. It doesn't have the PLE code path
    2. Doesn't pipe `shared_kv_states` around
    """

    def __init__(self, config: DiffusionGemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = DiffusionGemmaEncoderTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = DiffusionGemmaText4MLP(config, layer_idx)
        self.input_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.router = DiffusionGemmaTextRouter(config)
        self.experts = DiffusionGemmaTextExperts(config)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # Take hidden states before MLP here
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # Combine mlp and moe outputs
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class DiffusionGemmaDecoderTextLayer(Gemma4TextDecoderLayer):
    """Decoder layer for the diffusion decoder.

    Identical to `Gemma4TextDecoderLayer` except that:
    1. Uses `DiffusionGemmaDecoderTextAttention`, which reads from the encoder KV cache without updating it
    2. It doesn't have the PLE code path
    3. Doesn't pipe `shared_kv_states` around
    """

    def __init__(self, config: DiffusionGemmaConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = DiffusionGemmaDecoderTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = DiffusionGemmaText4MLP(config, layer_idx)
        self.input_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.router = DiffusionGemmaTextRouter(config)
        self.experts = DiffusionGemmaTextExperts(config)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        # Take hidden states before MLP here
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        # Combine mlp and moe outputs
        hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class DiffusionGemmaTextScaledWordEmbedding(Gemma4TextScaledWordEmbedding):
    pass


class DiffusionGemmaMultimodalEmbedder(Gemma4MultimodalEmbedder):
    def __init__(
        self,
        multimodal_config: PreTrainedConfig,
        text_config: DiffusionGemmaTextConfig,
    ):
        super().__init__(multimodal_config, text_config)


class DiffusionGemmaSelfConditioning(nn.Module):
    """
    Self-conditioning module using a feed-forward block.

    Processes soft-embeddings from the previous denoising step, converted from the returned logits, into a
    self-conditioning signal that is added to the decoder's input embeddings. Uses Gemma4's Gated MLP structure,
    with pre/post rms norm.
    """

    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.pre_norm = DiffusionGemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_norm = DiffusionGemmaRMSNorm(hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, inputs_embeds, self_conditioning_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            self_conditioning_signal: Soft-embeddings from previous denoising step
                of shape `(batch_size, canvas_length, hidden_size)`.

        Returns:
            Processed self-conditioning signal, same shape.
        """
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.down_proj(self.act_fn(self.gate_proj(normed)) * self.up_proj(normed))
        combined = inputs_embeds + sc_signal
        return self.post_norm(combined)


class DiffusionGemmaPreTrainedModel(T5Gemma2PreTrainedModel):
    _no_split_modules = [
        "DiffusionGemmaDecoderTextLayer",
        "DiffusionGemmaEncoderTextLayer",
        "DiffusionGemmaVisionEncoderLayer",
    ]
    supports_gradient_checkpointing = False
    _can_record_outputs = None  # override
    _supports_flash_attn = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, DiffusionGemmaTextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_init_fn_kwargs = {"layer_type": layer_type}
                if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
                    rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

                curr_inv_freq, _ = rope_init_fn(module.config, **rope_init_fn_kwargs)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)

        elif isinstance(module, DiffusionGemmaTextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, DiffusionGemmaTextRouter):
            init.ones_(module.scale)
            init.ones_(module.per_expert_scale)
        elif isinstance(module, DiffusionGemmaTextExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DiffusionGemmaDecoderTextLayer):
            init.ones_(module.layer_scalar)
        elif isinstance(module, DiffusionGemmaClippableLinear) and module.use_clipped_linears:
            init.constant_(module.input_min, -float("inf"))
            init.constant_(module.input_max, float("inf"))
            init.constant_(module.output_min, -float("inf"))
            init.constant_(module.output_max, float("inf"))
        # Gemma4 modules' classes won't be correctly expanded with modular, so we match the class name
        # Gemma4VisionPatchEmbedder
        elif module.__class__.__name__.endswith("VisionPatchEmbedder"):
            init.ones_(module.position_embedding_table)
        # Gemma4VisionRotaryEmbedding
        elif module.__class__.__name__.endswith("VisionRotaryEmbedding"):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)
        # Gemma4VisionModel
        elif module.__class__.__name__.endswith("Gemma4VisionModel") and module.config.standardize:
            init.zeros_(module.std_bias)
            init.ones_(module.std_scale)

    def prepare_decoder_input_ids_from_labels(self, **kwargs):
        raise NotImplementedError("Diffusion Gemma doesn't uses noise-init canvas as decoder inputs")


class DiffusionGemmaEncoderTextModel(DiffusionGemmaPreTrainedModel):
    config: DiffusionGemmaTextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaEncoderTextLayer,
        "attentions": DiffusionGemmaEncoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # DiffusionGemmaEncoder downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = DiffusionGemmaTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [DiffusionGemmaEncoderTextLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # decoder layers
        for i, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The DiffusionGemma encoder model comprising a vision backbone and a language model, *without* a language modeling
    head. It is very similar to Gemma4Model, except that it doesn't support audio or video inputs, and always
    assumes the MoE code path in the inner layers.
    """
)
class DiffusionGemmaEncoderModel(DiffusionGemmaPreTrainedModel, Gemma4Model):
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaEncoderTextLayer,
        "attentions": DiffusionGemmaEncoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaConfig):
        DiffusionGemmaPreTrainedModel.__init__(config)
        self.vocab_size = config.text_config.vocab_size

        self.language_model = DiffusionGemmaEncoderTextModel(config=config.text_config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.embed_vision = DiffusionGemmaMultimodalEmbedder(config.vision_config, config.text_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.BoolTensor:
        """
        Obtains mask for multimodal placeholders (replaced by soft tokens) and hard text tokens.

        Masks will be obtained from `input_ids` or `inputs_embeds` as available and in that
        precedence order.

        Args:
            input_ids: A tensor containing the hard token IDs from the text tokenizer.
            inputs_embeds: A tensor containing the embeddings for all hard text tokens.

        Returns:
            image_mask
        """
        if input_ids is not None:
            special_image_mask = input_ids == self.config.image_token_id
        else:
            image_token_embeddings = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token_embeddings).all(-1)

        return special_image_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_mask = self.get_placeholder_mask(input_ids, inputs_embeds)

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[image_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
                f" {image_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
            )

        # It may already have been prepared by, e.g., `generate`
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            self.create_masks_for_generate(
                config=self.config.get_text_config(),
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_audio_features(self, *args, **kwargs):
        raise NotImplementedError("DiffusionGemma does not support audio inputs.")

    def get_video_features(self, *args, **kwargs):
        raise NotImplementedError("DiffusionGemma does not support video inputs.")

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> dict:
        # TODO(joao): this fn exists in a gemma4 class, but not in Gemma4Model. Move it there, and remove the modular
        # overwrite in DiffusionGemma. Also rewrite Gemma4Model to use this function.
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        # Larger Gemma 4 models use Gemma 3's bidirectional attention mask for vision inputs
        # Smaller Gemma models use a conventional casual attention mask
        if getattr(config.get_text_config(), "use_bidirectional_attention", None) == "vision":
            block_sequence_ids = torch.full([*inputs_embeds.size()[:-1]], -1, device=inputs_embeds.device)
            if mm_token_type_ids is not None:
                block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)

            mask_kwargs["block_sequence_ids"] = block_sequence_ids

        return create_masks_for_generate(**mask_kwargs)


class DiffusionGemmaDecoderModel(DiffusionGemmaPreTrainedModel):
    """
    Decoder model for DiffusionGemma.

    Processes canvas tokens with bidirectional self-attention and cross-attention to the encoder's KV cache.
    The decoder reads but does not update the KV cache. Excluding these differences, it is similar to
    `DiffusionGemmaEncoderTextModel`, and they share all weights they have in common.
    """

    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(DiffusionGemmaTextRouter, index=0),
        "hidden_states": DiffusionGemmaDecoderTextLayer,
        "attentions": DiffusionGemmaDecoderTextAttention,
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)
        self.text_config = config.text_config
        self.padding_idx = config.text_config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.embed_tokens = DiffusionGemmaTextScaledWordEmbedding(
            num_embeddings=config.text_config.vocab_size,
            embedding_dim=config.text_config.hidden_size,
            padding_idx=self.padding_idx,
            embed_scale=config.text_config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                DiffusionGemmaDecoderTextLayer(config.text_config, layer_idx)
                for layer_idx in range(config.text_config.num_hidden_layers)
            ]
        )
        self.norm = DiffusionGemmaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = DiffusionGemmaTextRotaryEmbedding(config.text_config)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config.text_config)
        self.unique_layer_types = set(config.text_config.layer_types)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`):
            Token IDs for the canvas to be refined.
        self_conditioning_logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, vocab_size)`, *optional*):
            Self-conditioning logits from the previous denoising step, used to compute the
            self-conditioning embeddings.
        self_conditioning_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
            Per-example mask over `self_conditioning_logits`: examples set to `False` get a zeroed self-conditioning
            signal, as if no logits were passed for them. Useful for training, where self-conditioning is enabled per
            example with some probability.
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
            Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            The position IDs for the tokens in the canvas.
        """
        if "use_cache" in kwargs:
            raise ValueError(
                "The decoder of DiffusionGemma always uses a cache, so it doesn't accept the `use_cache` argument"
            )

        inputs_embeds = self.embed_tokens(decoder_input_ids)

        # If no self-conditioning signal is passed, the self-conditioning embeddings should be set to zeros.
        # This corresponds to the first denoising step.
        if self_conditioning_logits is not None:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(self.embed_tokens.weight.dtype),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
            if self_conditioning_mask is not None:
                soft_embeddings = soft_embeddings * self_conditioning_mask.to(soft_embeddings.dtype)[:, None, None]
        else:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        # The decoder positions continue after the encoder sequence. These are the position ids to be used in the
        # canvas.
        if decoder_position_ids is None:
            canvas_length = inputs_embeds.shape[1]
            cache_seq_length = past_key_values.get_seq_length(layer_idx=0) if past_key_values is not None else 0
            decoder_position_ids = torch.arange(
                cache_seq_length,
                cache_seq_length + canvas_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            decoder_position_ids = decoder_position_ids.unsqueeze(0)

        if not isinstance(mask_mapping := decoder_attention_mask, dict):
            mask_mapping = self.create_diffusion_decoder_attention_mask(
                config=self.text_config,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                decoder_attention_mask=decoder_attention_mask,
            )

        # Embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, decoder_position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.text_config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[self.text_config.layer_types[i]],
                attention_mask=mask_mapping[self.text_config.layer_types[i]],
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # No past_key_values in the output: the decoder doesn't produce a KV cache
        return BaseModelOutput(last_hidden_state=hidden_states)

    @staticmethod
    def create_diffusion_decoder_attention_mask(
        config: DiffusionGemmaConfig,
        inputs_embeds: torch.Tensor,
        past_key_values: Cache,
        decoder_attention_mask: torch.Tensor | dict | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Creates the bidirectional attention mask for the decoder model.

        Args:
            config (`DiffusionGemmaConfig`):
                The config used by the model.
            inputs_embeds (`torch.Tensor` of shape `(batch_size, canvas_length, hidden_dimension)`):
                The input embeddings used in the current forward pass. Only used to obtain the first two dimensions.
            past_key_values (`Cache`):
                The cache produced by the encoder part of the model.
            decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
                Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        """

        if past_key_values is None:
            raise ValueError(
                "The diffusion mask requires `past_key_values` to construct the next attention mask correctly"
            )

        # DiT module doesn't need a sliding mask and has to attend fully to prev context and itself
        # To enforce a full mask we pass `or_mask_function`, while keeping the functionality of
        # `create_bidirectional_sliding_window_mask` to get correct the mask shape and offsets
        LAYER_TYPE_TO_MASK_MAPPING = {
            "full_attention": create_bidirectional_mask,
            "sliding_attention": create_bidirectional_sliding_window_mask,
        }
        additional_kv_length = config.canvas_length if past_key_values.is_compileable else 0
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": decoder_attention_mask,
            "past_key_values": past_key_values,
            "or_mask_function": bidirectional_mask_function,
        }
        mask_mapping = {}
        for layer_pattern in set(config.get_text_config().layer_types):
            # DiffusionGemma decoder doesn't calls `append` on cache and always expects
            # `max-len + query` length, and passes `additional_kv_length` to account for it
            mask_kwargs["additional_kv_length"] = additional_kv_length

            # `StaticSlidingLayer` concatenates new key with cache when cache is full instead
            # of rolling back. Thus the final length is `window+query-1`, not fixed-length
            # `sliding_window`. Adding another `query_length` will result on mask shape mismatch
            # see - cache_utils.py::L592-595 for more details
            if layer_pattern == "sliding_attention" and past_key_values.is_compileable:
                layer_idx = past_key_values.is_sliding.index(True)
                sliding_layer = past_key_values.layers[layer_idx]
                if sliding_layer.cumulative_length_int >= sliding_layer.max_cache_len:
                    mask_kwargs["additional_kv_length"] = 1

            mask_mapping[layer_pattern] = LAYER_TYPE_TO_MASK_MAPPING[layer_pattern](**mask_kwargs)

        return mask_mapping


@auto_docstring
@dataclass
class DiffusionGemmaModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden states at the output of the last layer of the encoder. Only set when `input_ids` is
        provided, e.g. to compute an autoregressive loss on the encoder during training.
    """

    encoder_last_hidden_state: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class DiffusionGemmaBlockDiffusionOutputWithPast(CausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Language modeling loss.
    logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden states at the output of the last layer of the encoder. Only set when `input_ids` is
        provided, e.g. to compute an autoregressive loss on the encoder during training.
    """

    encoder_last_hidden_state: torch.FloatTensor | None = None


class DiffusionGemmaModel(DiffusionGemmaPreTrainedModel, T5Gemma2Model):
    """
    DiffusionGemma model consisting of an auto-regressive encoder (DiffusionGemmaEncoderModel, very similar to a
    Gemma4Model), and a diffusion decoder (DiffusionGemmaDecoderModel).

    NOTE: contrarily to most encoder-decoder models, where the encoder feeds its hidden states to the decoder, here the
    encoder only feeds its KV cache to the decoder. From the decoder's perspective, the KV cache is read-only.
    """

    # All weights in the text part of the encoder are present in the decoder. However, only the decoder has the
    # self-conditioning layers. At the time of writing, HF code assumes only weights can be tied.
    _tied_weights_keys = {
        "encoder.language_model.norm.weight": "decoder.norm.weight",
        # The lines below are equivalent to `"encoder.language_model.layers": "decoder.layers"`, but don't tie buffers
        # (see comment above).
        r"encoder.language_model.layers\.(?:[^.]+\.)*weight": r"decoder.layers\.(?:[^.]+\.)*weight",
        r"encoder.language_model.layers\.(?:[^.]+\.)*scale": r"decoder.layers\.(?:[^.]+\.)*scale",
        r"encoder.language_model.layers\.(?:[^.]+\.)*per_expert_scale": r"decoder.layers\.(?:[^.]+\.)*per_expert_scale",
        r"encoder.language_model.layers\.(?:[^.]+\.)*gate_up_proj": r"decoder.layers\.(?:[^.]+\.)*gate_up_proj",
        r"encoder.language_model.layers\.(?:[^.]+\.)*down_proj": r"decoder.layers\.(?:[^.]+\.)*down_proj",
        "encoder.language_model.embed_tokens.weight": "decoder.embed_tokens.weight",
    }

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)
        self.encoder = DiffusionGemmaEncoderModel(config)
        self.decoder = DiffusionGemmaDecoderModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DiffusionGemmaModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Uncached token IDs for the prompt to be encoded as context for the canvas.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `dict`, *optional*):
            Mask for the input tokens.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            Token IDs for the canvas to be refined.
        self_conditioning_logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, vocab_size)`, *optional*):
            Self-conditioning logits from the previous denoising step, used to compute the
            self-conditioning embeddings.
        self_conditioning_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
            Per-example mask over `self_conditioning_logits`: examples set to `False` get a zeroed self-conditioning
            signal, as if no logits were passed for them. Useful for training, where self-conditioning is enabled per
            example with some probability.
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
            Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            The position IDs for the tokens in the canvas.
        """

        # 1: Encode new prompt tokens into the KV cache
        encoder_last_hidden_state = None
        if input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            past_key_values = encoder_outputs.past_key_values
            encoder_last_hidden_state = encoder_outputs.last_hidden_state
        elif past_key_values is None:
            raise ValueError("Either `input_ids` or `past_key_values` must be provided.")

        # 2: Run decoder with bidirectional self-attention in the canvas, and cross-attention to the KV cache.
        # In other words, the decoder attends to all tokens, KV cache and canvas, by default.

        # 2.a.: Prepare inputs for the decoder
        # If the canvas is unset, randomly sample from the vocabulary with uniform distribution
        if decoder_input_ids is None:
            decoder_input_ids = torch.randint(
                low=0,
                high=self.config.text_config.vocab_size,
                size=(input_ids.shape[0], self.config.canvas_length),
                device=self.decoder.device,
            )

        # 2.b.: Run the decoder
        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            **kwargs,
        )

        return DiffusionGemmaModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            past_key_values=past_key_values,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )


class DiffusionGemmaForBlockDiffusion(DiffusionGemmaPreTrainedModel, DiffusionGemmaGenerationMixin):
    """
    DiffusionGemma model for block diffusion. It calls `DiffusionGemmaModel` to obtains the hidden states for
    the input canvas, conditioned by a prompt KV cache. Using its LM Head and self-conditioning blocks, it converts
    those hidden states into logits to sample the next canvas, as well as the self-conditioning embeddings for the
    next block diffusion step.
    """

    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}
    generation_config_class = DiffusionGemmaGenerationConfig

    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__(config)

        self.model = DiffusionGemmaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.final_logit_softcapping = config.text_config.final_logit_softcapping

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DiffusionGemmaBlockDiffusionOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Uncached token IDs for the prompt to be encoded as context for the canvas.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)` or `dict`, *optional*):
            Mask for the input tokens.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            Token IDs for the canvas to be refined.
        self_conditioning_logits (`torch.FloatTensor` of shape `(batch_size, canvas_length, vocab_size)`, *optional*):
            Self-conditioning logits from the previous denoising step, used to compute the self-conditioning
            embeddings.
        self_conditioning_mask (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
            Per-example mask over `self_conditioning_logits`: examples set to `False` get a zeroed self-conditioning
            signal, as if no logits were passed for them. Useful for training, where self-conditioning is enabled per
            example with some probability.
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length+canvas_length)` or `dict`, *optional*):
            Attention mask for the decoder KV cache. Used to specify padded/unpopulated encoder KV cached entries.
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, canvas_length)`, *optional*):
            The position IDs for the tokens in the canvas.
        """

        # 1: Call the model
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            **kwargs,
        )

        # 2. Obtain the logits and apply logits softcapping
        logits = self.lm_head(model_outputs.last_hidden_state)
        logits = logits.to(torch.float32)
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping

        return DiffusionGemmaBlockDiffusionOutputWithPast(
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            past_key_values=model_outputs.past_key_values,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state,
        )


__all__ = [
    "DiffusionGemmaTextConfig",
    "DiffusionGemmaConfig",
    "DiffusionGemmaPreTrainedModel",
    "DiffusionGemmaModel",
    "DiffusionGemmaDecoderModel",
    "DiffusionGemmaEncoderModel",
    "DiffusionGemmaEncoderTextModel",
    "DiffusionGemmaForBlockDiffusion",
]
