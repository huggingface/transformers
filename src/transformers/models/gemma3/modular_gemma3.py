# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from typing import Any, Optional

import torch
import torch.nn as nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...masking_utils import create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
from ...modeling_layers import GenericForSequenceClassification, GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, SequenceClassifierOutputWithPast
from ...modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    RopeParameters,
    dynamic_rope_update,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import maybe_autocast
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2ForCausalLM,
    Gemma2MLP,
    Gemma2Model,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..paligemma.modeling_paligemma import (
    PaliGemmaCausalLMOutputWithPast,
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
    PaligemmaModelOutputWithPast,
    token_type_ids_mask_function,
)
from ..siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)


class Gemma3TextConfig(Gemma2Config, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3TextModel`]. It is used to instantiate an Gemma3Text
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma3Text-7B.
    e.g. [google/gemma3_text-7b](https://huggingface.co/google/gemma3_text-7b)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 262208):
            Vocabulary size of the Gemma3Text model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Gemma3TextModel`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
            if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256):
            Scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096):
            In Gemma3Text, every other layer uses sliding window attention. This is the size of the sliding window.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        final_logit_softcapping (`float`, *optional*):
            Scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*):
            Scaling factor when applying tanh softcapping on the attention scores.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        use_bidirectional_attention (`bool`, *optional*, defaults to `False`):
            If True, the model will attend to all text tokens instead of using a causal mask. This does not change
            behavior for vision tokens.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings

    ```python
    >>> from transformers import Gemma3TextModel, Gemma3TextConfig
    >>> # Initializing a Gemma3Text gemma3_text-7b style configuration
    >>> configuration = Gemma3TextConfig()
    >>> # Initializing a model from the gemma3_text-7b style configuration
    >>> model = Gemma3TextModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "gemma3_text"
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __init__(
        self,
        vocab_size: int | None = 262_208,
        hidden_size: int | None = 2304,
        intermediate_size: int | None = 9216,
        num_hidden_layers: int | None = 26,
        num_attention_heads: int | None = 8,
        num_key_value_heads: int | None = 4,
        head_dim: int | None = 256,
        hidden_activation: str | None = "gelu_pytorch_tanh",
        max_position_embeddings: int | None = 131_072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = 0,
        eos_token_id: int | None = 1,
        bos_token_id: int | None = 2,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        query_pre_attn_scalar: int | None = 256,
        sliding_window: int | None = 4096,
        layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = None,
        attn_logit_softcapping: float | None = None,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        use_bidirectional_attention: bool | None = False,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.layer_types = layer_types

        self.use_bidirectional_attention = use_bidirectional_attention
        if use_bidirectional_attention:
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        PreTrainedConfig.__init__(**kwargs)

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`. If we find `rope_parameters`
        # as arg in the inputs, we can safely assume that it is in the new format. New naming used -> new format
        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs


class Gemma3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3ForConditionalGeneration`]. It is used to instantiate an
    Gemma3ForConditionalGeneration according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PaliGemma-2B.

    e.g. [google/gemma-3-4b](https://huggingface.co/google/gemma-3-4b)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[Gemma3TextConfig, dict]`, *optional*):
            The config object of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom vision config or dict.
        mm_tokens_per_image (`int`, *optional*, defaults to 256):
            The number of tokens per image embedding.
        boi_token_index (`int`, *optional*, defaults to 255999):
            The begin-of-image token index to wrap the image prompt.
        eoi_token_index (`int`, *optional*, defaults to 256000):
            The end-of-image token index to wrap the image prompt.
        image_token_index (`int`, *optional*, defaults to 262144):
            The image token index to encode the image prompt.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings

    Example:

    ```python
    >>> from transformers import Gemma3ForConditionalGeneration, Gemma3Config, SiglipVisionConfig, Gemma3TextConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a Gemma3 Text config
    >>> text_config = Gemma3TextConfig()

    >>> # Initializing a Gemma3 gemma-3-4b style configuration
    >>> configuration = Gemma3Config(vision_config, text_config)

    >>> # Initializing a model from the gemma-3-4b style configuration
    >>> model = Gemma3TextConfig(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma3"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
    sub_configs = {
        "text_config": Gemma3TextConfig,
        "vision_config": SiglipVisionConfig,
    }

    def __init__(
        self,
        text_config: Gemma3TextConfig | dict[str, Any] | None = None,
        vision_config: SiglipVisionConfig | dict[str, Any] | None = None,
        mm_tokens_per_image: int | None = 256,
        boi_token_index: int | None = 255_999,
        eoi_token_index: int | None = 256_000,
        image_token_index: int | None = 262_144,
        initializer_range: float | None = 0.02,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Gemma3TextConfig()
            logger.info("text_config is None, using default Gemma3TextConfig text config.")
        elif isinstance(text_config, dict):
            text_config = Gemma3TextConfig(**text_config)

        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = SiglipVisionConfig()
            logger.info("vision_config is None, using default SiglipVisionConfig vision config.")

        self.text_config = text_config
        self.vision_config = vision_config
        self.mm_tokens_per_image = mm_tokens_per_image
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


class Gemma3ModelOutputWithPast(PaligemmaModelOutputWithPast):
    pass


class Gemma3CausalLMOutputWithPast(PaliGemmaCausalLMOutputWithPast):
    pass


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(Gemma2MLP):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)


class Gemma3RMSNorm(Gemma2RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim=dim, eps=eps)


class Gemma3RotaryEmbedding(Gemma2RotaryEmbedding):
    def __init__(self, config: Gemma3TextConfig, device=None, layer_type=None):
        nn.Module.__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.layer_types = list(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Gemma3TextConfig | None = None,
        device: Optional["torch.device"] = None,
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
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Weird way to inherit but otherwise the sliding window gets defined first and can't access `is_sliding`
class Gemma3Attention(Gemma2Attention):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_causal = not self.config.use_bidirectional_attention

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


GEMMA3_START_DOCSTRING = None


class Gemma3PreTrainedModel(Gemma2PreTrainedModel):
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    _no_split_modules = [
        "Gemma3DecoderLayer",
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
        "SiglipMultiheadAttentionPoolingHead",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Gemma3MultiModalProjector):
            init.zeros_(module.mm_input_projection_weight)
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif "RMSNorm" in module.__class__.__name__:
            init.zeros_(module.weight)
        elif isinstance(module, Gemma3TextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, Gemma3RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


def _bidirectional_window_overlay(sliding_window: int) -> Callable[[int, int, int, int], bool]:
    """
    Enables a bidirectional mask within the sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        """A token can attend to any other token if their absolute distance is within
        the (exclusive) sliding window size (distance < sliding_window)."""
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


class Gemma3TextModel(Gemma2Model):
    config: Gemma3TextConfig
    input_modalities = ("text",)

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if self.config.use_bidirectional_attention:
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Gemma3ForCausalLM(Gemma2ForCausalLM):
    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, hidden_size = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, hidden_size, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)


def create_causal_mask_mapping(
    config: PreTrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    token_type_ids: torch.Tensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    is_training: bool = False,
    is_first_iteration: bool | None = None,
    **kwargs,
) -> dict:
    """
    Overwrites the base `create_masks_for_generate` with `token_type_ids` masking to create the causal mask mapping
    for all kinds of forward passes. Gemma3 uses a bidirectional mask for images.

    Uses `pixel_values` as an optional input to disambiguate edge cases.
    """
    if is_training and token_type_ids is None:
        raise ValueError("`token_type_ids` is required as a model input when training")

    mask_kwargs = {
        "config": config.get_text_config(),
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    # NOTE: this `may_have_image_input` logic is not flawless, it fails when we're using a cache eagerly initialized
    # (e.g. compiled prefill) AND `pixel_values` are not provided (i.e. the image data is provided through other
    # means). Determining prefill in that case requires checking data values, which is not compile-compatible.
    is_first_iteration = (
        is_first_iteration
        if is_first_iteration is not None
        else (past_key_values is None or not past_key_values.is_initialized or pixel_values is not None)
    )
    if token_type_ids is not None and is_first_iteration:
        # We need to pass an additional mask function to account for token type ids, and it needs to be an `or` (to
        # undo the causal masking)

        # First find where a new image block starts: 1 if image and previous not image
        # The images cannot attend to future images, but can attend to all prev images and to itself bidirectionally
        is_image = (token_type_ids == 1).to(cache_position.device)
        is_previous_image = nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
        new_image_start = is_image & ~is_previous_image
        image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
        image_group_ids = torch.where(is_image, image_group_ids, -1)
        mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
            token_type_ids.to(cache_position.device), image_group_ids
        )

    return create_masks_for_generate(**mask_kwargs)


class Gemma3Model(PaliGemmaModel):
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        del self.text_config_dtype

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the vision model into language model space.")
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True, **kwargs)
        last_hidden_state = vision_outputs.last_hidden_state
        vision_outputs.pooler_output = self.multi_modal_projector(last_hidden_state)

        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **lm_kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Gemma3ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = create_causal_mask_mapping(
                self.config,
                inputs_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
                token_type_ids,
                pixel_values,
                is_training=self.training,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class Gemma3ForConditionalGeneration(PaliGemmaForConditionalGeneration):
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    # Fix: https://github.com/huggingface/transformers/issues/40564
    accepts_loss_kwargs = False

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **lm_kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Gemma3CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
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
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            cache_position=cache_position,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        return Gemma3CausalLMOutputWithPast(
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
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Pixel values are used only in the first iteration if available
        # In subsquent iterations, they are already merged with text and cached
        # NOTE: first iteration doesn't have to be prefill, it can be the first
        # iteration with a question and cached system prompt (continue generate from cache). NOTE: use_cache=False needs pixel_values always
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


class Gemma3ForSequenceClassification(Gemma3PreTrainedModel):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
    }

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3Model(config)
        self.score = nn.Linear(config.text_config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.text_config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.text_config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.text_config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Gemma3TextForSequenceClassification(GenericForSequenceClassification, Gemma3PreTrainedModel):
    """
    Gemma3TextForSequenceClassification is a text-only sequence classification model that works with Gemma3TextConfig.
    It uses the generic sequence classification implementation for efficiency and consistency.
    """

    config: Gemma3TextConfig
    input_modalities = ("text",)


__all__ = [
    "Gemma3Config",
    "Gemma3TextConfig",
    "Gemma3PreTrainedModel",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3Model",
    "Gemma3ForSequenceClassification",
    "Gemma3TextForSequenceClassification",
]
