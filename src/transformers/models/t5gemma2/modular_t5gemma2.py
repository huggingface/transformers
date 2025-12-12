# coding=utf-8
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
import copy
from collections.abc import Callable
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from ... import initialization as init
from ...cache_utils import DynamicCache, EncoderDecoderCache, StaticCache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...generation import GenerationConfig, GenerationMixin, GenerationMode
from ...masking_utils import create_bidirectional_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import OutputRecorder, check_model_inputs
from ..auto import AutoModel
from ..gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig
from ..gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3MLP,
    Gemma3MultiModalProjector,
    Gemma3PreTrainedModel,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
    apply_rotary_pos_emb,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
)
from ..siglip import SiglipVisionConfig
from ..t5gemma.modeling_t5gemma import (
    T5GemmaClassificationHead,
    T5GemmaEncoderLayer,
    T5GemmaLMHead,
    bidirectional_mask_function,
)


logger = logging.get_logger(__name__)


class T5Gemma2TextConfig(Gemma3TextConfig, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5Gemma2TextModel`]. It is used to instantiate the encoder's
    text model portion of the T5Gemma2 Model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the T5Gemma2Text-7B.
    e.g. [google/t5gemma2_text-7b](https://huggingface.co/google/t5gemma2_text-7b)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 262208):
            Vocabulary size of the T5Gemma2Text model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Gemma2TextModel`]
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
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256):
            Scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096):
            In T5Gemma2Text, every other layer uses sliding window attention. This is the size of the sliding window.
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
    """

    model_type = "t5gemma2_text"

    def __init__(
        self,
        vocab_size: Optional[int] = 262_208,
        hidden_size: Optional[int] = 2304,
        intermediate_size: Optional[int] = 9216,
        num_hidden_layers: Optional[int] = 26,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 4,
        head_dim: Optional[int] = 256,
        hidden_activation: Optional[str] = "gelu_pytorch_tanh",
        max_position_embeddings: Optional[int] = 131_072,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = True,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        query_pre_attn_scalar: Optional[int] = 256,
        sliding_window: Optional[int] = 4096,
        layer_types: Optional[list[str]] = None,
        final_logit_softcapping: Optional[float] = None,
        attn_logit_softcapping: Optional[float] = None,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        **kwargs,
    ):
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

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        PreTrainedConfig.__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class T5Gemma2EncoderConfig(Gemma3Config):
    model_type = "t5gemma2_encoder"

    sub_configs = {
        "text_config": T5Gemma2TextConfig,
        "vision_config": SiglipVisionConfig,
    }


class T5Gemma2DecoderConfig(Gemma3TextConfig, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5Gemma2DecoderModel`]. It is used to instantiate the decoder
    text model portion of the T5Gemma2 Model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the T5Gemma2Decoder-7B.
    e.g. [google/t5gemma2_text-7b](https://huggingface.co/google/t5gemma2_text-7b)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 262208):
            Vocabulary size of the T5Gemma2Decoder model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Gemma2DecoderModel`]
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
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256):
            Scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096):
            In T5Gemma2Decoder, every other layer uses sliding window attention. This is the size of the sliding window.
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
    """

    model_type = "t5gemma2_decoder"

    def __init__(
        self,
        vocab_size: Optional[int] = 262_208,
        hidden_size: Optional[int] = 2304,
        intermediate_size: Optional[int] = 9216,
        num_hidden_layers: Optional[int] = 26,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 4,
        head_dim: Optional[int] = 256,
        hidden_activation: Optional[str] = "gelu_pytorch_tanh",
        max_position_embeddings: Optional[int] = 131_072,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = True,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        query_pre_attn_scalar: Optional[int] = 256,
        sliding_window: Optional[int] = 4096,
        layer_types: Optional[list[str]] = None,
        final_logit_softcapping: Optional[float] = None,
        attn_logit_softcapping: Optional[float] = None,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        **kwargs,
    ):
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

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        PreTrainedConfig.__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class T5Gemma2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5Gemma2Model`]. It is used to instantiate an T5Gemma2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to a hypothetical balanced Gemma3 encoder-decoder model.
    e.g. [google/t5gemma-2-270m-270m](https://huggingface.co/google/t5gemma-2-270m-270m)
    Configuration objects inherit from [PreTrainedConfig] and can be used to control the model outputs. Read the
    documentation from [PreTrainedConfig] for more information.

    Args:
        encoder (`Union[T5Gemma2EncoderConfig, dict]`, optional, *optional*):
            Configuration for the encoder.
        decoder (`Union[T5Gemma2DecoderConfig, dict]`, optional, *optional*):
            Configuration for the decoder.
        is_encoder_decoder (bool, optional, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The ratio for all dropout layers (following T5).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention.
        classifier_dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier (following T5).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        image_token_index (`int`, *optional*, defaults to 256001):
            The image token index to encode the image prompt. Defaults to 256001, which is right after the eoi_token_index.
            Note this is different from Gemma 3.
    ```python
    >>> from transformers import T5Gemma2Config, T5Gemma2Model
    >>> t5gemma2_config = T5Gemma2Config.from_pretrained("google/t5gemma-270m-270m")
    >>> model = T5Gemma2Model(t5gemma2_config)
    ```
    """

    model_type = "t5gemma2"
    keys_to_ignore_at_inference = ["past_key_values"]

    sub_configs = {
        "encoder": T5Gemma2EncoderConfig,
        "decoder": T5Gemma2DecoderConfig,
    }

    attribute_map = {
        "image_token_id": "image_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    def __init__(
        self,
        encoder: Optional[Union[T5Gemma2EncoderConfig, dict[str, Any]]] = None,
        decoder: Optional[Union[T5Gemma2DecoderConfig, dict[str, Any]]] = None,
        is_encoder_decoder: bool = True,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        classifier_dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        image_token_index: int = 256_001,
        **kwargs,
    ):
        if isinstance(encoder, dict):
            encoder = T5Gemma2EncoderConfig(**encoder)
        elif encoder is None:
            encoder = T5Gemma2EncoderConfig()
            logger.info("encoder is None, using default T5Gemma2EncoderConfig encoder config.")
        else:
            if not isinstance(encoder, T5Gemma2EncoderConfig):
                raise ValueError(f"{type(encoder)} is not supported.")

        if isinstance(decoder, dict):
            decoder = T5Gemma2DecoderConfig(**decoder)
        elif decoder is None:
            decoder = T5Gemma2DecoderConfig()
            logger.info("decoder is None, using default T5Gemma2DecoderConfig decoder config.")
        else:
            if not isinstance(decoder, T5Gemma2DecoderConfig):
                raise ValueError(f"{type(decoder)} is not supported.")

        if encoder.text_config.hidden_size != decoder.hidden_size:
            raise ValueError(
                "Imbalanced encoder-decoder is not supported in T5Gemma2: "
                f"encoder ({encoder.text_config.hidden_size}) vs decoder ({decoder.hidden_size})."
            )

        if not is_encoder_decoder:
            raise ValueError("T5Gemma2Model only support encoder-decoder modeling.")

        if encoder.text_config.vocab_size != decoder.vocab_size:
            raise ValueError(
                "Imbalanced encoder-decoder vocabulary size is not supported in T5Gemma2: "
                f"encoder ({encoder.text_config.vocab_size}) vs decoder ({decoder.vocab_size})."
            )

        # Encoder.
        encoder.text_config.dropout_rate = dropout_rate
        encoder.text_config.attention_dropout = attention_dropout
        encoder.vision_config.attention_dropout = attention_dropout
        encoder.image_token_index = image_token_index
        self.encoder = encoder

        # Decoder.
        decoder.dropout_rate = dropout_rate
        decoder.attention_dropout = attention_dropout
        self.decoder = decoder

        for special_token_key in ["bos_token_id", "pad_token_id", "eos_token_id", "vocab_size"]:
            if special_token_key not in kwargs:
                kwargs[special_token_key] = getattr(decoder, special_token_key)

        super().__init__(**kwargs)

        self.is_encoder_decoder = is_encoder_decoder
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.classifier_dropout_rate = classifier_dropout_rate
        self.initializer_range = initializer_range
        self.eoi_token_index = encoder.eoi_token_index
        self.image_token_index = image_token_index

    def __setattr__(self, key, value):
        shared_attr_with_submodules = [
            "output_hidden_states",
            "output_attentions",
            "_attn_implementation_internal",
            "dropout_rate",
            "attention_dropout",
            "vocab_size",
            "dtype",
        ]

        if key in shared_attr_with_submodules:
            setattr(self.encoder.text_config, key, value)
            setattr(self.encoder.vision_config, key, value)
            setattr(self.decoder, key, value)
            setattr(self.encoder, key, value)
        super().__setattr__(key, value)


class T5Gemma2RMSNorm(Gemma3RMSNorm):
    pass


class T5Gemma2MLP(Gemma3MLP):
    def __init__(self, config: T5Gemma2TextConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        down_proj = self.down_proj(hidden_states)
        return down_proj


class T5Gemma2RotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: T5Gemma2TextConfig, device=None):
        super().__init__(config, device)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[T5Gemma2TextConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
    ) -> tuple["torch.Tensor", float]:
        return super().compute_default_rope_parameters(config, device, seq_len, layer_type)


class T5Gemma2SelfAttention(Gemma3Attention):
    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False  # Only used by the encoder


class T5Gemma2MergedAttention(Gemma3Attention):
    """Merged self-attention and cross-attention for decoder."""

    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False  # Fused causal and encoder mask

    def forward(
        self,
        # decoder self-attention inputs
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        merged_attention_mask: Optional[torch.Tensor],
        # cross-attention inputs
        encoder_hidden_states: torch.Tensor,
        # cache inputs
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # others
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # attention shapes.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cross_input_shape = encoder_hidden_states.shape[:-1]
        cross_hidden_shape = (*cross_input_shape, -1, self.head_dim)

        # self-attention.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # self-attention.
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            self_attention_cache = past_key_values.self_attention_cache
            key_states, value_states = self_attention_cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            # cross-attention.
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            cross_attention_cache = past_key_values.cross_attention_cache

        if past_key_values is None or not is_updated:
            cross_key_states = self.k_proj(encoder_hidden_states).view(cross_hidden_shape).transpose(1, 2)
            cross_value_states = self.v_proj(encoder_hidden_states).view(cross_hidden_shape).transpose(1, 2)

            cross_key_states = self.k_norm(cross_key_states)

            if past_key_values is not None:
                cross_key_states, cross_value_states = cross_attention_cache.update(
                    cross_key_states, cross_value_states, self.layer_idx
                )
                past_key_values.is_updated[self.layer_idx] = True
        else:
            cross_key_states = cross_attention_cache.layers[self.layer_idx].keys
            cross_value_states = cross_attention_cache.layers[self.layer_idx].values

        # merged attention.
        query_states = query_states
        cross_key_size = cross_input_shape[1]
        key_states = torch.cat([key_states, cross_key_states], dim=2)
        value_states = torch.cat([value_states, cross_value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            merged_attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # decompose merged attention weights into self & cross attention weights
        if attn_weights is not None:
            self_attn_weights = attn_weights[..., :-cross_key_size]
            cross_attn_weights = attn_weights[..., -cross_key_size:]
        else:
            self_attn_weights, cross_attn_weights = None, None
        return attn_output, self_attn_weights, cross_attn_weights


def sliding_window_mask_function(sliding_window: int, is_causal=True) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        if is_causal:
            left_window_size, right_window_size = sliding_window, 0
        else:
            left_window_size, right_window_size = ((sliding_window + 1) // 2, (sliding_window) // 2 + 1)

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


class T5Gemma2EncoderLayer(T5GemmaEncoderLayer):
    pass


class T5Gemma2DecoderLayer(T5GemmaEncoderLayer):
    """Decoder sub-layer: merged attention instead of vanilla self-attention."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # replace vanilla self-attention with merged attention to support joint cross-attention.
        self.self_attn = T5Gemma2MergedAttention(
            config=config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        merged_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)

        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            merged_attention_mask=merged_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class T5Gemma2LMHead(T5GemmaLMHead):
    pass


class T5Gemma2ClassificationHead(T5GemmaClassificationHead):
    pass


class T5Gemma2MultiModalProjector(Gemma3MultiModalProjector):
    def __init__(self, config: T5Gemma2EncoderConfig):
        super().__init__(config)


class T5Gemma2TextScaledWordEmbedding(Gemma3TextScaledWordEmbedding):
    """T5Gemma2 Embedding: override to add eoi token embedding separately."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
        eoi_token_index: int = 256_000,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx, embed_scale)
        self.eoi_token_index = eoi_token_index
        self.eoi_embedding = nn.Parameter(torch.zeros(self.embedding_dim))

    def forward(self, input_ids: torch.Tensor):
        input_embeddings = super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
        input_embeddings[input_ids == self.eoi_token_index] = self.eoi_embedding.to(input_embeddings.dtype)
        return input_embeddings


@auto_docstring
class T5Gemma2PreTrainedModel(Gemma3PreTrainedModel):
    config: T5Gemma2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    # Mask creation is incompatible
    # FA due to non-default creation / SWA
    _supports_flash_attn = False
    # Flex due to custom masks not compatible to be merged after creation
    _supports_flex_attn = False

    _no_split_modules = [
        "T5Gemma2EncoderLayer",
        "T5Gemma2DecoderLayer",
        "SiglipVisionEmbeddings",
        "SiglipEncoderLayer",
        "SiglipMultiheadAttentionPoolingHead",
    ]
    _can_record_outputs = {
        "hidden_states": [T5Gemma2EncoderLayer, T5Gemma2DecoderLayer],
        "attentions": [
            OutputRecorder(T5Gemma2SelfAttention, index=1, layer_name="self_attn"),
            OutputRecorder(T5Gemma2MergedAttention, index=1, layer_name="self_attn"),
            OutputRecorder(T5Gemma2MergedAttention, index=2, layer_name="cross_attn"),
        ],
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, T5Gemma2MultiModalProjector):
            init.zeros_(module.mm_input_projection_weight)
        elif isinstance(module, T5Gemma2TextScaledWordEmbedding):
            init.zeros_(module.eoi_embedding)
        elif isinstance(module, T5Gemma2ClassificationHead):
            scale = module.out_proj.weight.shape[0] ** -0.5
            init.normal_(module.out_proj.weight, mean=0.0, std=self.config.initializer_range * scale)
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                init.zeros_(module.out_proj.bias)
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif "RMSNorm" in module.__class__.__name__:
            init.zeros_(module.weight)

    def prepare_decoder_input_ids_from_labels(self, input_ids):
        """
        Shifts input_ids to the right, prepends the decoder_start_token_id, and handles
        pad_token_id replacement for labels that were -100.
        This is a common preparation step for decoder inputs in sequence-to-sequence models.
        """
        decoder_config = self.config.decoder
        decoder_start_token_id = decoder_config.bos_token_id
        pad_token_id = decoder_config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError("self.model.config.decoder.bos_token_id has to be defined. ")

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.decoder.pad_token_id has to be defined.")

        # Is this T5 specific?
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class T5Gemma2Encoder(T5Gemma2PreTrainedModel):
    config: T5Gemma2EncoderConfig
    _can_record_outputs = {
        "attentions": T5Gemma2SelfAttention,
        "hidden_states": T5Gemma2EncoderLayer,
    }

    def __init__(
        self,
        config: T5Gemma2EncoderConfig,
        eoi_token_index: int = 256_000,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        vision_config = config.vision_config
        text_config = config.text_config

        # setup vision tower
        self.vision_tower = AutoModel.from_config(config=vision_config)
        self.multi_modal_projector = T5Gemma2MultiModalProjector(config)

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            self.padding_idx,
            embed_scale=text_config.hidden_size**0.5,
            eoi_token_index=eoi_token_index,
        )
        self.norm = T5Gemma2RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [T5Gemma2EncoderLayer(text_config, layer_idx) for layer_idx in range(text_config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(text_config.dropout_rate)
        self.rotary_emb = T5Gemma2RotaryEmbedding(text_config)

        self.text_config = text_config

        # Initialize weights and apply final processing
        self.post_init()

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert pixel image to image features via the encoder and projector."""
        # pixel_values: (batch_size, channels, height, width)
        # image_features: Image feature tensor of shape (num_images, image_length, embed_dim).
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def get_image_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        image_features: torch.FloatTensor,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        image_token_id = self.config.image_token_id
        if input_ids is None:
            if inputs_embeds is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` has to be provided.")
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def preprocess_image_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Convert pixel images to image features and merge into input embeds."""
        image_features = self.get_image_features(pixel_values)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

        image_mask = self.get_image_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        return inputs_embeds

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        # Unused for processor compatibility kept in signature.
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        del token_type_ids
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # As we want to pass `past_key_values=None` explicitly everywhere, we need to pop them from kwargs if present
        kwargs.pop("past_key_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            inputs_embeds = self.preprocess_image_features(
                pixel_values, input_ids=input_ids, inputs_embeds=inputs_embeds
            )

        if position_ids is None:
            position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            self_attn_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_mask(
                    **mask_kwargs,
                    and_mask_function=sliding_window_mask_function(self.text_config.sliding_window, is_causal=False),
                ),
            }

        # input layer
        hidden_states = inputs_embeds

        # global and local position embeddings
        position_embeddings = {}
        for layer_type in self.text_config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # dropout
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers[: self.text_config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings[layer_module.attention_type],
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class T5Gemma2Decoder(T5Gemma2PreTrainedModel):
    config: T5Gemma2DecoderConfig
    _can_record_outputs = {
        "attentions": OutputRecorder(T5Gemma2MergedAttention, index=1),
        "cross_attentions": OutputRecorder(T5Gemma2MergedAttention, index=2),
        "hidden_states": T5Gemma2DecoderLayer,
    }

    def __init__(self, config: T5Gemma2DecoderConfig, eoi_token_index: int = 256_000):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=config.hidden_size**0.5,
            eoi_token_index=eoi_token_index,
        )
        self.norm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [T5Gemma2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.rotary_emb = T5Gemma2RotaryEmbedding(config)
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states` must be given in decoder")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache())

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values.self_attention_cache if past_key_values is not None else None,
                "position_ids": position_ids,
            }
            # this masking function did nothing to masking but forces `allow_is_causal_skip` to be False
            # as we always need a mask during decoding for merged attention.
            mask_kwargs["and_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
            self_attn_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        if not isinstance(cross_attn_mask_mapping := encoder_attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": encoder_hidden_states,
                "attention_mask": encoder_attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": None,
            }
            cross_attn_mask_mapping = {
                "full_attention": create_causal_mask(
                    **mask_kwargs,
                    or_mask_function=bidirectional_mask_function(encoder_attention_mask),
                ),
            }

        merged_attn_mask_mapping = {
            "full_attention": torch.cat(
                [self_attn_mask_mapping["full_attention"], cross_attn_mask_mapping["full_attention"]], dim=-1
            ),
            "sliding_attention": torch.cat(
                [self_attn_mask_mapping["sliding_attention"], cross_attn_mask_mapping["full_attention"]], dim=-1
            ),
        }

        # input layer
        hidden_states = inputs_embeds

        # global and local position embeddings
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # dropout
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings[layer_module.attention_type],
                merged_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                encoder_hidden_states,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class T5Gemma2Model(T5Gemma2PreTrainedModel):
    _tied_weights_keys = {
        "decoder.embed_tokens.weight": "encoder.embed_tokens.weight",
        "decoder.embed_tokens.eoi_embedding": "encoder.embed_tokens.eoi_embedding",
    }

    def __init__(self, config: T5Gemma2Config):
        super().__init__(config)

        # setup encoder and decoder
        self.encoder = T5Gemma2Encoder(config.encoder, config.eoi_token_index)
        self.decoder = T5Gemma2Decoder(config.decoder, config.eoi_token_index)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        # encoder inputs
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # decoder inputs
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        # others (mainly inference or cache related)
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        """
        # encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                return_dict=True,
                **kwargs,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        # decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class T5Gemma2ForConditionalGeneration(T5Gemma2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {
        "lm_head.out_proj.weight": "model.encoder.embed_tokens.weight",
    }
    _tp_plan = {"lm_head.out_proj": "colwise_rep"}
    _pp_plan = {"lm_head.out_proj": (["hidden_states"], ["logits"])}

    def __init__(self, config: T5Gemma2Config):
        super().__init__(config)

        self.model = T5Gemma2Model(config)
        self.vocab_size = config.decoder.vocab_size
        self.lm_head = T5Gemma2LMHead(config.decoder.hidden_size, self.vocab_size)
        self.loss_type = "ForMaskedLM"

        self.post_init()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.out_proj = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head.out_proj

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values):
        return self.get_encoder().get_image_features(pixel_values)

    @property
    def vision_tower(self):
        return self.get_encoder().vision_tower

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        # encoder inputs
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # decoder inputs
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        # others (mainly inference or cache related)
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)

        decoder_outputs: Seq2SeqModelOutput = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = decoder_outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        decoder_config = self.config.decoder
        if decoder_config.final_logit_softcapping is not None:
            logits = logits / decoder_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * decoder_config.final_logit_softcapping

        loss = None
        if labels is not None:
            # Input has right-shifted so we directly perform masked lm loss
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
            encoder_hidden_states=decoder_outputs.encoder_hidden_states,
            encoder_attentions=decoder_outputs.encoder_attentions,
        )

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        generation_mode: GenerationMode,
        batch_size: int,
        max_cache_length: int,
    ) -> bool:
        """Override cache preparation to support T5Gemma2-specific EncoderDecoder Cache."""

        # Build cache and past_key_values structure first and then override as needed.
        super()._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        )

        # If use_cache is False, do not prepare the cache.
        if generation_config.use_cache is False:
            return

        cache_implementation = generation_config.cache_implementation
        if cache_implementation is None:
            offload_cache = False
        else:
            offload_cache = "offloaded" in generation_config.cache_implementation

        # Main change: use full cache for cross-attention.
        cross_attn_config = copy.deepcopy(self.config.get_text_config(decoder=True))

        # cross-attention does not use sliding window
        del cross_attn_config.sliding_window
        del cross_attn_config.layer_types

        cross_attn_cache_kwargs = {
            "config": cross_attn_config,
            "offloading": offload_cache,
        }

        past_key_values = model_kwargs.get("past_key_values")
        if past_key_values is not None:
            if not isinstance(past_key_values, EncoderDecoderCache):
                raise ValueError(
                    "The `past_key_values` in `model_kwargs` must be of type `EncoderDecoderCache` for T5Gemma2 model."
                )

            # Cache already established, no need to re-initialize.
            if len(past_key_values.is_updated) > 0 and past_key_values.is_updated.get(0):
                return

            cross_attn_cls = type(past_key_values.cross_attention_cache)
            if cross_attn_cls == StaticCache:
                cross_attn_cache_kwargs["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
            # Update cross-attention cache only (switch from sliding_window to full).
            past_key_values.cross_attention_cache = cross_attn_cls(**cross_attn_cache_kwargs)
        else:
            # Initialize new cache.
            model_kwargs["past_key_values"] = EncoderDecoderCache(
                DynamicCache(
                    **{
                        "config": self.config.get_text_config(decoder=True),
                        "offloading": offload_cache,
                    }
                ),  # self-attention cache
                DynamicCache(),  # cross-attention cache
            )

        if hasattr(self, "_cache") and self._cache is not None:
            if not isinstance(self._cache, EncoderDecoderCache):
                raise ValueError("The internal cache must be of type `EncoderDecoderCache` for T5Gemma2 model.")

            self._cache = model_kwargs["past_key_values"]


@auto_docstring
class T5Gemma2ForSequenceClassification(T5Gemma2PreTrainedModel):
    def __init__(self, config: T5Gemma2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.decoder.hidden_size

        self.model = T5Gemma2Model(config)

        classifier_dropout = getattr(config, "classifier_dropout_rate", 0.1)
        self.score = T5Gemma2ClassificationHead(self.hidden_size, self.num_labels, classifier_dropout)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if inputs_embeds is not None or decoder_inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}."
            )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        if decoder_input_ids is None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(input_ids)

        outputs: Seq2SeqModelOutput = self.model(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=False,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        hidden_states = outputs.decoder_hidden_states
        attentions = outputs.decoder_attentions

        logits = self.score(last_hidden_state)

        batch_size = input_ids.shape[0]
        # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
        non_pad_mask = (decoder_input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
        token_indices = torch.arange(decoder_input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        last_non_pad_token = torch.clamp(last_non_pad_token, max=decoder_input_ids.shape[-1] - 1)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


@auto_docstring
class T5Gemma2ForTokenClassification(T5Gemma2PreTrainedModel):
    def __init__(self, config: T5Gemma2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.decoder.hidden_size

        self.model = T5Gemma2Model(config)

        classifier_dropout = getattr(config, "classifier_dropout_rate", 0.1)
        self.score = T5Gemma2ClassificationHead(self.hidden_size, self.num_labels, classifier_dropout)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if inputs_embeds is not None or decoder_inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}."
            )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        if decoder_input_ids is None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(input_ids)

        outputs: Seq2SeqModelOutput = self.model(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=False,
            **kwargs,
        )
        last_hidden_state = outputs.last_hidden_state
        hidden_states = outputs.decoder_hidden_states
        attentions = outputs.decoder_attentions

        logits = self.score(last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = [
    "T5Gemma2Config",
    "T5Gemma2TextConfig",
    "T5Gemma2EncoderConfig",
    "T5Gemma2DecoderConfig",
    "T5Gemma2ForConditionalGeneration",
    "T5Gemma2Model",
    "T5Gemma2PreTrainedModel",
    "T5Gemma2ForSequenceClassification",
    "T5Gemma2ForTokenClassification",
]
