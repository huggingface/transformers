# coding=utf-8
# Copyright 2025 The Nari Labs and HuggingFace Inc. team. All rights reserved.
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
"""Dia model configuration"""

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class DiaEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DiaEncoder`]. It is used to instantiate a Dia
    encoder according to the specified arguments, defining the encoder architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key and value heads for each attention layer in the Transformer encoder.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of the attention head.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the Dia model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DiaModel`].
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"swish"` and `"gelu_new"` are supported.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "dia_encoder"

    def __init__(
        self,
        max_position_embeddings: int = 1024,
        num_hidden_layers: int = 12,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 128,
        intermediate_size: int = 4096,
        norm_eps: float = 1e-5,
        vocab_size: int = 256,
        hidden_act: str = "silu",
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class DiaDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DiaDecoder`]. It is used to instantiate a Dia
    decoder according to the specified arguments, defining the decoder architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        max_position_embeddings (`int`, *optional*, defaults to 3072):
            The maximum sequence length that this model might ever be used with.
        num_hidden_layers (`int`, *optional*, defaults to 18):
            Number of hidden layers in the Transformer decoder.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the decoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key and value heads for each attention layer in the Transformer decoder.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of the attention head.
        cross_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each cross-attention layer in the Transformer decoder.
        cross_head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of the cross-attention head.
        cross_num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key and value heads for each cross-attention layer in the Transformer decoder.
        cross_hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the cross-attention layers.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        vocab_size (`int`, *optional*, defaults to 1028):
            Vocabulary size of the Dia model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DiaModel`].
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder. If string, `"gelu"`, `"relu"`,
            `"swish"` and `"gelu_new"` are supported.
        num_channels (`int`, *optional*, defaults to 9):
            Number of channels for the Dia decoder.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Indicating that this model is part of an encoder-decoder architecture.
    """

    model_type = "dia_decoder"

    def __init__(
        self,
        max_position_embeddings: int = 3072,
        num_hidden_layers: int = 18,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        cross_num_attention_heads: int = 16,
        cross_head_dim: int = 128,
        cross_num_key_value_heads: int = 16,
        cross_hidden_size: int = 1024,
        norm_eps: float = 1e-5,
        vocab_size: int = 1028,
        hidden_act: str = "silu",
        num_channels: int = 9,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        is_encoder_decoder: bool = True,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.cross_num_key_value_heads = cross_num_key_value_heads
        self.cross_num_attention_heads = cross_num_attention_heads
        self.cross_head_dim = cross_head_dim
        self.cross_hidden_size = cross_hidden_size
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.num_channels = num_channels
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


class DiaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DiaModel`]. It is used to instantiate a
    Dia model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_config (`DiaEncoderConfig`, *optional*):
            Configuration for the encoder part of the model. If not provided, a default `DiaEncoderConfig` will be used.
        decoder_config (`DiaDecoderConfig`, *optional*):
            Configuration for the decoder part of the model. If not provided, a default `DiaDecoderConfig` will be used.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Indicating that this model uses an encoder-decoder architecture.
        pad_token_id (`int`, *optional*, defaults to 1025):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1024):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 1026):
            Beginning of stream token id.
        delay_pattern (`list[int]`, *optional*, defaults to `[0, 8, 9, 10, 11, 12, 13, 14, 15]`):
            The delay pattern for the decoder. The length of this list must match `decoder_config.num_channels`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import DiaConfig, DiaModel

    >>> # Initializing a DiaConfig with default values
    >>> configuration = DiaConfig()

    >>> # Initializing a DiaModel (with random weights) from the configuration
    >>> model = DiaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dia"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder_config": DiaEncoderConfig, "decoder_config": DiaDecoderConfig}

    def __init__(
        self,
        encoder_config: Optional[DiaEncoderConfig] = None,
        decoder_config: Optional[DiaDecoderConfig] = None,
        norm_eps: float = 1e-5,
        is_encoder_decoder: bool = True,
        pad_token_id: int = 1025,
        eos_token_id: int = 1024,
        bos_token_id: int = 1026,
        delay_pattern: Optional[list[int]] = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config = DiaEncoderConfig(**encoder_config)
        if isinstance(decoder_config, dict):
            decoder_config = DiaDecoderConfig(**decoder_config)
        self.encoder_config = encoder_config if encoder_config is not None else DiaEncoderConfig()
        self.decoder_config = decoder_config if decoder_config is not None else DiaDecoderConfig()
        self.norm_eps = norm_eps
        self.delay_pattern = delay_pattern if delay_pattern is not None else [0, 8, 9, 10, 11, 12, 13, 14, 15]
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        assert self.decoder_config.num_channels == len(self.delay_pattern), (
            "Number of channels must match delay pattern length."
        )

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

    def get_text_config(self, *args, **kwargs):
        """Defaulting to audio config as it's the decoder in this case which is usually the text backbone"""
        return self.decoder_config


__all__ = ["DiaConfig", "DiaEncoderConfig", "DiaDecoderConfig"]
