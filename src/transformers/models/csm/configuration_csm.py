# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class CsmDepthDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CsmDepthDecoderModel`]. It is used to instantiate an CSM depth decoder
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the csm-1b.

    e.g. [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_codebooks (`int`, *optional*, defaults to 32):
            Number of codebooks used in the underlying codec model responsible for tokenizing the audio.
        backbone_hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations of the backbone model used with this depth decoder.
        vocab_size (`int`, *optional*, defaults to 2051):
            Vocabulary size of the CsmDepthDecoder model. Defines the number of different audio tokens that can be represented by each codebook.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 33):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 2050):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        rope_theta (`float`, *optional*, defaults to 500000):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
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
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads

    ```python
    >>> from transformers import CsmDepthDecoder, CsmDepthDecoderConfig

    >>> # Initializing a CsmDepthDecoder
    >>> configuration = CsmDepthDecoderConfig()
    >>> model = CsmDepthDecoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "csm_depth_decoder_model"
    base_config_key = "depth_decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_codebooks=32,
        backbone_hidden_size=2048,
        vocab_size=2051,
        hidden_size=1024,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=33,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        rope_theta=500000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("`tie_word_embeddings=True` is not supported for CsmDepthDecoderConfig")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.backbone_hidden_size = backbone_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)


class CsmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CsmForConditionalGeneration`]. It is used to instantiate an CSM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the csm-1b.

    e.g. [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_codebooks (`int`, *optional*, defaults to 32):
            Number of codebooks used in the underlying codec model responsible for tokenizing the audio.
        vocab_size (`int`, *optional*, defaults to 2051):
            Vocabulary size of the Csm model. Defines the number of different audio tokens that can be represented by each codebook.
        text_vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the text input for the Csm model. Defines the number of different text tokens that can be represented.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations of the backbone model.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations of the backbone model.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the backbone model Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the backbone model Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the backbone model Transformer decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 128002):
            Padding token id.
        codebook_pad_token_id (`int`, *optional*, defaults to 2050):
            Padding token id for codebook tokens.
        codebook_eos_token_id (`int`, *optional*, defaults to 0):
            End of stream token id for codebook tokens.
        bos_token_id (`int`, *optional*, defaults to 128000):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        audio_token_id (`int`, *optional*, defaults to 128002):
            Audio token id in the text input.
        audio_eos_token_id (`int`, *optional*, defaults to 128003):
            End of stream token id for audio in the text input.
        rope_theta (`float`, *optional*, defaults to 500000):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*, defaults to `{'factor': 32.0, 'high_freq_factor': 0.5, 'low_freq_factor': 0.125, 'original_max_position_embeddings': 1024, 'rope_type': 'llama3'}`):
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
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads
        tie_codebooks_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the codebook tokens embeddings of the backbone model to the codebook tokens embeddings of the depth decoder.
        depth_decoder_config (`CsmDepthDecoderConfig`, *optional*):
            Configuration for the depth decoder.
        codec_config (`PretrainedConfig`, *optional*):
            Configuration for the codec.

    ```python
    >>> from transformers import CsmForConditionalGeneration, CsmConfig

    >>> # Initializing a CsmConfig
    >>> configuration = CsmConfig()

    >>> # Initializing a model
    >>> model = CsmForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "csm"
    base_config_key = "csm_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "codec_config": AutoConfig,
        "depth_decoder_config": CsmDepthDecoderConfig,
    }

    def __init__(
        self,
        num_codebooks=32,
        vocab_size=2051,
        text_vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=128002,
        codebook_pad_token_id=2050,
        codebook_eos_token_id=0,
        bos_token_id=128000,
        eos_token_id=None,
        audio_token_id=128002,
        audio_eos_token_id=128003,
        rope_theta=500000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        tie_codebooks_embeddings=True,
        depth_decoder_config=None,
        codec_config=None,
        **kwargs,
    ):
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("`tie_word_embeddings=True` is not supported for CsmConfig")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )

        if depth_decoder_config is None:
            self.depth_decoder_config = CsmDepthDecoderConfig()
            logger.info("depth_decoder_config is None, using default depth decoder config.")
        elif isinstance(depth_decoder_config, dict):
            self.depth_decoder_config = CsmDepthDecoderConfig(**depth_decoder_config)
        elif isinstance(depth_decoder_config, CsmDepthDecoderConfig):
            self.depth_decoder_config = depth_decoder_config

        if codec_config is None:
            self.codec_config = AutoConfig.for_model("mimi")
            logger.info("codec_config is None, using default audio encoder config.")
        elif isinstance(codec_config, dict):
            self.codec_config = AutoConfig.for_model(**codec_config)
        elif isinstance(codec_config, PretrainedConfig):
            self.codec_config = codec_config

        self.text_vocab_size = text_vocab_size
        self.num_codebooks = num_codebooks
        self.audio_token_id = audio_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.codebook_pad_token_id = codebook_pad_token_id
        self.codebook_eos_token_id = codebook_eos_token_id
        self.tie_codebooks_embeddings = tie_codebooks_embeddings

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)


__all__ = [
    "CsmDepthDecoderConfig",
    "CsmConfig",
]
