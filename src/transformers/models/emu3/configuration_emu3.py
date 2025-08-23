# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class Emu3VQVAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Emu3VQVAE`]. It is used to instantiate an VQ-VAE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration to the VQ model presented in Emu3 paper.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        codebook_size (`int`, *optional*, defaults to 32768):
            Codebook size of the VQ model.
        embed_dim (`int`, *optional*, defaults to 4):
            Dimension of the quantized vector in codebook.
        latent_channels (`int`, *optional*, defaults to 4):
            Dimension of the output channel of encoder and the input channel of decoder
        double_latent (`bool`, *optional*, defaults to `False`):
            Whether double the output dim of the encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Input channel of encoder.
        out_channels (`int`, *optional*, defaults to 3):
            Output channel of decoder.
        temporal_downsample_factor (`int`, *optional*, defaults to 4):
            Temporal downsample factor.
        base_channels (`int`, *optional*, defaults to 256):
            Basic channel number of the intermediate blocks.
        channel_multiplier (`list[int]`, *optional*, defaults to `[1, 2, 2, 4]`):
            Channel scaling factor of the intermediate blocks.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Residual block number in each stage.
        attn_resolutions (`list[int]`, *optional*, defaults to `[3]`):
            Stage indices to apply attention.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations in the attention layer.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Emu3VQVAE, Emu3VQVAEConfig

    >>> # Initializing a video VQ model of Emu3 configuration
    >>> configuration = Emu3VQVAEConfig()

    >>> # Initializing a model from the Emu3 VQ model style configuration
    >>> model = Emu3VQVAE(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "emu3_vqgan"
    base_config_key = "vq_config"

    def __init__(
        self,
        codebook_size: int = 32768,
        embed_dim: int = 4,
        latent_channels: int = 4,
        double_latent: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        temporal_downsample_factor: int = 4,
        base_channels: int = 256,
        channel_multiplier: list[int] = [1, 2, 2, 4],
        num_res_blocks: int = 2,
        attn_resolutions: list[int] = [3],
        hidden_size: int = 1024,
        num_attention_heads: int = 1,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self.double_latent = double_latent
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_downsample_factor = temporal_downsample_factor
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout


class Emu3TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Emu3TextModel`]. It is used to instantiate a
    emu3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Emu3-community/Emu3-Chat-hf](https://huggingface.co/Emu3-community/Emu3-Chat-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 184622):
            Vocabulary size of the Emu3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Emu3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 9216):
            The maximum sequence length that this model might ever be used with. Emu supports up to 9216 tokens,
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 151643):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 151849):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 151850):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 1000000.0):
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
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    ```python
    >>> from transformers import Emu3Model, Emu3Config

    >>> # Initializing a Emu3-community/Emu3-Chat-hf style configuration
    >>> configuration = Emu3Config()

    >>> # Initializing a model from the Emu3-community/Emu3-Chat-hf style configuration
    >>> model = Emu3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "emu3_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 184622,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 9216,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 151643,
        bos_token_id: int = 151849,
        eos_token_id: int = 151850,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional = None,
        mlp_bias=False,
        attention_bias=False,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.mlp_bias = mlp_bias
        self.attention_bias = attention_bias
        self.initializer_range = initializer_range
        rope_config_validation(self)

        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Emu3Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Emu3Model`]. It is used to instantiate a
    emu3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [Emu3-community/Emu3-Chat-hf](https://huggingface.co/Emu3-community/Emu3-Chat-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vq_config (`Union[Dict, Emu3VQVAEConfig]`, *optional*):
            Emu3VQVAEConfig instance containing the configuration for the VQ-VAE model.
        text_config (`Union[Dict, Emu3TextConfig]``, *optional*):
            Emu3TextConfig instance containing the configuration for the language model.
        vocabulary_map (`dict`, *optional*):
            A dictionary containing the vocabulary map from the tokenizer. Used to obtain tokens from the image inputs.
    """

    model_type = "emu3"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"text_config": Emu3TextConfig, "vq_config": Emu3VQVAEConfig}

    def __init__(
        self,
        vq_config: Union[dict, Emu3VQVAEConfig] = None,
        text_config: Union[dict, Emu3TextConfig] = None,
        vocabulary_map: Optional[dict[int, int]] = None,
        **kwargs,
    ):
        if vq_config is None:
            vq_config = Emu3VQVAEConfig()
        elif isinstance(vq_config, dict):
            vq_config = Emu3VQVAEConfig(**vq_config)

        if text_config is None:
            text_config = Emu3TextConfig()
        elif isinstance(text_config, dict):
            text_config = Emu3TextConfig(**text_config)

        self.vq_config = vq_config
        self.text_config = text_config
        self.vocabulary_map = vocabulary_map
        self.image_token_id = vocabulary_map.get("<image>") if vocabulary_map is not None else None

        super().__init__(**kwargs)


__all__ = ["Emu3Config", "Emu3TextConfig", "Emu3VQVAEConfig"]
