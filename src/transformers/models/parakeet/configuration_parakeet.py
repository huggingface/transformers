# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Parakeet model configuration."""

from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ParakeetEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParakeetEncoder`]. It is used to instantiate a
    `ParakeetEncoder` model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the layers and the hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        conv_kernel_size (`int`, *optional*, defaults to 9):
            The kernel size of the convolution layers in the Conformer block.
        subsampling_factor (`int`, *optional*, defaults to 8):
            The factor by which the input sequence is subsampled.
        subsampling_conv_channels (`int`, *optional*, defaults to 256):
            The number of channels in the subsampling convolution layers.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel features.
        subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size of the subsampling convolution layers.
        subsampling_conv_stride (`int`, *optional*, defaults to 2):
            The stride of the subsampling convolution layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for all fully connected layers in the embeddings, encoder, and pooler.
        dropout_positions (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the positions in the input sequence.
        layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the layers in the encoder.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention layers.
        max_position_embeddings (`int`, *optional*, defaults to 5000):
            The maximum sequence length that this model might ever be used with.
        scale_input (`bool`, *optional*, defaults to `True`):
            Whether to scale the input embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
        ```python
        >>> from transformers import ParakeetEncoderModel, ParakeetEncoderConfig

        >>> # Initializing a `ParakeetEncoder` configuration
        >>> configuration = ParakeetEncoderConfig()

        >>> # Initializing a model from the configuration
        >>> model = ParakeetEncoderModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the ParakeetEncoder architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "parakeet_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_act="silu",
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        num_mel_bins=80,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.1,
        dropout_positions=0.0,
        layerdrop=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=5000,
        scale_input=True,
        initializer_range=0.02,
        use_bias=True,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads  # LlamaAttention compatibility
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        if (conv_kernel_size - 1) % 2 != 0:
            raise ValueError(f"conv_kernel_size must be odd, got {conv_kernel_size}")
        self.conv_kernel_size = conv_kernel_size

        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride

        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.num_mel_bins = num_mel_bins

        self.dropout = dropout
        self.dropout_positions = dropout_positions
        self.layerdrop = layerdrop
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.scale_input = scale_input
        self.initializer_range = initializer_range

        self.attention_bias = use_bias
        self.use_bias = use_bias


class ParakeetTDTDecoderConfig(PretrainedConfig):
    model_type = "parakeet_tdt_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        pred_hidden=640,
        pred_n_layers=2,
        joint_hidden=640,
        vocab_size=1024,
        durations=[0,1,2,3,4],
        norm=None,
        forget_gate_bias=1.0,
        pred_dropout=0.0,
        norm_first_rnn=None,
        t_max=None,
        weights_init_scale=1.0,
        hidden_hidden_bias_scale=0,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.pred_hidden = pred_hidden
        self.pred_n_layers = pred_n_layers
        self.joint_hidden = joint_hidden
        self.vocab_size = vocab_size
        self.durations = durations
        self.norm = norm
        self.forget_gate_bias=forget_gate_bias
        self.t_max=t_max
        self.pred_dropout=pred_dropout
        self.norm_first_rnn=norm_first_rnn
        self.weights_init_scale=weights_init_scale
        self.hidden_hidden_bias_scale=hidden_hidden_bias_scale

class ParakeetTDTJointConfig(PretrainedConfig):
    model_type = "parakeet_tdt_joint"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        encoder_hidden=1024,
        pred_hidden=640,
        joint_hidden=640,
        vocab_size=1024,
        durations=[0,1,2,3,4],
        norm=None,
        joint_dropout=0.0,
        joint_activation='relu',
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.encoder_hidden = encoder_hidden
        self.pred_hidden = pred_hidden
        self.joint_hidden = joint_hidden
        self.vocab_size = vocab_size
        self.durations = durations
        self.joint_dropout=joint_dropout
        self.joint_activation=joint_activation



class ParakeetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParakeetCTC`]. It is used to instantiate a
    Parakeet CTC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 1025):
            Vocabulary size of the CTC head. Defines the number of different tokens that can be predicted by the model.
        blank_token_id (`int`, *optional*, defaults to 1024):
            The id of the blank token used in CTC. Typically 0.
        pad_token_id (`int`, *optional*, defaults to 1024):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`):
            The reduction method for CTC loss. Can be "mean", "sum", or "none".
        ctc_zero_infinity (`bool`, *optional*, defaults to `True`):
            Whether to set infinite losses to zero in CTC loss computation.
        encoder_config (`Union[dict, ParakeetEncoderConfig]`, *optional*):
            Configuration for the ParakeetEncoder encoder.

    Example:
        ```python
        >>> from transformers import ParakeetForCTC, ParakeetConfig

        >>> # Initializing a Parakeet configuration
        >>> configuration = ParakeetConfig()

        >>> # Initializing a model from the configuration
        >>> model = ParakeetConfig(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```

    This configuration class is based on the Parakeet CTC architecture from NVIDIA NeMo. You can find more details
    and pre-trained models at [nvidia/parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b).
    """

    model_type = "parakeet"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder_config": ParakeetEncoderConfig}

    def __init__(
        self,
        vocab_size=1025,
        blank_token_id=1024,
        pad_token_id=1024,
        bos_token_id=1,
        eos_token_id=2,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        encoder_config: Union[dict, ParakeetEncoderConfig] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if encoder_config is None:
            encoder_config = {}
            logger.info("`encoder_config` is `None`. Initializing the `ParakeetEncoderConfig` with default values.")

        if encoder_config is None:
            encoder_config = ParakeetEncoderConfig()
        elif isinstance(encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**encoder_config)
        elif isinstance(encoder_config, ParakeetEncoderConfig):
            self.encoder_config = encoder_config
        else:
            raise ValueError(
                f"`encoder_config` must be a dictionary or an instance of `ParakeetEncoderConfig`, got {type(encoder_config)}"
            )

        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.use_bias = self.encoder_config.use_bias
        self.initializer_range = self.encoder_config.initializer_range

    @classmethod
    def from_encoder_config(cls, encoder_config: ParakeetEncoderConfig, **kwargs):
        r"""
        Instantiate a [`ParakeetConfig`] (or a derived class) from parakeet encoder model configuration.

        Returns:
            [`ParakeetConfig`]: An instance of a configuration object
        """

        return cls(encoder_config=encoder_config.to_dict(), **kwargs)


class ParakeetTDTConfig(PretrainedConfig):

    model_type = "parakeet_tdt"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder_config": ParakeetEncoderConfig, "decoder_config": ParakeetTDTDecoderConfig, "joint_config": ParakeetTDTJointConfig}

    def __init__(
        self,
        vocab_size=1025,
        blank_token_id=1024,
        pad_token_id=1024,
        bos_token_id=1,
        eos_token_id=2,
        tdt_loss_reduction="mean",
        encoder_config: Union[dict, ParakeetEncoderConfig] = None,
        decoder_config: Union[dict, ParakeetTDTDecoderConfig] = None,
        joint_config: Union[dict, ParakeetTDTJointConfig] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if encoder_config is None:
            encoder_config = {}
            logger.info("`encoder_config` is `None`. Initializing the `ParakeetEncoderConfig` with default values.")

        if encoder_config is None:
            encoder_config = ParakeetEncoderConfig()
        elif isinstance(encoder_config, dict):
            self.encoder_config = ParakeetEncoderConfig(**encoder_config)
        elif isinstance(encoder_config, ParakeetEncoderConfig):
            self.encoder_config = encoder_config
        else:
            raise ValueError(
                f"`encoder_config` must be a dictionary or an instance of `ParakeetEncoderConfig`, got {type(encoder_config)}"
            )

        if decoder_config is None:
            decoder_config = {}
            logger.info("`encoder_config` is `None`. Initializing the `ParakeetEncoderConfig` with default values.")

        if decoder_config is None:
            decoder_config = ParakeetTDTDecoderConfig()
        elif isinstance(decoder_config, dict):
            self.decoder_config = ParakeetTDTDecoderConfig(**encoder_config)
        elif isinstance(decoder_config, ParakeetTDTDecoderConfig):
            self.decoder_config = decoder_config
        else:
            raise ValueError(
                f"`decoder_config` must be a dictionary or an instance of `ParakeetEncoderConfig`, got {type(encoder_config)}"
            )

        if joint_config is None:
            joint_config = {}
            logger.info("`encoder_config` is `None`. Initializing the `ParakeetEncoderConfig` with default values.")

        if joint_config is None:
            joint_config = ParakeetTDTJointConfig()
        elif isinstance(joint_config, dict):
            self.joint_config = ParakeetTDTJointConfig(**joint_config)
        elif isinstance(joint_config, ParakeetTDTJointConfig):
            self.joint_config = joint_config
        else:
            raise ValueError(
                f"`decoder_config` must be a dictionary or an instance of `ParakeetEncoderConfig`, got {type(encoder_config)}"
            )


        self.vocab_size = vocab_size

        self.blank_token_id = blank_token_id
        self.tdt_loss_reduction = tdt_loss_reduction

        self.use_bias = self.encoder_config.use_bias
        self.initializer_range = self.encoder_config.initializer_range

    @classmethod
    def from_encoder_config(cls, encoder_config: ParakeetEncoderConfig, **kwargs):
        r"""
        Instantiate a [`ParakeetConfig`] (or a derived class) from parakeet encoder model configuration.

        Returns:
            [`ParakeetConfig`]: An instance of a configuration object
        """

        return cls(encoder_config=encoder_config.to_dict(), **kwargs)


__all__ = ["ParakeetConfig", "ParakeetTDTConfig", "ParakeetEncoderConfig", "ParakeetTDTDecoderConfig", "ParakeetTDTJointConfig"]
