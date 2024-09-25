# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""MSCLAP model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class MSClapTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MSClapTextModel`]. It is used to instantiate a CLAP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MSCLAP
    [ms_clapt](https://huggingface.co/microsoft/ms_clap) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        projection_dropout_prob (`float`, *optional*, defaults to `0.0`):
            The dropout rate for the projection layer.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing the model parameters.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimension of the projection head of the `MSClapTextModelWithProjection`.
        projection_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function used in projection layer
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the MSClap Text model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ClapTextModel`].

    Examples:

    ```python
    >>> from transformers import MSClapTextConfig, MSClapTextModel

    >>> # Initializing a MSCLAP text configuration
    >>> configuration = MSClapTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MSClapTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "msclap_text_model"

    def __init__(
        self,
        hidden_size=768,
        projection_dropout_prob=0,
        initializer_factor=1.0,
        projection_dim=768,
        projection_hidden_act="gelu",
        num_hidden_layers=None,
        num_attention_heads=None,
        vocab_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if "text_encoder" not in kwargs:
            text_encoder_config = {"model_type": "gpt2"}
        else:
            text_encoder_config = kwargs.pop("text_encoder")

        self.text_model_config = AutoConfig.for_model(**text_encoder_config)

        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.projection_dropout_prob = projection_dropout_prob
        self.initializer_factor = initializer_factor
        self.projection_hidden_act = projection_hidden_act
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers else self.text_model_config.n_layer
        self.num_attention_heads = num_attention_heads if num_attention_heads else self.text_model_config.n_head
        self.vocab_size = vocab_size if vocab_size else self.text_model_config.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from MSClapConfig
        if config_dict.get("model_type") == "msclap":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        **kwargs,
    ):
        return cls(
            text_encoder=text_encoder_config.to_dict(),
            **kwargs,
        )


# Adapted from transformers.models.clap.configuration_clap.ClapAudioConfig with Clap->MSClap, CLAP->MSCLAP, laion/clap-htsat-fused->microsoft/ms_clap
class MSClapAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MSClapAudioModel`]. It is used to instantiate a
    MSCLAP audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the MSCLAP
    [microsoft/ms_clap](https://huggingface.co/microsoft/ms_clap) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        window_size (`int`, *optional*, defaults to 8):
            Image size of the spectrogram
        num_mel_bins (`int`, *optional*, defaults to 64):
            Number of mel features used per frames. Should correspond to the value used in the `ClapProcessor` class.
        spec_size (`int`, *optional*, defaults to 256):
            Desired input size of the spectrogram that the model supports. It can be different from the output of the
            `ClapFeatureExtractor`, in which case the input features will be resized. Corresponds to the `image_size`
            of the audio models.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        patch_size (`int`, *optional*, defaults to 4):
            Patch size for the audio spectrogram
        patch_stride (`list`, *optional*, defaults to `[4, 4]`):
            Patch stride for the audio spectrogram
        num_classes (`int`, *optional*, defaults to 527):
            Number of classes used for the head training
        hidden_size (`int`, *optional*, defaults to 768):
            Hidden size of the output of the audio encoder. Correspond to the dimension of the penultimate layer's
            output,which is sent to the projection MLP layer.
        projection_dim (`int`, *optional*, defaults to 1024):
            Hidden size of the projection layer.
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            Depths used for the Swin Layers of the audio model
        num_attention_heads (`list`, *optional*, defaults to `[4, 8, 16, 32]`):
            Number of attention heads used for the Swin Layers of the audio model
        enable_fusion (`bool`, *optional*, defaults to `False`):
            Whether or not to enable patch fusion. This is the main contribution of the authors, and should give the
            best results.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder.
        fusion_type (`[type]`, *optional*):
            Fusion type used for the patch fusion.
        patch_embed_input_channels (`int`, *optional*, defaults to 1):
            Number of channels used for the input spectrogram
        flatten_patch_embeds (`bool`, *optional*, defaults to `True`):
            Whether or not to flatten the patch embeddings
        patch_embeds_hidden_size (`int`, *optional*, defaults to 96):
            Hidden size of the patch embeddings. It is used as the number of output channels.
        enable_patch_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether or not to enable layer normalization for the patch embeddings
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Drop path rate for the patch fusion
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        projection_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout rate for the projection layer.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to add a bias to the query, key, value projections.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the mlp hidden dim to embedding dim.
        aff_block_r (`int`, *optional*, defaults to 4):
            downsize_ratio used in the AudioFF block
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        projection_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        layer_norm_eps (`[type]`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import MSClapAudioConfig, MSClapAudioModel

    >>> # Initializing a MSClapAudioConfig with microsoft/ms_clap style configuration
    >>> configuration = MSClapAudioConfig()

    >>> # Initializing a MSClapAudioModel (with random weights) from the microsoft/ms_clap style configuration
    >>> model = MSClapAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "msclap_audio_model"

    def __init__(
        self,
        window_size=8,
        num_mel_bins=64,
        spec_size=256,
        hidden_act="gelu",
        patch_size=4,
        patch_stride=[4, 4],
        num_classes=527,
        hidden_size=768,
        projection_dim=1024,
        depths=[2, 2, 6, 2],
        num_attention_heads=[4, 8, 16, 32],
        enable_fusion=False,
        hidden_dropout_prob=0.1,
        fusion_type=None,
        patch_embed_input_channels=1,
        flatten_patch_embeds=True,
        patch_embeds_hidden_size=96,
        enable_patch_layer_norm=True,
        drop_path_rate=0.0,
        attention_probs_dropout_prob=0.0,
        projection_dropout_prob=0.1,
        qkv_bias=True,
        mlp_ratio=4.0,
        aff_block_r=4,
        num_hidden_layers=4,
        projection_hidden_act="gelu",
        layer_norm_eps=1e-5,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_mel_bins = num_mel_bins
        self.spec_size = spec_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.depths = depths
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.projection_dim = projection_dim
        self.flatten_patch_embeds = flatten_patch_embeds
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        self.enable_patch_layer_norm = enable_patch_layer_norm
        self.drop_path_rate = drop_path_rate
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.projection_dropout_prob = projection_dropout_prob
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.patch_embed_input_channels = patch_embed_input_channels
        self.aff_block_r = aff_block_r
        self.layer_norm_eps = layer_norm_eps
        self.initializer_factor = initializer_factor
        self.projection_hidden_act = projection_hidden_act

    @classmethod
    # Copied from transformers.models.clap.configuration_clap.ClapAudioConfig.from_pretrained with clap->msclap
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the audio config dict if we are loading from ClapConfig
        if config_dict.get("model_type") == "msclap":
            config_dict = config_dict["audio_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MSClapConfig(PretrainedConfig):
    r"""
    [`MSClapConfig`] is the configuration class to store the configuration of a [`MSClapModel`]. It is used to instantiate
    a MSCLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MS
    [microsoft/ms_clap](https://huggingface.co/microsoft/ms_clap) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MSClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MSClapAudioConfig`].
        logit_scale_init_value (`float`, *optional*, defaults to 14.29):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLAP implementation.
        projection_dim (`int`, *optional*, defaults to 1024):
            Dimensionality of text and audio projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import MSClapConfig, MSClapModel

    >>> # Initializing a MSClapConfig with microsoft/ms_clap style configuration
    >>> configuration = MSClapConfig()

    >>> # Initializing a MSClapModel (with random weights) from the microsoft/ms_clap style configuration
    >>> model = MSClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a MSClapConfig from a MSClapTextConfig and a MSClapAudioConfig
    >>> from transformers import MSClapTextConfig, MSClapAudioConfig

    >>> # Initializing a MSClapText and MSClapAudioConfig configuration
    >>> config_text = MSClapTextConfig()
    >>> config_audio = MSClapAudioConfig()

    >>> config = MSClapConfig.from_text_audio_configs(config_text, config_audio)
    ```"""

    model_type = "msclap"

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        logit_scale_init_value=(1 / 0.07),
        projection_dim=1024,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the MSClapTextConfig with default values.")

        if audio_config is None:
            audio_config = {}
            logger.info("audio_config is None. initializing the MSClapAudioConfig with default values.")

        self.text_config = MSClapTextConfig(**text_config)
        self.audio_config = MSClapAudioConfig(**audio_config)
        self.text_config.projection_dim = projection_dim
        self.audio_config.projection_dim = projection_dim

        self.projection_dim = projection_dim
        self.hidden_size = self.text_config.hidden_size

        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor
        self.num_hidden_layers = self.audio_config.num_hidden_layers + len(self.audio_config.depths)

    @classmethod
    # Copied from transformers.models.clap.configuration_clap.ClapConfig.from_text_audio_configs with clap->msclap, Clap->MSClap
    def from_text_audio_configs(cls, text_config: MSClapTextConfig, audio_config: MSClapAudioConfig, **kwargs):
        r"""
        Instantiate a [`MSClapConfig`] (or a derived class) from msclap text model configuration and msclap audio model
        configuration.

        Returns:
            [`MSClapConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)
