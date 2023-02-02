# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" CLAP model configuration"""

import copy
import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CLAP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "laion-ai/base": "https://huggingface.co/laion-ai/base/resolve/main/config.json",
}


class CLAPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLAPTextModel`] or a [`TFCLAPTextModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    [roberta-base](https://huggingface.co/roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the RoBERTa model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CLAPTextModel`] or [`TFCLAPTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`,
            `"relu"`, `"silu"` and `"relu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`CLAPTextModel`] or [`TFCLAPTextModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import CLAPTextConfig, CLAPTextModel

    >>> # Initializing a RoBERTa configuration
    >>> configuration = CLAPTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CLAPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "roberta"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        fusion_hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        projection_hidden_size=768,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.projection_hidden_size = projection_hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from CLAPConfig
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLAPAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLAPVisionModel`]. It is used to instantiate a
    CLAP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLAP
    [laion-ai/base](https://huggingface.co/laion-ai/base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`,
            `"relu"`, `"selu"` and `"relu_new"` ``"relu"` are supported. layer_norm_eps (`float`, *optional*, defaults
            to 1e-5): The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import CLAPAudioConfig, CLAPVisionModel

    >>> # Initializing a CLAPAudioConfig with laion-ai/base style configuration
    >>> configuration = CLAPAudioConfig()

    >>> # Initializing a CLAPVisionModel (with random weights) from the laion-ai/base style configuration
    >>> model = CLAPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clap_vision_model"

    def __init__(
        self,
        sample_rate=48000,
        audio_length=1024,
        window_size=8,
        hop_size=1024,
        fmin=50,
        fmax=14000,
        mel_bins=64,
        clip_samples=480000,
        spec_size=256,
        hidden_act="relu",
        patch_size=4,
        patch_stride=(4, 4),
        num_classes=527,
        hidden_size=96,
        projection_hidden_size=768,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        enable_fusion=False,
        hidden_dropout_prob=0.1,
        fusion_type=None,
        image_size=224,
        input_channels=3,
        patch_embed_input_channels=1,
        flatten_patch_embeds=True,
        patch_embeds_hidden_size=96,
        enable_patch_layer_norm=True,
        swin_drop_rate=0.0,
        swin_attention_drop_rate=0.0,
        swin_drop_path_rate=0.1,
        swin_qkv_bias=True,
        swin_norm_before_mlp="ln",
        swin_mlp_ratio=4.0,
        swin_use_checkpoint=False,
        swin_absolute_positional_embedding=False,
        swin_hidden_act="gelu",
        aff_block_r=4,
        enable_patch_fusion=False,
        spectrogram_window_size=1024,
        spectrogram_window='hann',
        spectrogram_center=True,
        spectrogram_pad_mode='reflect',
        spectrogram_freeze_parameters=True,
        spectrogram_ref=1.0,
        spectrogram_amin=1e-10,
        spectrogram_top_db=None,
        spectrogram_time_drop_width=64, 
        spectrogram_time_stripes_num=2, 
        spectrogram_freq_drop_width=8, 
        spectrogram_freq_stripes_num=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.mel_bins = mel_bins
        self.clip_samples = clip_samples
        self.spec_size = spec_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.projection_hidden_size = projection_hidden_size
        self.image_size = image_size
        self.input_channels = input_channels
        self.flatten_patch_embeds = flatten_patch_embeds
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        self.enable_patch_layer_norm = enable_patch_layer_norm
        self.swin_drop_rate = swin_drop_rate
        self.swin_attention_drop_rate = swin_attention_drop_rate
        self.swin_drop_path_rate = swin_drop_path_rate
        self.swin_qkv_bias = swin_qkv_bias
        self.swin_norm_before_mlp = swin_norm_before_mlp
        self.swin_mlp_ratio = swin_mlp_ratio
        self.swin_use_checkpoint = swin_use_checkpoint
        self.swin_absolute_positional_embedding = swin_absolute_positional_embedding
        self.patch_embed_input_channels = patch_embed_input_channels
        self.swin_hidden_act = swin_hidden_act
        self.aff_block_r = aff_block_r
        self.enable_patch_fusion = enable_patch_fusion
        self.spectrogram_window_size = spectrogram_window_size
        self.spectrogram_window = spectrogram_window
        self.spectrogram_center = spectrogram_center
        self.spectrogram_pad_mode = spectrogram_pad_mode
        self.spectrogram_freeze_parameters = spectrogram_freeze_parameters
        self.spectrogram_ref = spectrogram_ref
        self.spectrogram_amin = spectrogram_amin
        self.spectrogram_top_db = spectrogram_top_db
        self.spectrogram_time_drop_width = spectrogram_time_drop_width
        self.spectrogram_time_stripes_num = spectrogram_time_stripes_num
        self.spectrogram_freq_drop_width = spectrogram_freq_drop_width
        self.spectrogram_freq_stripes_num = spectrogram_freq_stripes_num

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLAPConfig
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLAPConfig(PretrainedConfig):
    r"""
    [`CLAPConfig`] is the configuration class to store the configuration of a [`CLAPModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion-ai/base](https://huggingface.co/laion-ai/base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLAPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLAPAudioConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLAP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CLAPConfig, CLAPModel

    >>> # Initializing a CLAPConfig with laion-ai/base style configuration
    >>> configuration = CLAPConfig()

    >>> # Initializing a CLAPModel (with random weights) from the laion-ai/base style configuration
    >>> model = CLAPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLAPConfig from a CLAPTextConfig and a CLAPAudioConfig
    >>> from transformers import CLAPTextConfig, CLAPAudioConfig

    >>> # Initializing a CLAPText and CLAPVision configuration
    >>> config_text = CLAPTextConfig()
    >>> config_vision = CLAPAudioConfig()

    >>> config = CLAPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "clap"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        logit_scale_init_value=(1 / 0.07),
        fusion_num_hidden_layers=2,
        projection_dim=512,
        projection_hidden_act="relu",
        **kwargs
    ):
        super().__init__(**kwargs)

        # If `_config_dict` exist, we use them for the backward compatibility.
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        if text_config_dict is not None:
            text_config = text_config_dict
        if vision_config_dict is not None:
            vision_config = vision_config_dict

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the CLAPTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the CLAPAudioConfig with default values.")

        self.text_config = CLAPTextConfig(**text_config)
        self.vision_config = CLAPAudioConfig(**vision_config)

        self.text_config.fusion_num_hidden_layers = fusion_num_hidden_layers
        self.vision_config.fusion_num_hidden_layers = fusion_num_hidden_layers

        self.text_config.projection_dim = projection_dim
        self.vision_config.projection_dim = projection_dim

        self.text_config.projection_hidden_act = projection_hidden_act
        self.vision_config.projection_hidden_act = projection_hidden_act

        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        self.hidden_size = self.text_config.hidden_size

        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: CLAPTextConfig, vision_config: CLAPAudioConfig, **kwargs):
        r"""
        Instantiate a [`CLAPConfig`] (or a derived class) from clap text model configuration and clap vision model
        configuration.

        Returns:
            [`CLAPConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
