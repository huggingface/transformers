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
"""ImageBind model configuration"""

import copy
import os
from typing import Any, Dict, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ImageBindTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ImageBindTextModel`]. It is used to instantiate a ImageBind
    text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the text encoder of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the ImageBind text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`ImageBindModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden size in the feedforward network to the hidden size in the encoder layers.
        projection_dim (`int`, *optional*, defaults to 1024):
            If the ImageBind text model has an output projection layer, the dimension to which that projection layer
            maps to.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        add_kv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to 14.2857):
            The initial value of the `logit_scale` parameter for the vision component. If `None`, the logits will not
            be scaled.
        learnable_logit_scale (`bool`, *optional*, defaults to `True`):
            Whether the `logit_scale` is learnable or fixed.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:

    ```python
    >>> from transformers import ImageBindTextConfig, ImageBindTextModel

    >>> # Initializing a ImageBindTextConfig with facebook/imagebind-huge style configuration
    >>> configuration = ImageBindTextConfig()

    >>> # Initializing a ImageBindTextModel (with random weights) from the facebook/imagebind-huge style configuration
    >>> model = ImageBindTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "imagebind_text_model"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=1024,
        mlp_ratio=4.0,
        projection_dim=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        add_kv_bias=False,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        logit_scale_init_value=14.2857,
        learnable_logit_scale=True,
        pad_token_id=0,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.add_kv_bias = add_kv_bias
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.logit_scale_init_value = logit_scale_init_value
        self.learnable_logit_scale = learnable_logit_scale

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from ImageBindConfig
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ImageBindVisionModel`]. It is used to instantiate a
    ImageBind vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden size in the feedforward network to the hidden size in the encoder layers.
        projection_dim (`int`, *optional*, defaults to 1024):
            If the ImageBind vision model has an output projection layer, the dimension to which that projection layer
            maps to.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the input images.
        num_frames (`int`, *optional*, defaults to 2):
            If using video (spatiotemporal) input, the number of video frames in the spatiotemporal data.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        add_kv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*):
            The initial value of the `logit_scale` parameter for the vision component. If `None`, the logits will not
            be scaled.
        learnable_logit_scale (`bool`, *optional*, defaults to `False`):
            Whether the `logit_scale` is learnable or fixed.

    Example:

    ```python
    >>> from transformers import ImageBindVisionConfig, ImageBindVisionModel

    >>> # Initializing a ImageBindVisionConfig with facebook/imagebind-huge style configuration
    >>> configuration = ImageBindVisionConfig()

    >>> # Initializing a ImageBindVisionModel (with random weights) from the facebook/imagebind-huge style configuration
    >>> model = ImageBindVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "imagebind_vision_model"

    def __init__(
        self,
        hidden_size=1280,
        mlp_ratio=4.0,
        projection_dim=1024,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        num_frames=2,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        add_kv_bias=False,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        logit_scale_init_value=None,
        learnable_logit_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.add_kv_bias = add_kv_bias
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.logit_scale_init_value = logit_scale_init_value
        self.learnable_logit_scale = learnable_logit_scale

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from ImageBindConfig
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ImageBindAudioModel`]. It is used to instantiate a
    ImageBind audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden size in the feedforward network to the hidden size in the encoder layers.
        projection_dim (`int`, *optional*, defaults to 1024):
            If the ImageBind audio model has an output projection layer, the dimension to which that projection layer
            maps to.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_mel_bins (`int`, *optional*, defaults to 128):
            The number of frequency bins in the log-mel spectrogram.
        target_len (`int`, *optional*, defaults to 204):
            The length of the target sequence.
        num_channels (`int`, *optional*, defaults to 1):
            The number of channels in the input audio data.
        patch_size (`int`, *optional*, defaults to 16):
            The kernel size of the patch embedding 2D convolution layer.
        stride (`int`, *optional*, defaults to 10):
            The stride of the patch embedding 2D convolution layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        add_kv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to 20.0):
            The initial value of the `logit_scale` parameter for the vision component. If `None`, the logits will not
            be scaled.
        learnable_logit_scale (`bool`, *optional*, defaults to `False`):
            Whether the `logit_scale` is learnable or fixed.

    Example:
    ```python
    >>> from transformers import ImageBindAudioConfig, ImageBindAudioModel

    >>> # Initializing a ImageBindAudioConfig with facebook/imagebind-huge style configuration
    >>> configuration = ImageBindAudioConfig()

    >>> # Initializing a ImageBindAudioModel (with random weights) from the facebook/imagebind-huge style configuration
    >>> model = ImageBindAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        hidden_size=768,
        mlp_ratio=4.0,
        projection_dim=1024,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_mel_bins=128,
        target_len=204,
        num_channels=1,
        patch_size=16,
        stride=10,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        add_kv_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.1,
        initializer_range=0.02,
        initializer_factor=1.0,
        logit_scale_init_value=20.0,
        learnable_logit_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_mel_bins = num_mel_bins
        self.target_len = target_len
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.add_kv_bias = add_kv_bias
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.logit_scale_init_value = logit_scale_init_value
        self.learnable_logit_scale = learnable_logit_scale

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the audio config dict if we are loading from ImageBindConfig
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["audio_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindConfig(PretrainedConfig):
    r"""
    [`ImageBindConfig`] is the configuration class to store the configuration of a [`ImageBindModel`]. It is used to instantiate
    a ImageBind model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict` or `ImageBindTextConfig`, *optional*):
            Dictionary or an instance of `ImageBindTextConfig` that defines the text model configuration.
        vision_config (`dict` or `ImageBindVisionConfig`, *optional*):
            Dictionary or an instance of `ImageBindVisionConfig` that defines the vision model configuration.
        audio_config (`dict` or `ImageBindAudioConfig`, *optional*):
            Dictionary or an instance of `ImageBindAudioConfig` that defines the audio model configuration.
        projection_dim (`int`, *optional*, defaults to 1024):
            Dimentionality of text and vision projection layers.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ImageBindConfig, ImageBindModel

    >>> # Initializing a ImageBindConfig with facebook/imagebind-huge style configuration
    >>> configuration = ImageBindConfig()

    >>> # Initializing a ImageBindModel (with random weights) from the facebook/imagebind-huge style configuration
    >>> model = ImageBindModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ImageBindConfig from a ImageBindTextConfig and a ImageBindVisionConfig
    >>> from transformers import ImageBindTextConfig, ImageBindVisionConfig

    >>> # Initializing a ImageBindText and ImageBindVision configuration
    >>> config_text = ImageBindTextConfig()
    >>> config_vision = ImageBindVisionConfig()

    >>> config = ImageBindConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "imagebind"
    is_composition = True

    def __init__(
        self,
        text_config: Optional[Union[Dict[str, Any], ImageBindTextConfig]] = None,
        vision_config: Optional[Union[Dict[str, Any], ImageBindVisionConfig]] = None,
        audio_config: Optional[Union[Dict[str, Any], ImageBindAudioConfig]] = None,
        projection_dim: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `ImageBindTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `ImageBindVisionConfig` with default values.")

        if audio_config is None:
            audio_config = {}
            logger.info("`audio_config` is `None`. initializing the `ImageBindAudioConfig` with default values.")

        self.text_config = ImageBindTextConfig(**text_config) if isinstance(text_config, dict) else text_config
        self.vision_config = (
            ImageBindVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        )
        self.audio_config = ImageBindAudioConfig(**audio_config) if isinstance(audio_config, dict) else audio_config

        self.projection_dim = projection_dim
        self.initializer_factor = 1.0

    @classmethod
    # Copied from transformers.models.clip.configuration_clip.CLIPConfig.from_text_vision_configs with CLIP->ImageBind, clip->imagebind
    def from_text_vision_configs(
        cls, text_config: ImageBindTextConfig, vision_config: ImageBindVisionConfig, **kwargs
    ):
        r"""
        Instantiate a [`ImageBindConfig`] (or a derived class) from imagebind text model configuration and imagebind vision model
        configuration.

        Returns:
            [`ImageBindConfig`]: An instance of a configuration object
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
        output["audio_config"] = self.audio_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
