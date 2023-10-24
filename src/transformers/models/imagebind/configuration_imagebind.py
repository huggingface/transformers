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
""" ImageBind model configuration"""


import copy
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union


if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

IMAGEBIND_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/imagebind-huge": "https://huggingface.co/facebook/imagebind-huge/resolve/main/config.json",
}


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
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
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
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        add_kv_bias(`bool`, *optional*, defaults to `False`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to `14.2857`):
            The initial value of the `logit_scale` parameter for the vision component. If `None`, the logits will not
            be scaled.
        learnable_logit_scale (`bool`, *optional*, defaults to `True`):
            Whether the `logit_scale` is learnable or fixed.

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
        intermediate_size=4096,
        projection_dim=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-6,
        add_kv_bias=False,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        logit_scale_init_value=14.2857,
        learnable_logit_scale=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
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
        patch_size (`int` or `Tuple[int]`, *optional*, defaults to `(2, 14, 14)`):
            The size (resolution) of each spatialtemporal patch. If `patch_size` is an int, spatial patches of shape
            `(patch_size, patch_size)` will be used; otherwise, `patch_size` should be a tuple of shape
            `(time_patch_size, height_patch_size, width_patch_size)`.
        stride (`int` or `Tuple[int]`, *optional*, defaults to `(2, 14, 14)`):
            The stride of the imate patch embedding. If `stride` is an int, spatial strides of shape
            `(stride, stride)` will be used; otherwise, `patch_size` should be a tuple of shape
            `(time_stride, height_stride, width_stride)`.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        add_kv_bias(`bool`, *optional*, defaults to `False`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to `None`):
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
        intermediate_size=5120,
        projection_dim=1024,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        num_frames=2,
        image_size=224,
        patch_size=(2, 14, 14),
        stride=(2, 14, 14),
        hidden_act="quick_gelu",
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
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.stride = stride
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
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
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
            TODO
        num_channels (`int`, *optional*, defaults to 1):
            The number of channels in the input audio data.
        patch_size (`int`, *optional*, defaults to 16):
            The kernel size of the patch embedding 2D convolution layer.
        stride (`int`, *optional*, defaults to 10):
            The stride of the patch embedding 2D convolution layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        add_kv_bias(`bool`, *optional*, defaults to `True`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to `20.0`):
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
        intermediate_size=3072,
        projection_dim=1024,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_mel_bins=128,
        target_len=204,
        num_channels=1,
        patch_size=16,
        stride=10,
        hidden_act="quick_gelu",
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
        self.intermediate_size = intermediate_size
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


class ImageBindDepthConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ImageBindDepthModel`]. It is used to instantiate a
    ImageBind depth encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the depth encoder of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Args:
        hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 1024):
            If the ImageBind depth model has an output projection layer, the dimension to which that projection layer
            maps to.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 1):
            The number of channels in the input depth data.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The kernel size of the depth patch embedding 2D convolution layer.
        stride (`int`, *optional*, defaults to 16):
            The stride of the depth patch embedding 2D convolution layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        add_kv_bias(`bool`, *optional*, defaults to `True`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to `5.0`):
            The initial value of the `logit_scale` parameter for the vision component. If `None`, the logits will not
            be scaled.
        learnable_logit_scale (`bool`, *optional*, defaults to `False`):
            Whether the `logit_scale` is learnable or fixed.
    
    Example:
    ```python
    >>> from transformers import ImageBindDepthConfig, ImageBindDepthModel

    >>> # Initializing a ImageBindDepthConfig with facebook/imagebind-huge style configuration
    >>> configuration = ImageBindDepthConfig()

    >>> # Initializing a ImageBindDepthModel (with random weights) from the facebook/imagebind-huge style configuration
    >>> model = ImageBindDepthModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    def __init__(
        self,
        hidden_size=384,
        intermediate_size=1536,
        projection_dim=1024,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_channels=1,
        image_size=224,
        patch_size=16,
        stride=16,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-6,
        add_kv_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        logit_scale_init_value=5.0,
        learnable_logit_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
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
            config_dict = config_dict["depth_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindThermalConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ImageBindThermalModel`]. It is used to instantiate a
    ImageBind thermal encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the thermal encoder of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 1024):
            If the ImageBind thermal model has an output projection layer, the dimension to which that projection layer
            maps to.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 1):
            The number of channels in the input thermal data.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The kernel size of the thermal patch embedding 2D convolution layer.
        stride (`int`, *optional*, defaults to 16):
            The stride of the thermal patch embedding 2D convolution layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        add_kv_bias(`bool`, *optional*, defaults to `True`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for the DropPath (stochastic) regularization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to `10.0`):
            The initial value of the `logit_scale` parameter for the vision component. If `None`, the logits will not
            be scaled.
        learnable_logit_scale (`bool`, *optional*, defaults to `False`):
            Whether the `logit_scale` is learnable or fixed.
    
    Example:
    ```python
    >>> from transformers import ImageBindThermalConfig, ImageBindThermalModel

    >>> # Initializing a ImageBindThermalConfig with facebook/imagebind-huge style configuration
    >>> configuration = ImageBindThermalConfig()

    >>> # Initializing a ImageBindThermalModel (with random weights) from the facebook/imagebind-huge style configuration
    >>> model = ImageBindThermalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=1024,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=1,
        image_size=224,
        patch_size=16,
        stride=16,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-6,
        add_kv_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        logit_scale_init_value=10.0,
        learnable_logit_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
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
            config_dict = config_dict["thermal_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindImuConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ImageBindImuModel`]. It is used to instantiate a
    ImageBind IMU encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the IMU encoder of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Args:
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 1024):
            If the ImageBind thermal model has an output projection layer, the dimension to which that projection layer
            maps to.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        input_shape ('Tuple[int]`, *optional*, defaults to `(6, 2000)`):
            The shape of the input IMU data.
        kernel_size (`int`, *optional*, defaults to 8):
            The kernel size of the 2D convolution layers. (TODO)
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        add_kv_bias(`bool`, *optional*, defaults to `True`):
            Whether to add an extra learnable bias token to the attention key and value sequences. This is based on the
            `add_kv_bias` argument to [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.7):
            The dropout probability for the DropPath (stochastic) regularization layers.
        final_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability for the dropout layer that occurs after the post layer norm and before the linear
            projection is applied.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        logit_scale_init_value (`float`, *optional*, defaults to `5.0`):
            The initial value of the `logit_scale` parameter for the vision component. If `None`, the logits will not
            be scaled.
        learnable_logit_scale (`bool`, *optional*, defaults to `False`):
            Whether the `logit_scale` is learnable or fixed.
    
    Example:
    ```python
    >>> from transformers import ImageBindImuConfig, ImageBindImuModel

    >>> # Initializing a ImageBindImuConfig with facebook/imagebind-huge style configuration
    >>> configuration = ImageBindImuConfig()

    >>> # Initializing a ImageBindImuModel (with random weights) from the facebook/imagebind-huge style configuration
    >>> model = ImageBindImuModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    def __init__(
        self,
        hidden_size=512,
        intermediate_size=2048,
        projection_dim=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        input_shape=(6, 2000),
        kernel_size=8,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-6,
        add_kv_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.7,
        final_dropout=0.5,
        initializer_range=0.02,
        initializer_factor=1.0,
        logit_scale_init_value=5.0,
        learnable_logit_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.add_kv_bias = add_kv_bias
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.final_dropout = final_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.logit_scale_init_value = logit_scale_init_value
        self.learnable_logit_scale = learnable_logit_scale
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the audio config dict if we are loading from ImageBindConfig
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["imu_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


# TODO: add configs for other modalities (audio, depth, thermal, IMU)
class ImageBindConfig(PretrainedConfig):
    r"""
    [`ImageBindConfig`] is the configuration class to store the configuration of a [`ImageBindModel`]. It is used to instantiate
    a ImageBind model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the ImageBind
    [facebook/imagebind-huge](https://huggingface.co/facebook/imagebind-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ImageBindTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ImageBindVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original ImageBind implementation.
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
        text_config=None,
        vision_config=None,
        audio_config=None,
        depth_config=None,
        thermal_config=None,
        imu_config=None,
        projection_dim=1024,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        audio_config_dict = kwargs.pop("audio_config_dict", None)
        depth_config_dict = kwargs.pop("depth_config_dict", None)
        thermal_config_dict = kwargs.pop("thermal_config_dict", None)
        imu_config_dict = kwargs.pop("imu_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = ImageBindTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `ImageBindTextConfig`. The "
                            f'value `text_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = ImageBindVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `ImageBindVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)
        
        if audio_config_dict is not None:
            if audio_config is None:
                audio_config = {}

            # This is the complete result when using `audio_config_dict`.
            _audio_config_dict = ImageBindAudioConfig(**audio_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_audio_config_dict` and `audio_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in audio_config and value != audio_config[key] and key not in ["transformers_version"]:
                    # If specified in `audio_config_dict`
                    if key in audio_config_dict:
                        message = (
                            f"`{key}` is found in both `audio_config_dict` and `audio_config` but with different "
                            f'values. The value `audio_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`audio_config_dict` is provided which will be used to initialize `ImageBindAudioConfig`. "
                            f'The value `audio_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_audio_config_dict`.
            audio_config.update(_audio_config_dict)
        
        if depth_config_dict is not None:
            if depth_config is None:
                depth_config = {}

            # This is the complete result when using `depth_config_dict`.
            _depth_config_dict = ImageBindDepthConfig(**depth_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _depth_config_dict:
                _depth_config_dict["id2label"] = {
                    str(key): value for key, value in _depth_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_depth_config_dict` and `depth_config` but being different.
            for key, value in _depth_config_dict.items():
                if key in depth_config and value != depth_config[key] and key not in ["transformers_version"]:
                    # If specified in `depth_config_dict`
                    if key in depth_config_dict:
                        message = (
                            f"`{key}` is found in both `depth_config_dict` and `depth_config` but with different "
                            f'values. The value `depth_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`depth_config_dict` is provided which will be used to initialize `ImageBindDepthConfig`. "
                            f'The value `depth_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_depth_config_dict`.
            depth_config.update(_depth_config_dict)
        
        if thermal_config_dict is not None:
            if thermal_config is None:
                thermal_config = {}

            # This is the complete result when using `thermal_config_dict`.
            _thermal_config_dict = ImageBindThermalConfig(**thermal_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _thermal_config_dict:
                _thermal_config_dict["id2label"] = {
                    str(key): value for key, value in _thermal_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_thermal_config_dict` and `thermal_config` but being different.
            for key, value in _thermal_config_dict.items():
                if key in thermal_config and value != thermal_config[key] and key not in ["transformers_version"]:
                    # If specified in `thermal_config_dict`
                    if key in thermal_config_dict:
                        message = (
                            f"`{key}` is found in both `thermal_config_dict` and `thermal_config` but with different "
                            f'values. The value `thermal_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`thermal_config_dict` is provided which will be used to initialize `ImageBindThermalConfig`. "
                            f'The value `thermal_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_thermal_config_dict`.
            thermal_config.update(_thermal_config_dict)
        
        if imu_config_dict is not None:
            if imu_config is None:
                imu_config = {}

            # This is the complete result when using `imu_config_dict`.
            _imu_config_dict = ImageBindImuConfig(**imu_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _imu_config_dict:
                _imu_config_dict["id2label"] = {
                    str(key): value for key, value in _imu_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_imu_config_dict` and `imu_config` but being different.
            for key, value in _imu_config_dict.items():
                if key in imu_config and value != imu_config[key] and key not in ["transformers_version"]:
                    # If specified in `imu_config_dict`
                    if key in imu_config_dict:
                        message = (
                            f"`{key}` is found in both `imu_config_dict` and `imu_config` but with different "
                            f'values. The value `imu_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`imu_config_dict` is provided which will be used to initialize `ImageBindImuConfig`. "
                            f'The value `imu_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `imu_config` with the ones in `_imu_config_dict`.
            imu_config.update(_imu_config_dict)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `ImageBindTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `ImageBindVisionConfig` with default values.")
        
        if audio_config is None:
            audio_config = {}
            logger.info("`audio_config` is `None`. initializing the `ImageBindAudioConfig` with default values.")
        
        if depth_config is None:
            depth_config = {}
            logger.info("`depth_config` is `None`. initializing the `ImageBindDepthConfig` with default values.")
        
        if thermal_config is None:
            thermal_config = {}
            logger.info("`thermal_config` is `None`. initializing the `ImageBindThermalConfig` with default values.")
        
        if imu_config is None:
            imu_config = {}
            logger.info("`imu_config` is `None`. initializing the `ImageBindImuConfig` with default values.")

        self.text_config = ImageBindTextConfig(**text_config)
        self.vision_config = ImageBindVisionConfig(**vision_config)
        self.audio_config = ImageBindAudioConfig(**audio_config)
        self.depth_config = ImageBindDepthConfig(**depth_config)
        self.thermal_config = ImageBindThermalConfig(**thermal_config)
        self.imu_config = ImageBindImuConfig(**imu_config)

        self.projection_dim = projection_dim
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: ImageBindTextConfig, vision_config: ImageBindVisionConfig, **kwargs):
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
        output["depth_config"] = self.depth_config.to_dict()
        output["thermal_config"] = self.thermal_config.to_dict()
        output["imu_config"] = self.imu_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output

# TODO: add other modalities
class ImageBindOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("logits_per_image", {0: "batch"}),
                ("logits_per_text", {0: "batch"}),
                ("text_embeds", {0: "batch"}),
                ("image_embeds", {0: "batch"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        image_input_dict = super().generate_dummy_inputs(
            processor.feature_extractor, batch_size=batch_size, framework=framework
        )
        return {**text_input_dict, **image_input_dict}

    @property
    def default_onnx_opset(self) -> int:
        return 14