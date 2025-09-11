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

from ...configuration_utils import PretrainedConfig
from ..qwen2.configuration_qwen2 import Qwen2Config


class Ovis2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ovis2VisionModel`]. It is used to instantiate a
    Ovis2VisionModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of Ovis2.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2816):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMSNorm layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a learnable bias to the query, key, and value sequences at each attention head.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a learnable bias to the MLP layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        vocab_size (`int`, *optional*, defaults to 16384):
            Vocabulary size of the Vision Transformer.
        hidden_stride (`int`, *optional*, defaults to 1):
            The stride of the hidden layer in the Vision Transformer.
        num_visual_indicator_tokens (`int`, *optional*, defaults to 5):
            Number of visual indicator tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.
        tokenize_function (`str`, *optional*, defaults to `"softmax"`):
            The function used to tokenize the visual indicator tokens.
    """

    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        qkv_bias: bool = False,
        mlp_bias: bool = False,
        hidden_act="silu",
        vocab_size=16384,
        hidden_stride=1,
        num_visual_indicator_tokens=5,
        initializer_range=0.02,
        tokenize_function="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size

        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.hidden_stride = hidden_stride
        self.num_visual_indicator_tokens = num_visual_indicator_tokens
        self.tokenize_function = tokenize_function
        self.initializer_range = initializer_range


class Ovis2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ovis2ForConditionalGeneration`]. It is used to instantiate a
    Ovis2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of Ovis2.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    e.g. [thisisiron/Ovis2-1B-hf](https://huggingface.co/thisisiron/Ovis2-1B-hf)

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `Ovis2VisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 151665):
            The image token id to encode the image prompt.
        visual_indicator_token_ids (`List[int]`, *optional*, defaults to `[151666, 151667, 151668, 151669, 151670]`):
            The visual indicator token ids to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 151643):
            Vocabulary size of the text model.
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.

    ```python
    >>> from transformers import Ovis2ForConditionalGeneration, Ovis2Config

    >>> # Initializing a Ovis2 style configuration
    >>> configuration = Ovis2Config()

    >>> # Initializing a model from the Ovis2-2B style configuration
    >>> model = Ovis2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "ovis2"
    sub_configs = {"text_config": Qwen2Config, "vision_config": Ovis2VisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151665,
        visual_indicator_token_ids=[151666, 151667, 151668, 151669, 151670],
        vocab_size=151643,
        hidden_size=1536,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Ovis2VisionConfig(**vision_config)
        elif isinstance(vision_config, Ovis2VisionConfig):
            self.vision_config = vision_config
        if vision_config is None:
            self.vision_config = Ovis2VisionConfig(num_visual_indicator_tokens=len(visual_indicator_token_ids))

        if isinstance(text_config, dict):
            self.text_config = Qwen2Config(**text_config)
        elif isinstance(text_config, Qwen2Config):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = Qwen2Config()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_token_id = image_token_id
        self.visual_indicator_token_ids = visual_indicator_token_ids
        super().__init__(**kwargs)


__all__ = ["Ovis2VisionConfig", "Ovis2Config"]
