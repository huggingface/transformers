# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" Llava model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llava/llava-v1.5-7b": "https://huggingface.co/llava/llava-v1.5-7b/resolve/main/config.json",
}


class LlavaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaVisionModel`]. It is used to instantiate an
    Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llava-9B.

    e.g. [HuggingFaceM4/llava-9b](https://huggingface.co/HuggingFaceM4/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer. (elsewhere referred to as `hidden_size`)
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_num_channels (`int`, *optional*, defaults to `3`):
            Number of image channels.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "llava"
    attribute_map = {
        "hidden_size": "embed_dim",
    }

    def __init__(
        self,
        embed_dim=1024,
        image_size=336,
        intermediate_size=4096,
        patch_size=14,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act

        super().__init__(**kwargs)


class LlavaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaVisionModel`]. It is used to instantiate an
    Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llava-9B.

    e.g. [HuggingFaceM4/llava-9b](https://huggingface.co/HuggingFaceM4/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`LlavaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to -200):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Llava model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LlavaVisionModel`]

    Example:

    ```python
    >>> from transformers import LlavaVisionModel, LlavaConfig

    >>> # Initializing a Llava llava-9b style configuration
    >>> configuration = LlavaConfig()

    >>> # Initializing a model from the llava-9b style configuration
    >>> model = LlavaVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=-200,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vocab_size=32000,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vocab_size = vocab_size

        if vision_config is None:
            self.vision_config = LlavaVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = LlavaVisionConfig(**vision_config)
        elif isinstance(vision_config, LlavaVisionConfig):
            self.vision_config = vision_config

        self.text_config = text_config

        if isinstance(self.text_config, dict):
            text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"
            self.text_config = CONFIG_MAPPING[text_model_type](**text_config)
            self.vocab_size = self.text_config.vocab_size

        super().__init__(**kwargs)
