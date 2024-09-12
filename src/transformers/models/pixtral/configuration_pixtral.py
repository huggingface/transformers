# coding=utf-8
# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""Pixtral model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class PixtralConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PixtralModel`]. It is used to instantiate an
    Pixtral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Pixtral-9B.

    e.g. [pixtral-hf/pixtral-9b](https://huggingface.co/pixtral-hf/pixtral-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.

    Example:

    ```python
    >>> from transformers import PixtralModel, PixtralConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Pixtral pixtral-1.5-7b style configuration
    >>> configuration = PixtralConfig(vision_config, text_config)

    >>> # Initializing a model from the pixtral-1.5-7b style configuration
    >>> model = PixtralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pixtral"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_activation="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        rope_theta=10000.0,
        tie_word_embeddings=False,
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
        self.layer_norm_eps = layer_norm_eps
        self.hidden_activation = hidden_activation
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.head_dim = hidden_size // num_attention_heads
