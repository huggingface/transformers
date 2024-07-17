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
"""MPLUGDocOwl model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class MPLUGDocOwlConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MPLUGDocOwlForConditionalGeneration`]. It is used to instantiate an
    MPLUGDocOwl model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MPLUGDocOwl-Chat.

    e.g. [mplugdocowl-hf/mplugdocowl-Chat](https://huggingface.co/mplugdocowl-hf/mplugdocowl-Chat)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        hreducer_hidden_size (`int`, *optional*, defaults to 1024): The hidden size for the hreducer.
        hreducer_layer_norm (`float`, *optional*, defaults to 1e-06): The layer normalization parameter for the hreducer.
        hreducer_conv_shape (`str`, *optional*, defaults to `"1x4"`): The kernel size for the convolutional layer in the hreducer.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.

    Example:

    ```python
    >>> from transformers import MPLUGDocOwlForConditionalGeneration, MPLUGDocOwlConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a MPLUGDocOwl mplugdocowl-1.5-Chat style configuration
    >>> configuration = MPLUGDocOwlConfig(vision_config, text_config)

    >>> # Initializing a model from the mplugdocowl-1.5-Chat style configuration
    >>> model = MPLUGDocOwlForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mplugdocowl"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        hreducer_hidden_size=1024,
        hreducer_layer_norm=1e-6,
        hreducer_conv_shape="1x4",
        ignore_index=-100,
        image_token_index=32000,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=448,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
                layer_norm_eps=1e-6,
                attention_dropout=0.0,
                initializer_range=0.02,
                initializer_factor=1.0,
                hidden_act="quick_gelu",
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        self._vocab_size = self.text_config.vocab_size
        self.hreducer_hidden_size = hreducer_hidden_size
        self.hreducer_layer_norm = hreducer_layer_norm
        self.hreducer_conv_shape = hreducer_conv_shape
        super().__init__(**kwargs)
