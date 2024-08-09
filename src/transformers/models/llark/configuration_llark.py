# coding=utf-8
# Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
""" Llark model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

LLARK_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llark-hf/llark-v1.5-7b": "https://huggingface.co/llark-hf/llark-v1.5-7b/resolve/main/config.json",
}


class LlarkConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlarkForConditionalGeneration`]. It is used to instantiate an
    Llark model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llark-9B.

    e.g. [llark-hf/llark-9b](https://huggingface.co/llark-hf/llark-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom audio config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Llark model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LlarkForConditionalGeneration`]

    Example:

    ```python
    >>> from transformers import LlarkForConditionalGeneration, LlarkConfig, MusicgenConfig, LlamaConfig

    >>> # Initializing a MusicGen config
    >>> audio_config = MusicgenConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llark llark-1.5-7b style configuration
    >>> configuration = LlarkConfig(audio_config, text_config)

    >>> # Initializing a model from the llark-1.5-7b style configuration
    >>> model = LlarkForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llark"
    is_composition = False

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        ignore_index=-100,
        audio_feature_layer=-1,
        downsample_rate=5,
        vocab_size=32000,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.audio_feature_layer = audio_feature_layer
        self.downsample_rate = downsample_rate
        self.vocab_size = vocab_size

        self.audio_config = audio_config

        if isinstance(self.audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "clip_vision_model"
            )
            self.audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            self.audio_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )
        self.vocab_size = self.vocab_size

        self.text_config = text_config

        self.AUDIO_START_TOKEN = "<AUDIO_START>"
        self.AUDIO_END_TOKEN = "<AUDIO_END>"

        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        super().__init__(**kwargs)
