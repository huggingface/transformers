# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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
from ..auto import CONFIG_MAPPING, AutoConfig


class Mistral3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mistral3ForConditionalGeneration`]. It is used to instantiate an
    Mistral3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `PixtralVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MistralConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 10):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_layer (`Union[int, List[int]]`, *optional*, defaults to -1):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        multimodal_projector_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the multimodal projector.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The downsampling factor for the spatial merge operation.

    Example:

    ```python
    >>> from transformers import Mistral3ForConditionalGeneration, Mistral3Config, PixtralVisionConfig, MistralConfig

    >>> # Initializing a Pixtral-vision config
    >>> vision_config = PixtralVisionConfig()

    >>> # Initializing a Mistral config
    >>> text_config = MistralConfig()

    >>> # Initializing a Mistral3 configuration
    >>> configuration = Mistral3Config(vision_config, text_config)

    >>> # Initializing a model from the mistral3.1 configuration
    >>> model = Mistral3ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mistral3"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=10,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        multimodal_projector_bias=False,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config["model_type"] if "model_type" in vision_config else "pixtral"
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["pixtral"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=1540,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                head_dim=64,
                hidden_act="gelu",
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "mistral"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["mistral"](
                attention_dropout=0.0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=5120,
                initializer_range=0.02,
                intermediate_size=32768,
                max_position_embeddings=131072,
                model_type="mistral",
                num_attention_heads=32,
                num_hidden_layers=40,
                num_key_value_heads=8,
                rms_norm_eps=1e-05,
                rope_theta=1000000000.0,
                sliding_window=None,
                use_cache=True,
                vocab_size=131072,
            )

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias
        self.spatial_merge_size = spatial_merge_size


__all__ = ["Mistral3Config"]
