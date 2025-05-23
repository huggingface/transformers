# coding=utf-8
# Copyright 2025 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""PerceptionLM model configuration"""

from transformers.configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig
from typing import Tuple


logger = logging.get_logger(__name__)


class PerceptionEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PerceptionEncoder`]. It is used to instantiate a
    PerceptionEncoder model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_type (`str`, *optional*, defaults to `"perception_encoder"`):
            The type of the model.
        use_cls_token (`bool`, *optional*, defaults to `True`):
            Whether to use a CLS token.
        architecture (`str`, *optional*, defaults to `"vit_pe_core_large_patch14_336"`):
            The architecture of the model.
        width (`int`, *optional*, defaults to `1024`):
            The width of the model.
        img_size (`Tuple[int, int]`, *optional*, defaults to `(448, 448)`):
            The size of the input image.
        depth (`int`, *optional*, defaults to `23`):
            The depth of the model.
        num_classes (`int`, *optional*, defaults to `0`):
            The number of classes for classification.
        global_pool (`str`, *optional*, defaults to `""`):
            The global pooling strategy.
        use_post_transformer_norm (`bool`, *optional*, defaults to `False`):
            Whether to use post-transformer normalization.
        init_values (`float`, *optional*, defaults to `0.1`):
            The initialization values.
        ref_feat_shape (`Tuple[int, int]`, *optional*, defaults to `(32, 32)`):
            The shape of the reference feature.

    Example:

    ```python
    >>> from transformers import PerceptionEncoder, PerceptionEncoderConfig

    >>> # Initializing a PerceptionEncoder configuration
    >>> configuration = PerceptionEncoderConfig()

    >>> # Initializing a model from the configuration
    >>> model = PerceptionEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "perception_encoder"
    def __init__(
        self,
        use_cls_token=True,
        architecture="vit_pe_core_large_patch14_336",
        width=1024,
        img_size=(448, 448),
        depth=23,
        num_classes=0,
        global_pool="",
        use_post_transformer_norm=False,
        init_values=0.1,
        ref_feat_shape=(32, 32),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_cls_token = use_cls_token
        self.architecture = architecture
        self.width = width
        self.img_size = img_size
        self.depth = depth
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.use_post_transformer_norm = use_post_transformer_norm
        self.init_values = init_values
        self.ref_feat_shape = ref_feat_shape

class PerceptionLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PerceptionLMForConditionalGeneration`]. It is used to instantiate an
    PerceptionLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PerceptionLM-9B.

    e.g. [perception_lm-hf/perception_lm-9b](https://huggingface.co/perception_lm-hf/perception_lm-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`Union[int, List[int]]`, *optional*, defaults to -2):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.

    Example:

    ```python
    >>> from transformers import PerceptionLMForConditionalGeneration, PerceptionLMConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a PerceptionLM perception_lm-1.5-7b style configuration
    >>> configuration = PerceptionLMConfig(vision_config, text_config)

    >>> # Initializing a model from the perception_lm-1.5-7b style configuration
    >>> model = PerceptionLMForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "perception_lm"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projector_pooling_ratio=1,
        image_token_id=128002,
        video_token_id=128003,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        if isinstance(vision_config, dict):
            vision_config = PerceptionEncoderConfig(**vision_config)
        elif isinstance(vision_config, PerceptionEncoderConfig):
            vision_config = vision_config
        elif vision_config is None:
            vision_config = PerceptionEncoderConfig()
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        self.projector_pooling_ratio = projector_pooling_ratio
        super().__init__(**kwargs)


__all__ = ["PerceptionLMConfig", "PerceptionEncoderConfig"]
