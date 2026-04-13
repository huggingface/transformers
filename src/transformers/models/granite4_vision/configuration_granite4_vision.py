# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Granite 4 Vision model configuration"""

from typing import Literal, Optional

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ibm-granite/granite-4.0-3b-vision")
@strict(accept_kwargs=True)
class Granite4VisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Granite4VisionForConditionalGeneration`]. It is used to instantiate a
    Granite 4 Vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Granite 4 Vision
    [ibm-granite/granite-4.0-3b-vision](https://huggingface.co/ibm-granite/granite-4.0-3b-vision) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 100352):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"full"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to select the vision feature.
        image_grid_pinpoints (`List`, *optional*):
            A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        downsample_rate (`str`, *optional*, defaults to `"4/8"`):
            The downsample rate for the vision features, specified as "query_side/window_side".
        use_image_newline_parameter (`bool`, *optional*, defaults to `True`):
            Whether to use a learnable newline parameter for image features.
        deepstack_layer_map (`List[List[int]]`, *optional*):
            List of (vision_layer_idx, llm_layer_idx) tuples for deepstack feature injection.
            Features from each vision layer are extracted, downsampled, and injected at the corresponding LLM layer.
        use_spatial_sampling (`bool`, *optional*, defaults to `False`):
            Whether to use spatial sampling for additional feature extraction.
        spatial_stride (`int`, *optional*, defaults to 2):
            Stride for spatial sampling.
        spatial_vision_layer (`int`, *optional*, defaults to -1):
            Vision layer to use for spatial sampling.
        spatial_target_layers (`List[int]`, *optional*):
            LLM layers to inject spatial features into.
        projector_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability for the projector.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of image features.

    Example:

    ```python
    >>> from transformers import Granite4VisionForConditionalGeneration, Granite4VisionConfig, SiglipVisionConfig, GraniteMoeHybridConfig

    >>> # Initializing a Siglip vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a GraniteMoeHybrid config
    >>> text_config = GraniteMoeHybridConfig()

    >>> # Initializing a Granite4Vision configuration
    >>> configuration = Granite4VisionConfig(vision_config, text_config)

    >>> # Initializing a model from the configuration
    >>> model = Granite4VisionForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite4_vision"
    attribute_map = {"image_token_id": "image_token_index"}
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 100352
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: Literal["default", "full"] = "full"
    vision_feature_layer: int = -1
    tie_word_embeddings: bool = True
    image_grid_pinpoints: list | None = None
    image_seq_length: int = 576
    downsample_rate: str = "4/8"
    use_image_newline_parameter: bool = True
    deepstack_layer_map: list | None = None
    use_spatial_sampling: bool = False
    spatial_stride: int = 2
    spatial_vision_layer: int = -1
    spatial_target_layers: list | None = None
    projector_dropout: float = 0.1

    def __post_init__(self, **kwargs):
        # Deepstack layer map: list of (vision_layer_idx, llm_layer_idx) tuples.
        # Features from each vision layer are extracted, downsampled, and injected
        # at the corresponding LLM layer during forward pass.
        # e.g., [(-19, 9), (-13, 6), (-7, 3), (-1, 0)]
        if self.deepstack_layer_map is not None:
            self.deepstack_layer_map = [(int(v), int(l)) for v, l in self.deepstack_layer_map]
            if len(self.deepstack_layer_map) != len(set(self.deepstack_layer_map)):
                raise ValueError("deepstack_layer_map should not contain duplicates")

        # Spatial sampling: extracts 4 groups from a single vision layer using
        # spatial offset sampling (top-left, top-right, bottom-left, bottom-right
        # of each 2x2 block), each injected at a different LLM layer.
        if self.spatial_target_layers is None:
            self.spatial_target_layers = [0, 10, 20, 30]

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                image_size=384,
                intermediate_size=4304,
                num_attention_heads=16,
                num_hidden_layers=27,
                patch_size=16,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "granitemoehybrid")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["granitemoehybrid"]()

        self.image_grid_pinpoints = (
            self.image_grid_pinpoints
            if self.image_grid_pinpoints is not None
            else [
                [384, 384],
                [384, 768],
                [384, 1152],
                [384, 1536],
                [384, 1920],
                [384, 2304],
                [384, 2688],
                [384, 3072],
                [384, 3456],
                [384, 3840],
                [768, 384],
                [768, 768],
                [768, 1152],
                [768, 1536],
                [768, 1920],
                [1152, 384],
                [1152, 768],
                [1152, 1152],
                [1536, 384],
                [1536, 768],
                [1920, 384],
                [1920, 768],
                [2304, 384],
                [2688, 384],
                [3072, 384],
                [3456, 384],
                [3840, 384],
            ]
        )

        super().__post_init__(**kwargs)


__all__ = ["Granite4VisionConfig"]

# Made with Bob
