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
"""DepthAnything model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_backbone_utils import consolidate_backbone_kwargs_to_config
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class DepthAnythingConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DepthAnythingModel`]. It is used to instantiate a DepthAnything
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DepthAnything
    [LiheYoung/depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `Dinov2Config()`):
            The configuration of the backbone model.
        patch_size (`int`, *optional*, defaults to 14):
            The size of the patches to extract from the backbone features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        reassemble_hidden_size (`int`, *optional*, defaults to 384):
            The number of input channels of the reassemble layers.
        reassemble_factors (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`list[str]`, *optional*, defaults to `[48, 96, 192, 384]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 64):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the depth estimation head.
        head_hidden_size (`int`, *optional*, defaults to 32):
            The number of output channels in the second convolution of the depth estimation head.
        depth_estimation_type (`str`, *optional*, defaults to `"relative"`):
            The type of depth estimation to use. Can be one of `["relative", "metric"]`.
        max_depth (`float`, *optional*):
            The maximum depth to use for the "metric" depth estimation head. 20 should be used for indoor models
            and 80 for outdoor models. For "relative" depth estimation, this value is ignored.

    Example:

    ```python
    >>> from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation

    >>> # Initializing a DepthAnything small style configuration
    >>> configuration = DepthAnythingConfig()

    >>> # Initializing a model from the DepthAnything small style configuration
    >>> model = DepthAnythingForDepthEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "depth_anything"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        patch_size=14,
        initializer_range=0.02,
        reassemble_hidden_size=384,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[48, 96, 192, 384],
        fusion_hidden_size=64,
        head_in_index=-1,
        head_hidden_size=32,
        depth_estimation_type="relative",
        max_depth=None,
        **kwargs,
    ):
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="dinov2",
            default_config_kwargs={
                "image_size": 518,
                "hidden_size": 384,
                "num_attention_heads": 6,
                "out_indices": [9, 10, 11, 12],
                "reshape_hidden_states": False,
            },
            **kwargs,
        )

        self.backbone_config = backbone_config
        self.reassemble_hidden_size = reassemble_hidden_size
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.head_hidden_size = head_hidden_size
        if depth_estimation_type not in ["relative", "metric"]:
            raise ValueError("depth_estimation_type must be one of ['relative', 'metric']")
        self.depth_estimation_type = depth_estimation_type
        self.max_depth = max_depth if max_depth else 1

        super().__init__(**kwargs)


__all__ = ["DepthAnythingConfig"]
