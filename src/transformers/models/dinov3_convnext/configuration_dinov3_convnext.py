# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""ConvNeXT model configuration"""

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class DINOv3ConvNextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DINOv3ConvNextModel`]. It is used to instantiate an
    DINOv3ConvNext model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DINOv3ConvNext
    [facebook/dinov3-convnext-tiny-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        hidden_sizes (`list[int]`, *optional*, defaults to [96, 192, 384, 768]):
            Dimensionality (hidden size) at each stage.
        depths (`list[int]`, *optional*, defaults to [3, 3, 9, 3]):
            The number of layers for each stage.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            The initial value for the layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop rate for stochastic depth.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of input images.

    Example:
    ```python
    >>> from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

    >>> # Initializing a DINOv3ConvNext (tiny variant) style configuration
    >>> config = DINOv3ConvNextConfig()

    >>> # Initializing a model (with random weights)
    >>> model = DINOv3ConvNextModel(config)

    >>> # Accessing the model config
    >>> config = model.config
    ```"""

    model_type = "dinov3_convnext"

    def __init__(
        self,
        num_channels: int = 3,
        hidden_sizes: Optional[list[int]] = None,
        depths: Optional[list[int]] = None,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,
        layer_scale_init_value: float = 1e-6,
        drop_path_rate: float = 0.0,
        image_size: int = 224,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size

    @property
    def num_stages(self) -> int:
        return len(self.hidden_sizes)


__all__ = ["DINOv3ConvNextConfig"]
