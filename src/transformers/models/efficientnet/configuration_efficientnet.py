# Copyright 2023 Google Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""EfficientNet model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/efficientnet-b7")
class EfficientNetConfig(PreTrainedConfig):
    r"""
    width_coefficient (`float`, *optional*, defaults to 2.0):
        Scaling coefficient for network width at each stage.
    depth_coefficient (`float`, *optional*, defaults to 3.1):
        Scaling coefficient for network depth at each stage.
    depth_divisor (`int`, *optional*, defaults to 8):
        A unit of network width.
    num_block_repeats (`list[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
        List of the number of times each block is to repeated.
    expand_ratios (`list[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
        List of scaling coefficient of each block.
    squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
        Squeeze expansion ratio.
    pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
        Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
        `"max"`]
    drop_connect_rate (`float`, *optional*, defaults to 0.2):
        The drop rate for skip connections.
    kernel_sizes (`list[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
        List of kernel sizes to be used in each block.
    out_channels (`list[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
        List of output channel sizes to be used in each block for convolutional layers.
    depthwise_padding (`list[int]`, *optional*, defaults to `[]`):
        List of block indices with square padding.
    batch_norm_momentum (`float`, *optional*, defaults to 0.99):
        The momentum used by the batch normalization layers.

    Example:
    ```python
    >>> from transformers import EfficientNetConfig, EfficientNetModel

    >>> # Initializing a EfficientNet efficientnet-b7 style configuration
    >>> configuration = EfficientNetConfig()

    >>> # Initializing a model (with random weights) from the efficientnet-b7 style configuration
    >>> model = EfficientNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "efficientnet"

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 600,
        width_coefficient: float = 2.0,
        depth_coefficient: float = 3.1,
        depth_divisor: int = 8,
        kernel_sizes: list[int] = [3, 3, 5, 3, 5, 5, 3],
        in_channels: list[int] = [32, 16, 24, 40, 80, 112, 192],
        out_channels: list[int] = [16, 24, 40, 80, 112, 192, 320],
        depthwise_padding: list[int] = [],
        strides: list[int] = [1, 2, 2, 2, 1, 2, 1],
        num_block_repeats: list[int] = [1, 2, 2, 3, 3, 4, 1],
        expand_ratios: list[int] = [1, 6, 6, 6, 6, 6, 6],
        squeeze_expansion_ratio: float = 0.25,
        hidden_act: str = "swish",
        hidden_dim: int = 2560,
        pooling_type: str = "mean",
        initializer_range: float = 0.02,
        batch_norm_eps: float = 0.001,
        batch_norm_momentum: float = 0.99,
        dropout_rate: float = 0.5,
        drop_connect_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.depth_divisor = depth_divisor
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = depthwise_padding
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.squeeze_expansion_ratio = squeeze_expansion_ratio
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.initializer_range = initializer_range
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.num_hidden_layers = sum(num_block_repeats) * 4


__all__ = ["EfficientNetConfig"]
