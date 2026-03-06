# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""LeViT model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/levit-128S")
class LevitConfig(PreTrainedConfig):
    r"""
    stride (`int`, *optional*, defaults to 2):
        The stride size for the initial convolution layers of patch embedding.
    padding (`int`, *optional*, defaults to 1):
        The padding size for the initial convolution layers of patch embedding.
    key_dim (`list[int]`, *optional*, defaults to `[16, 16, 16]`):
        The size of key in each of the encoder blocks.
    mlp_ratios (`list[int]`, *optional*, defaults to `[2, 2, 2]`):
        Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
        encoder blocks.
    attention_ratio (`list[int]`, *optional*, defaults to `[2, 2, 2]`):
        Ratio of the size of the output dimension compared to input dimension of attention layers.

    Example:

    ```python
    >>> from transformers import LevitConfig, LevitModel

    >>> # Initializing a LeViT levit-128S style configuration
    >>> configuration = LevitConfig()

    >>> # Initializing a model (with random weights) from the levit-128S style configuration
    >>> model = LevitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "levit"

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        hidden_sizes=[128, 256, 384],
        num_attention_heads=[4, 8, 12],
        depths=[4, 4, 4],
        key_dim=[16, 16, 16],
        drop_path_rate=0,
        mlp_ratio=[2, 2, 2],
        attention_ratio=[2, 2, 2],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.depths = depths
        self.key_dim = key_dim
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.initializer_range = initializer_range
        self.down_ops = [
            ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
            ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
        ]


__all__ = ["LevitConfig"]
