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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/levit-128S")
@strict
class LevitConfig(PreTrainedConfig):
    r"""
    stride (`int`, *optional*, defaults to 2):
        The stride size for the initial convolution layers of patch embedding.
    padding (`int`, *optional*, defaults to 1):
        The padding size for the initial convolution layers of patch embedding.
    key_dim (`list[int]`, *optional*, defaults to `[16, 16, 16]`):
        The size of key in each of the encoder blocks.
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

    image_size: int | list[int] | tuple[int, int] = 224
    num_channels: int = 3
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_sizes: list[int] | tuple[int, ...] = (128, 256, 384)
    num_attention_heads: list[int] | tuple[int, ...] = (4, 8, 12)
    depths: list[int] | tuple[int, ...] = (4, 4, 4)
    key_dim: list[int] | tuple[int, ...] = (16, 16, 16)
    drop_path_rate: int = 0
    mlp_ratio: list[int] | tuple[int, ...] = (2, 2, 2)
    attention_ratio: list[int] | tuple[int, ...] = (2, 2, 2)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.down_ops = [
            ["Subsample", self.key_dim[0], self.hidden_sizes[0] // self.key_dim[0], 4, 2, 2],
            ["Subsample", self.key_dim[0], self.hidden_sizes[1] // self.key_dim[0], 4, 2, 2],
        ]
        super().__post_init__(**kwargs)


__all__ = ["LevitConfig"]
