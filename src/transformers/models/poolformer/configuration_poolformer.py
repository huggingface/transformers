# Copyright 2022 Sea AI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""PoolFormer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="sail/poolformer_s12")
@strict
class PoolFormerConfig(PreTrainedConfig):
    r"""
    stride (`int`, *optional*, defaults to 16):
        The stride of the input patch.
    pool_size (`int`, *optional*, defaults to 3):
        The size of the pooling window.
    patch_sizes (`list`, *optional*, defaults to `[7, 3, 3, 3]`):
        The size of the input patch for each encoder block.
    strides (`list`, *optional*, defaults to `[4, 2, 2, 2]`):
        The stride of the input patch for each encoder block.
    padding (`list`, *optional*, defaults to `[2, 1, 1, 1]`):
        The padding of the input patch for each encoder block.
    num_encoder_blocks (`int`, *optional*, defaults to 4):
        The number of encoder blocks.
    use_layer_scale (`bool`, *optional*, defaults to `True`):
        Whether to use layer scale.

    Example:

    ```python
    >>> from transformers import PoolFormerConfig, PoolFormerModel

    >>> # Initializing a PoolFormer sail/poolformer_s12 style configuration
    >>> configuration = PoolFormerConfig()

    >>> # Initializing a model (with random weights) from the sail/poolformer_s12 style configuration
    >>> model = PoolFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "poolformer"

    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    stride: int = 16
    pool_size: int = 3
    mlp_ratio: float = 4.0
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    hidden_sizes: list[int] | tuple[int, ...] = (64, 128, 320, 512)
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    padding: list[int] | tuple[int, ...] = (2, 1, 1, 1)
    num_encoder_blocks: int = 4
    drop_path_rate: float | int = 0.0
    hidden_act: str = "gelu"
    use_layer_scale: bool = True
    layer_scale_init_value: float = 1e-5
    initializer_range: float = 0.02


__all__ = ["PoolFormerConfig"]
