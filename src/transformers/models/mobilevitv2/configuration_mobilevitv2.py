# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""MobileViTV2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="apple/mobilevitv2-1.0")
@strict
class MobileViTV2Config(PreTrainedConfig):
    r"""
    aspp_out_channels (`int`, *optional*, defaults to 512):
        Number of output channels used in the ASPP layer for semantic segmentation.
    atrous_rates (`list[int]`, *optional*, defaults to `[6, 12, 18]`):
        Dilation (atrous) factors used in the ASPP layer for semantic segmentation.
    aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the ASPP layer for semantic segmentation.
    n_attn_blocks (`list[int]`, *optional*, defaults to `[2, 4, 3]`):
        The number of attention blocks in each MobileViTV2Layer
    base_attn_unit_dims (`list[int]`, *optional*, defaults to `[128, 192, 256]`):
        The base multiplier for dimensions of attention blocks in each MobileViTV2Layer
    width_multiplier (`float`, *optional*, defaults to 1.0):
        The width multiplier for MobileViTV2.
    ffn_multiplier (`int`, *optional*, defaults to 2):
        The FFN multiplier for MobileViTV2.
    ffn_dropout (`float`, *optional*, defaults to 0.0):
        The dropout between FFN layers.

    Example:

    ```python
    >>> from transformers import MobileViTV2Config, MobileViTV2Model

    >>> # Initializing a mobilevitv2-small style configuration
    >>> configuration = MobileViTV2Config()

    >>> # Initializing a model from the mobilevitv2-small style configuration
    >>> model = MobileViTV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mobilevitv2"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 256
    patch_size: int | list[int] | tuple[int, int] = 2
    expand_ratio: float = 2.0
    hidden_act: str = "swish"
    conv_kernel_size: int = 3
    output_stride: int = 32
    classifier_dropout_prob: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    aspp_out_channels: int = 512
    atrous_rates: list[int] | tuple[int, ...] = (6, 12, 18)
    aspp_dropout_prob: float | int = 0.1
    semantic_loss_ignore_index: int = 255
    n_attn_blocks: list[int] | tuple[int, ...] = (2, 4, 3)
    base_attn_unit_dims: list[int] | tuple[int, ...] = (128, 192, 256)
    width_multiplier: float | int = 1.0
    ffn_multiplier: int = 2
    attn_dropout: float | int = 0.0
    ffn_dropout: float | int = 0.0


__all__ = ["MobileViTV2Config"]
