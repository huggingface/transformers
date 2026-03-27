# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""MobileViT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/mobilenet_v2_1.0_224")
@strict
class MobileViTConfig(PreTrainedConfig):
    r"""
    neck_hidden_sizes (`list[int]`, *optional*, defaults to `[16, 32, 64, 96, 128, 160, 640]`):
        The number of channels for the feature maps of the backbone.
    aspp_out_channels (`int`, *optional*, defaults to 256):
        Number of output channels used in the ASPP layer for semantic segmentation.
    atrous_rates (`list[int]`, *optional*, defaults to `[6, 12, 18]`):
        Dilation (atrous) factors used in the ASPP layer for semantic segmentation.
    aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the ASPP layer for semantic segmentation.

    Example:

    ```python
    >>> from transformers import MobileViTConfig, MobileViTModel

    >>> # Initializing a mobilevit-small style configuration
    >>> configuration = MobileViTConfig()

    >>> # Initializing a model from the mobilevit-small style configuration
    >>> model = MobileViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mobilevit"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 256
    patch_size: int | list[int] | tuple[int, int] = 2
    hidden_sizes: list[int] | tuple[int, ...] = (144, 192, 240)
    neck_hidden_sizes: list[int] | tuple[int, ...] = (16, 32, 64, 96, 128, 160, 640)
    num_attention_heads: int = 4
    mlp_ratio: float = 2.0
    expand_ratio: float = 4.0
    hidden_act: str = "silu"
    conv_kernel_size: int = 3
    output_stride: int = 32
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.0
    classifier_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    qkv_bias: bool = True
    aspp_out_channels: int = 256
    atrous_rates: list[int] | tuple[int, ...] = (6, 12, 18)
    aspp_dropout_prob: float = 0.1
    semantic_loss_ignore_index: int = 255


__all__ = ["MobileViTConfig"]
