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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="apple/mobilevitv2-1.0")
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

    def __init__(
        self,
        num_channels=3,
        image_size=256,
        patch_size=2,
        expand_ratio=2.0,
        hidden_act="swish",
        conv_kernel_size=3,
        output_stride=32,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        aspp_out_channels=512,
        atrous_rates=[6, 12, 18],
        aspp_dropout_prob=0.1,
        semantic_loss_ignore_index=255,
        n_attn_blocks=[2, 4, 3],
        base_attn_unit_dims=[128, 192, 256],
        width_multiplier=1.0,
        ffn_multiplier=2,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.expand_ratio = expand_ratio
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.n_attn_blocks = n_attn_blocks
        self.base_attn_unit_dims = base_attn_unit_dims
        self.width_multiplier = width_multiplier
        self.ffn_multiplier = ffn_multiplier
        self.ffn_dropout = ffn_dropout
        self.attn_dropout = attn_dropout
        self.classifier_dropout_prob = classifier_dropout_prob

        # decode head attributes for semantic segmentation
        self.aspp_out_channels = aspp_out_channels
        self.atrous_rates = atrous_rates
        self.aspp_dropout_prob = aspp_dropout_prob
        self.semantic_loss_ignore_index = semantic_loss_ignore_index


__all__ = ["MobileViTV2Config"]
