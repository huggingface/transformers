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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/mobilenet_v2_1.0_224")
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

    def __init__(
        self,
        num_channels=3,
        image_size=256,
        patch_size=2,
        hidden_sizes=[144, 192, 240],
        neck_hidden_sizes=[16, 32, 64, 96, 128, 160, 640],
        num_attention_heads=4,
        mlp_ratio=2.0,
        expand_ratio=4.0,
        hidden_act="silu",
        conv_kernel_size=3,
        output_stride=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        qkv_bias=True,
        aspp_out_channels=256,
        atrous_rates=[6, 12, 18],
        aspp_dropout_prob=0.1,
        semantic_loss_ignore_index=255,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_sizes = hidden_sizes
        self.neck_hidden_sizes = neck_hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.expand_ratio = expand_ratio
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias

        # decode head attributes for semantic segmentation
        self.aspp_out_channels = aspp_out_channels
        self.atrous_rates = atrous_rates
        self.aspp_dropout_prob = aspp_dropout_prob
        self.semantic_loss_ignore_index = semantic_loss_ignore_index


__all__ = ["MobileViTConfig"]
