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
"""MobileNetV2 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/mobilenet_v2_1.0_224")
class MobileNetV2Config(PreTrainedConfig):
    r"""
    depth_divisible_by (`int`, *optional*, defaults to 8):
        The number of channels in each layer will always be a multiple of this number.
    min_depth (`int`, *optional*, defaults to 8):
        All layers will have at least this many channels.
    expand_ratio (`float`, *optional*, defaults to 6.0):
        The number of output channels of the first layer in each block is input channels times expansion ratio.
    output_stride (`int`, *optional*, defaults to 32):
        The ratio between the spatial resolution of the input and output feature maps. By default the model reduces
        the input dimensions by a factor of 32. If `output_stride` is 8 or 16, the model uses dilated convolutions
        on the depthwise layers instead of regular convolutions, so that the feature maps never become more than 8x
        or 16x smaller than the input image.
    first_layer_is_expansion (`bool`, *optional*, defaults to `True`):
        True if the very first convolution layer is also the expansion layer for the first expansion block.
    finegrained_output (`bool`, *optional*, defaults to `True`):
        If true, the number of output channels in the final convolution layer will stay large (1280) even if
        `depth_multiplier` is less than 1.
    tf_padding (`bool`, *optional*, defaults to `True`):
        Whether to use TensorFlow padding rules on the convolution layers.

    Example:

    ```python
    >>> from transformers import MobileNetV2Config, MobileNetV2Model

    >>> # Initializing a "mobilenet_v2_1.0_224" style configuration
    >>> configuration = MobileNetV2Config()

    >>> # Initializing a model from the "mobilenet_v2_1.0_224" style configuration
    >>> model = MobileNetV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mobilenet_v2"

    def __init__(
        self,
        num_channels=3,
        image_size=224,
        depth_multiplier=1.0,
        depth_divisible_by=8,
        min_depth=8,
        expand_ratio=6.0,
        output_stride=32,
        first_layer_is_expansion=True,
        finegrained_output=True,
        hidden_act="relu6",
        tf_padding=True,
        classifier_dropout_prob=0.8,
        initializer_range=0.02,
        layer_norm_eps=0.001,
        semantic_loss_ignore_index=255,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")

        self.num_channels = num_channels
        self.image_size = image_size
        self.depth_multiplier = depth_multiplier
        self.depth_divisible_by = depth_divisible_by
        self.min_depth = min_depth
        self.expand_ratio = expand_ratio
        self.output_stride = output_stride
        self.first_layer_is_expansion = first_layer_is_expansion
        self.finegrained_output = finegrained_output
        self.hidden_act = hidden_act
        self.tf_padding = tf_padding
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.semantic_loss_ignore_index = semantic_loss_ignore_index


__all__ = ["MobileNetV2Config"]
