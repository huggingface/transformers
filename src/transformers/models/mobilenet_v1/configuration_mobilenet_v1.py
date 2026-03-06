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
"""MobileNetV1 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/mobilenet_v1_1.0_224")
class MobileNetV1Config(PreTrainedConfig):
    r"""
    min_depth (`int`, *optional*, defaults to 8):
        All layers will have at least this many channels.
    tf_padding (`bool`, *optional*, defaults to `True`):
        Whether to use TensorFlow padding rules on the convolution layers.

    Example:

    ```python
    >>> from transformers import MobileNetV1Config, MobileNetV1Model

    >>> # Initializing a "mobilenet_v1_1.0_224" style configuration
    >>> configuration = MobileNetV1Config()

    >>> # Initializing a model from the "mobilenet_v1_1.0_224" style configuration
    >>> model = MobileNetV1Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mobilenet_v1"

    def __init__(
        self,
        num_channels=3,
        image_size=224,
        depth_multiplier=1.0,
        min_depth=8,
        hidden_act="relu6",
        tf_padding=True,
        classifier_dropout_prob=0.999,
        initializer_range=0.02,
        layer_norm_eps=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")

        self.num_channels = num_channels
        self.image_size = image_size
        self.depth_multiplier = depth_multiplier
        self.min_depth = min_depth
        self.hidden_act = hidden_act
        self.tf_padding = tf_padding
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


__all__ = ["MobileNetV1Config"]
