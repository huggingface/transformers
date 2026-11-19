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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/mobilenet_v1_1.0_224")
@strict
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

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    depth_multiplier: float | int = 1.0
    min_depth: int = 8
    hidden_act: str = "relu6"
    tf_padding: bool = True
    classifier_dropout_prob: float | int = 0.999
    initializer_range: float = 0.02
    layer_norm_eps: float = 0.001

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")


__all__ = ["MobileNetV1Config"]
