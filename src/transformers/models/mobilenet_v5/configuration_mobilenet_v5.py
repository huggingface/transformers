# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MobileNetV5 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

class MobileNetV5Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MobileNetV5Model`].
    It is used to instantiate a MobileNetV5 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a minimal configuration.

    Example:
        >>> from transformers import MobileNetV5Config, MobileNetV5Model
        >>> config = MobileNetV5Config()
        >>> model = MobileNetV5Model(config)
    """
    model_type = "mobilenet_v5"

    def __init__(
        self,
        num_channels=3,
        image_size=224,
        num_classes=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_classes = num_classes
        # TODO: Add more architecture-specific parameters when implementing full support 