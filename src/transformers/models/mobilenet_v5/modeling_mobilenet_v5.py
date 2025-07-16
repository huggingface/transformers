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
"""MobileNetV5 model stub implementation"""

from ...modeling_utils import PreTrainedModel
from .configuration_mobilenet_v5 import MobileNetV5Config
import torch.nn as nn

class MobileNetV5Model(PreTrainedModel):
    """
    This class provides a minimal stub for the MobileNetV5 model architecture.
    Currently, it does not implement any real logic and serves only to avoid 'Unknown Model' errors.
    Contributions for a full implementation are welcome!
    """
    config_class = MobileNetV5Config

    def __init__(self, config: MobileNetV5Config):
        super().__init__(config)
        # Minimal stub: a single dummy layer
        self.dummy = nn.Identity()

    def forward(self, pixel_values=None, **kwargs):
        # This is a stub. Real implementation required for actual use.
        if pixel_values is None:
            raise ValueError("pixel_values must be provided for MobileNetV5Model (stub)")
        return self.dummy(pixel_values) 