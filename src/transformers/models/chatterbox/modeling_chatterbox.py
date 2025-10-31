# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Chatterbox model implementation
"""

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_chatterbox import ChatterboxConfig


logger = logging.get_logger(__name__)


class ChatterboxModel(PreTrainedModel):
    config_class = ChatterboxConfig
    base_model_prefix = "chatterbox"
