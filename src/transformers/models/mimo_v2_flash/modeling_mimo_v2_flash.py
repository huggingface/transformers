# coding=utf-8
# Copyright 2024 Xiaomi and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MiMo-V2-Flash model."""

from transformers.configuration_utils import PretrainedConfig

from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mimo_v2_flash import MiMoV2FlashConfig

logger = logging.get_logger(__name__)


class MiMoV2FlashPreTrainedModel(PreTrainedModel):
    config_class = MiMoV2FlashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiMoV2FlashDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        # Placeholder for weight initialization
        pass


class MiMoV2FlashModel(MiMoV2FlashPreTrainedModel):
    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        # Placeholder
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Modeling code not yet implemented")


class MiMoV2FlashForCausalLM(MiMoV2FlashPreTrainedModel):
    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        # Placeholder
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Modeling code not yet implemented")
