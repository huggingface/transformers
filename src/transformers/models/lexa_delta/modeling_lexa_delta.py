# coding=utf-8
# Copyright 2025 Robi Labs. All rights reserved.
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
"""Lexa-Delta model implementation"""

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from .configuration_lexa_delta import LexaDeltaConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


class LexaDeltaForCausalLM(GptOssForCausalLM):
    """
    Lexa-Delta model for causal language modeling.
    
    This model inherits from GptOssForCausalLM and uses the same architecture
    but with Lexa-Delta specific configuration.
    """
    config_class = LexaDeltaConfig
    
    def __init__(self, config: LexaDeltaConfig):
        super().__init__(config)
