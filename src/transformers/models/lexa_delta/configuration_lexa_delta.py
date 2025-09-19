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
"""Lexa-Delta model configuration"""

from transformers.configuration_utils import PretrainedConfig


class LexaDeltaConfig(PretrainedConfig):
    model_type = "lexa_delta"

    def __init__(
        self,
        vocab_size=201088,
        hidden_size=2880,
        num_hidden_layers=24,
        num_attention_heads=64,
        intermediate_size=2880,
        max_position_embeddings=131072,
        rms_norm_eps=1e-05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
