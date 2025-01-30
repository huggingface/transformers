# coding=utf-8
# Copyright 2025 TeleAI and the HuggingFace Inc. team. All rights reserved.
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
"""TeleChat2 model configuration"""

from ..llama.configuration_llama import LlamaConfig


class TeleChat2Config(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`TeleChat2Model`]. It is used to instantiate an TeleChat2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TeleChat2-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """


    model_type = "telechat2"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `TeleChat2Model`
    base_model_tp_plan = {
        "h.*.self_attention.query": "colwise",
        "h.*.self_attention.key_value": "colwise",
        "h.*.self_attn.dense": "rowwise",
        "h.*.mlp.gate_proj": "colwise",
        "h.*.mlp.up_proj": "colwise",
        "h.*.mlp.down_proj": "rowwise",
    }


__all__ = ["TeleChat2Config"]
