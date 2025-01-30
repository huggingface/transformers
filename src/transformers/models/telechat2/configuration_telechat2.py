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

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        head_dim=None,
        **kwargs,
    ):
        del self.mlp_bias
        del self.attention_bias


__all__ = ["TeleChat2Config"]
