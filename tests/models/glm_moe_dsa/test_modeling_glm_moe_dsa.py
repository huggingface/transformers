# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch GlmMoeDsa model."""

import unittest

from transformers import Cache, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import GlmMoeDsaForCausalLM, GlmMoeDsaModel


class GlmMoeDsaModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GlmMoeDsaModel
        causal_lm_class = GlmMoeDsaForCausalLM

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_hidden_layers=2,
        mlp_layer_types=["sparse", "dense"],
    ):
        super().__init__(parent=parent, num_hidden_layers=num_hidden_layers)
        self.n_routed_experts = n_routed_experts
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mlp_layer_types = mlp_layer_types


@require_torch
class GlmMoeDsaModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = GlmMoeDsaModelTester
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.7, 0.8]

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as GLM-4.7-Flash has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)


@require_torch_accelerator
@slow
class GlmMoeDsaIntegrationTest(unittest.TestCase):
    pass
