# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM3 model."""

import unittest

from transformers import Cache, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPM3ForCausalLM, MiniCPM3Model
    from transformers.models.minicpm3.modeling_minicpm3 import MiniCPM3RotaryEmbedding


class MiniCPM3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniCPM3Model

    def __init__(
        self,
        parent,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scale_emb=1,
        scale_depth=1.4,
        dim_model_base=256,
    ):
        super().__init__(parent=parent)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scale_emb = scale_emb
        self.scale_depth = scale_depth
        self.dim_model_base = dim_model_base


@require_torch
class MiniCPM3ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = MiniCPM3ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = MiniCPM3ForCausalLM if is_torch_available() else None

    @unittest.skip("MiniCPM3 uses MLA attention which is incompatible with this test")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, Cache)

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

    def test_model_rope_scaling_frequencies(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        x = torch.randn(1, dtype=torch.float32, device=torch_device)
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)

        original_rope = MiniCPM3RotaryEmbedding(config=config).to(torch_device)
        original_freqs_cis_short = original_rope(x, position_ids_short)
        original_freqs_cis_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_freqs_cis_short, original_freqs_cis_long[:, :short_input_length, :])

        config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": scaling_factor}
        linear_scaling_rope = MiniCPM3RotaryEmbedding(config=config).to(torch_device)
        linear_freqs_cis_short = linear_scaling_rope(x, position_ids_short)
        linear_freqs_cis_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_freqs_cis_short, linear_freqs_cis_long[:, :short_input_length, :])

        config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": scaling_factor}
        ntk_scaling_rope = MiniCPM3RotaryEmbedding(config=config).to(torch_device)
        ntk_freqs_cis_short = ntk_scaling_rope(x, position_ids_short)
        ntk_freqs_cis_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_freqs_cis_long, original_freqs_cis_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        config.rope_parameters = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": scaling_factor}
        yarn_scaling_rope = MiniCPM3RotaryEmbedding(config=config).to(torch_device)
        yarn_freqs_cis_short = yarn_scaling_rope(x, position_ids_short)
        yarn_freqs_cis_long = yarn_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(yarn_freqs_cis_short, yarn_freqs_cis_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_long, original_freqs_cis_long)

    def test_tp_plan_matches_params(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        if config.q_lora_rank is not None:
            config.base_model_tp_plan.pop("layers.*.self_attn.q_proj")
        super().test_tp_plan_matches_params()
        config.base_model_tp_plan.update({"layers.*.self_attn.q_proj": "colwise"})


@slow
@require_torch_accelerator
class MiniCPM3IntegrationTest(unittest.TestCase):
    pass
