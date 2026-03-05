# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch DeepseekV32 model."""

import tempfile
import unittest

from parameterized import parameterized

from transformers import AutoModel, AutoModelForCausalLM, Cache, DeepseekV32Config, is_torch_available
from transformers.testing_utils import require_accelerate, require_torch, require_torch_accelerator

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION


if is_torch_available():
    import torch

    from transformers import DeepseekV32ForCausalLM, DeepseekV32Model


class DeepseekV32ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV32Model
        causal_lm_class = DeepseekV32ForCausalLM

    def __init__(
        self,
        parent,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        moe_intermediate_size=16,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        index_topk=4,
        index_head_dim=32,
        index_n_heads=4,
    ):
        super().__init__(parent=parent, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads


@require_torch
class DeepseekV32ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekV32ModelTester
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.7, 0.8]

    @unittest.skip("Float8 quantization + TP numerical noise exceeds match threshold")
    def test_tp_generation_quantized(self):
        pass

    def setUp(self):
        super().setUp()
        self.config_tester = ConfigTester(self, config_class=DeepseekV32Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_auto_model_classes(self):
        config = self.model_tester.get_config()
        self.assertIsInstance(AutoModel.from_config(config), DeepseekV32Model)
        self.assertIsInstance(AutoModelForCausalLM.from_config(config), DeepseekV32ForCausalLM)

    def test_official_config_fields(self):
        config = DeepseekV32Config(index_topk=8, moe_layer_freq=1, num_nextn_predict_layers=1, topk_method="noaux_tc")
        self.assertEqual(config.index_topk, 8)
        self.assertEqual(config.moe_layer_freq, 1)
        self.assertEqual(config.num_nextn_predict_layers, 1)
        self.assertEqual(config.topk_method, "noaux_tc")

    def test_official_rope_parameters_accept_ints(self):
        with self.assertNoLogs("transformers.modeling_rope_utils", level="WARNING"):
            config = DeepseekV32Config(
                rope_parameters={
                    "rope_type": "yarn",
                    "rope_theta": 10000.0,
                    "factor": 40,
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "original_max_position_embeddings": 4096,
                }
            )

        self.assertEqual(config.rope_parameters["factor"], 40.0)
        self.assertEqual(config.rope_parameters["beta_fast"], 32.0)
        self.assertEqual(config.rope_parameters["beta_slow"], 1.0)

    @require_accelerate
    def test_disk_offloaded_moe_preloads_gate_and_experts(self):
        config = DeepseekV32Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            q_lora_rank=32,
            kv_lora_rank=32,
            n_routed_experts=8,
            num_local_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=0,
            n_group=2,
            topk_group=1,
        )
        model = DeepseekV32ForCausalLM(config).eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            offloaded_model = DeepseekV32ForCausalLM.from_pretrained(
                tmp_dir,
                device_map={
                    "model.embed_tokens": "cpu",
                    "model.layers.0.input_layernorm": "cpu",
                    "model.layers.0.self_attn": "cpu",
                    "model.layers.0.post_attention_layernorm": "cpu",
                    "model.layers.0.mlp": "disk",
                    "model.norm": "cpu",
                    "lm_head": "cpu",
                },
                offload_folder=f"{tmp_dir}/offload",
                offload_buffers=True,
                low_cpu_mem_usage=True,
            ).eval()

            input_ids = torch.tensor([[1, 2, 3, 4]])
            outputs = offloaded_model(input_ids=input_ids)

        self.assertEqual(outputs.logits.shape, (1, 4, config.vocab_size))

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

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("DSA masking makes this generic eager-vs-sdpa inference test too model-specific")
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_training_overfit(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    @parameterized.expand([("random",), ("same",)])
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with static cache")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with compiled forward")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with compilation")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with static cache")
    def test_generate_with_static_cache(self):
        pass


@require_torch_accelerator
class DeepseekV32IntegrationTest(unittest.TestCase):
    pass
