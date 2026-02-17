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

import torch
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    FineGrainedFP8Config,
    GlmMoeDsaConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
)


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

    def test_default_mlp_layer_types(self):
        config = GlmMoeDsaConfig(num_hidden_layers=8)
        self.assertEqual(
            config.mlp_layer_types, ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"]
        )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("Won't fix: Blip2 + T5 backbone needs custom input preparation for this test")
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_left_padding_compatibility(
        self,
    ):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_sdpa_padding_matches_padding_free_with_position_ids(
        self,
    ):
        pass

    @unittest.skip("Not sure MoE can pass this + indexer outputs are not deterministic wrt padding")
    def test_training_overfit(
        self,
    ):
        pass

    @require_torch_accelerator
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Qwen2Moe flash attention does not support right padding")

    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    @parameterized.expand([("random",), ("same",)])
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DSA indexer mask shape mismatch with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Requires torch>=2.9.0 for grouped MM")
    def test_eager_matches_batched_and_grouped_inference(self):
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
@slow
class GlmMoeDsaIntegrationTest(unittest.TestCase):
    @unittest.skip("Test requires 2 nodes")
    def test_glm_moe_dsa_fp8_inference(self):
        # TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=ip-26-0-169-86 --master_port=29500
        set_seed(0)  # different ranks need the same seed
        model_id = "zai-org/GLM-5-FP8"

        quantization_config = FineGrainedFP8Config(
            modules_to_not_convert=[
                "model.layers.*.mlp.gate$",
                "model.layers.*.self_attn.indexer.weights_proj$",
                "lm_head",
            ],
            weight_block_size=(128, 128),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            tp_plan="auto",
            attn_implementation="eager",
        )

        prompt = ["Hi, introduce yourself", "The capital of France is known for"]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
            )

        output = tokenizer.decode(outputs, skip_special_tokens=False)
        self.assertqual(
            output,
            [
                "<|endoftext|><|endoftext|><|endoftext|>Hi, introduce yourself!\nI'm a 18 years old boy from Italy and I'm a student",
                "The capital of France is known for its rich history, culture, and the city of the of the of the of",
            ],
        )
