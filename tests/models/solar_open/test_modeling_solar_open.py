# Copyright 2025 Upstage and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SolarOpen model."""

import unittest

import torch

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import AutoTokenizer, SolarOpenConfig, SolarOpenForCausalLM, SolarOpenModel


class SolarOpenModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = SolarOpenModel

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        use_qk_norm=False,
    ):
        super().__init__(parent=parent, num_experts_per_tok=num_experts_per_tok)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm


@require_torch
class SolarOpenModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = SolarOpenModelTester
    model_split_percents = [0.5, 0.85, 0.9]  # it tries to offload everything with the default value


@require_torch_accelerator
@slow
class SolarOpenIntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_batch_generation_dummy_bf16(self):
        """Original model is 100B, hence using a dummy model on our CI to sanity check against"""
        model_id = "SSON9/solar-open-tiny-dummy"
        prompts = [
            "Orange is the new black",
            "Lorem ipsum dolor sit amet",
        ]
        # expected random outputs from the tiny dummy model
        EXPECTED_DECODED_TEXT = [
            "Orange is the new blackRIB yshift yshift catheter merits catheterCCTV meritsCCTVCCTVCCTVCCTVCCTVCCTV SyllabusCCTVCCTVCCTVCCTV Syllabus",
            "Lorem ipsum dolor sit amet=√=√=√ 치수 치수 치수 치수 치수 치수 치수 Shelley Shelley Shelley Shelley Shelley Shelley Shelley Shelley площа площа",
        ]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = SolarOpenForCausalLM.from_pretrained(
            model_id, experts_implementation="eager", device_map=torch_device, torch_dtype=torch.bfloat16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for text, expected_text in zip(generated_texts, EXPECTED_DECODED_TEXT):
            self.assertEqual(text, expected_text)


class SolarOpenConfigTest(unittest.TestCase):
    def test_default_config(self):
        """
        Simple test for SolarOpenConfig default values
        """
        config = SolarOpenConfig()
        self.assertEqual(config.model_type, "solar_open")
        self.assertEqual(config.n_routed_experts, 128)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.n_group, 1)
        self.assertEqual(config.topk_group, 1)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.moe_intermediate_size, 1280)
        self.assertEqual(config.routed_scaling_factor, 1.0)
        self.assertTrue(config.norm_topk_prob)
        self.assertFalse(config.use_qk_norm)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 48)
        self.assertEqual(config.num_attention_heads, 64)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.vocab_size, 196608)
        self.assertEqual(config.rms_norm_eps, 1e-05)
        self.assertEqual(config.max_position_embeddings, 131072)
        self.assertEqual(config.partial_rotary_factor, 1.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.rope_parameters["original_max_position_embeddings"], 65_536)
        self.assertEqual(config.tie_word_embeddings, False)

    def test_rope_parameters_partially_initialized(self):
        """
        Test for SolarOpenConfig when rope_parameters is partially initialized
        """
        config = SolarOpenConfig(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 65536,
            }
        )
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)

    def test_partial_rotary_factor_bc(self):
        """
        Test for SolarOpenConfig when partial_rotary_factor is set in rope_parameters
        """
        config = SolarOpenConfig(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 65536,
            },
            partial_rotary_factor=0.7,
        )
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.7)

    def test_disable_rope_scaling(self):
        """
        Test for SolarOpenConfig when rope_scaling is disabled
        """
        config = SolarOpenConfig(
            rope_parameters={
                "rope_type": "default",
            }
        )
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertNotIn("factor", config.rope_parameters)
        self.assertNotIn("original_max_position_embeddings", config.rope_parameters)
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)

    def test_rope_backward_compatibility(self):
        """
        Test for SolarOpenConfig backward compatibility for rope_parameters
        """
        config = SolarOpenConfig(
            rope_scaling={"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536},
            partial_rotary_factor=1.0,
            rope_theta=1_000_000,
        )
        self.assertEqual(config.rope_parameters["rope_theta"], 1_000_000)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.rope_parameters["original_max_position_embeddings"], 65_536)
