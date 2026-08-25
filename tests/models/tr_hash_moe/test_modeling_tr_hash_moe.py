# Copyright 2026 The Complexity-ML team and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch TR-HASH model."""

import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, TRHashForCausalLM, TRHashModel


class TRHashModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = TRHashModel

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent=parent,
            batch_size=4,
            seq_length=7,
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            shared_intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
            num_experts=4,
            num_experts_per_tok=2,
            route_hash_count=2,
            routed_output_scale=1.0,
            use_cache=True,
            **kwargs,
        )
        self.tie_word_embeddings = True


@require_torch
class TRHashModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = TRHashModelTester
    test_all_params_have_gradient = False
    _torch_compile_train_cls = TRHashForCausalLM if is_torch_available() else None

    @pytest.mark.generate
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip(
        "TR-HASH requires real token IDs for deterministic expert routing, so generation from inputs_embeds "
        "alone is unsupported."
    )
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass

    @unittest.skip(
        reason="TR-HASH requires real token IDs for deterministic expert routing, so generation cannot continue "
        "from inputs_embeds alone."
    )
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="TR-HASH cannot reconstruct deterministic expert routes from arbitrary input embeddings.")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip(reason="TR-HASH cannot reconstruct deterministic expert routes from arbitrary input embeddings.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    def test_multi_hash_route_tables_match_reference(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = TRHashModel(config)

        self.assertEqual(
            model.layers[0].mlp.router.route_table[:, :8].tolist(),
            [[0, 2, 1, 3, 1, 3, 0, 3], [3, 0, 3, 1, 2, 0, 3, 2]],
        )
        self.assertEqual(
            model.layers[1].mlp.router.route_table[:, :8].tolist(),
            [[1, 3, 0, 2, 2, 0, 3, 3], [0, 1, 2, 0, 3, 3, 2, 0]],
        )

    def test_route_table_controls_expert_selection(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TRHashForCausalLM(config).to(torch_device).eval()
        input_ids = input_dict["input_ids"].to(torch_device)

        with torch.no_grad():
            original = model(input_ids, use_cache=False).logits
            for layer in model.model.layers:
                layer.mlp.router.route_table[0].fill_(0)
                layer.mlp.router.route_table[1].fill_(1)
            rerouted = model(input_ids, use_cache=False).logits

        self.assertFalse(torch.equal(original, rerouted))

    def test_multi_token_cache_matches_full_forward(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TRHashForCausalLM(config).to(torch_device).eval()
        input_ids = input_dict["input_ids"][:1].to(torch_device)

        with torch.no_grad():
            full = model(input_ids, use_cache=False).logits
            prefix = model(input_ids[:, :3], use_cache=True)
            continuation = model(
                input_ids[:, 3:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
            ).logits

        torch.testing.assert_close(continuation, full[:, 3:], rtol=1e-5, atol=1e-5)

    def test_persisted_routing_metadata_roundtrip(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = TRHashForCausalLM(config)
        expected_table = model.model.layers[0].mlp.router.route_table.clone()

        with tempfile.TemporaryDirectory() as temporary_directory:
            model.save_pretrained(temporary_directory)
            reloaded = AutoModelForCausalLM.from_pretrained(temporary_directory)

        torch.testing.assert_close(reloaded.model.layers[0].mlp.router.route_table, expected_table)

    def test_config_does_not_expose_single_value_architecture_flags(self):
        config_class = self.model_tester.config_class
        config = config_class(
            attention_type="gqa",
            mlp_type="tr_hash_engine",
            norm_type="rmsnorm",
            routing_strategy="token_id_multi_hash",
            shared_expert=True,
            top_k=2,
            norm_eps=1e-5,
            rope_theta=500000.0,
            rope_type="standard",
            use_qk_norm=True,
        )

        for attribute in (
            "attention_type",
            "mlp_type",
            "norm_type",
            "routing_strategy",
            "shared_expert",
            "top_k",
            "norm_eps",
            "rope_theta",
            "rope_type",
            "use_qk_norm",
        ):
            self.assertFalse(hasattr(config, attribute))
        self.assertEqual(config.rms_norm_eps, 1e-5)
        self.assertEqual(config.rope_parameters["rope_theta"], 500000.0)

    def test_architecture_top_k_does_not_override_generation_top_k(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = TRHashForCausalLM(config)

        self.assertFalse(hasattr(config, "top_k"))
        self.assertNotEqual(model.generation_config.top_k, config.num_experts_per_tok)


@require_torch
class TRHashIntegrationTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT",
            revision="047e290291eac6e2543f1074c44f9ec5deef8c33",
            dtype=torch.float32,
            output_loading_info=True,
        )
        model = model.to(torch_device)
        self.assertEqual(loading_info["missing_keys"], set())
        self.assertEqual(loading_info["unexpected_keys"], set())
        self.assertEqual(loading_info["mismatched_keys"], set())
        self.assertEqual(model.model.layers[0].mlp.experts.gate_up_proj.shape, (4, 128, 896))
        self.assertEqual(model.model.layers[0].mlp.experts.down_proj.shape, (4, 896, 64))
        input_ids = torch.tensor([[2, 101, 2024, 17, 23, 31999, 7, 0]], device=torch_device)

        with torch.no_grad():
            predicted_tokens = model(input_ids, use_cache=False).logits.argmax(dim=-1)

        self.assertEqual(predicted_tokens.tolist(), [[13825, 12, 265, 202, 17, 17, 7, 224]])
