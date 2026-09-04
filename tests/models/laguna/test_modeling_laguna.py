# Copyright 2026 Poolside and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Laguna model."""

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import Expectations, require_torch, require_torch_accelerator, slow, torch_device


if is_torch_available():
    import torch

    from transformers import (
        LagunaConfig,
        LagunaForCausalLM,
        LagunaModel,
    )
    from transformers.models.laguna.modeling_laguna import LagunaTopKRouter


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class LagunaModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = LagunaModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.vocab_size = 64
        self.head_dim = 8
        self.sliding_window = 32
        self.shared_expert_intermediate_size = 16
        self.mlp_layer_types = ["dense", "sparse"]
        self.layer_types = ["full_attention", "sliding_attention"]


@require_torch
class LagunaModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = LagunaModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    def test_apply_router_weight_on_input_not_supported(self):
        """
        `moe_apply_router_weight_on_input=True` is not supported yet so we explicitly check that it
        raises and error on config construction time
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cfg_kwargs = config.to_dict()
        cfg_kwargs["moe_apply_router_weight_on_input"] = True
        with self.assertRaises(NotImplementedError):
            LagunaConfig(**cfg_kwargs)

    def test_router_score_func_validation(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        self.assertEqual(config.moe_router_score_func, "sigmoid")
        cfg_kwargs = config.to_dict()
        cfg_kwargs["moe_router_score_func"] = "softmax"
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "moe_router_score_func"):
            LagunaConfig(**cfg_kwargs)

    @parameterized.expand([("sigmoid",), ("sqrtsoftplus",)])
    def test_router_score_func(self, score_func):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.moe_router_score_func = score_func
        router = LagunaTopKRouter(config)
        hidden_states = torch.linspace(-1.0, 1.0, 2 * config.hidden_size).view(1, 2, config.hidden_size)

        with torch.no_grad():
            router.weight.copy_(
                torch.linspace(-0.5, 0.5, config.num_experts * config.hidden_size).view_as(router.weight)
            )
            router.e_score_correction_bias.copy_(torch.linspace(0.4, -0.4, config.num_experts))

        router_logits, routing_weights, selected_experts = router(hidden_states)
        expected_scores = (
            torch.sigmoid(router_logits)
            if score_func == "sigmoid"
            else torch.nn.functional.softplus(router_logits).sqrt()
        )
        expected_experts = torch.topk(
            expected_scores + router.e_score_correction_bias, config.num_experts_per_tok, dim=-1
        ).indices
        expected_weights = expected_scores.gather(-1, expected_experts)
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

        torch.testing.assert_close(selected_experts, expected_experts)
        torch.testing.assert_close(routing_weights, expected_weights)

        model = self.model_tester.base_model_class(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(input_ids=inputs_dict["input_ids"].to(torch_device))
        self.assertTrue(torch.isfinite(outputs.last_hidden_state).all())

    @parameterized.expand([(True,), ("per-head",), ("per-element",)])
    def test_gating_variations(self, gating):
        """Checking whether each flavor option is properly propagated"""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.gating = gating
        # We only check the underlying base class for simplicity
        model = self.model_tester.base_model_class(config).to(torch_device).eval()

        for layer in model.layers:
            if gating == "per-element":
                self.assertFalse(layer.self_attn.gate_per_head)
            else:
                self.assertTrue(layer.self_attn.gate_per_head)

            expected_shape = (
                layer.self_attn.num_heads if gating != "per-element" else layer.self_attn.num_heads * config.head_dim
            )
            self.assertEqual(layer.self_attn.g_proj.out_features, expected_shape)

        with torch.no_grad():
            model(input_ids=inputs_dict["input_ids"].to(torch_device))


@slow
@require_torch
@require_torch_accelerator
class LagunaIntegrationTest(unittest.TestCase):
    def test_per_element_gating_logits(self):
        """Logits of a small per-element-gating Laguna checkpoint, batched with padding."""
        model_id = "poolside/Laguna-tiny-per-element"
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = LagunaForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

        expected_left = Expectations(
            {
                ("cuda", 8): [[0.0033, 0.0581, -0.1718], [-0.0559, -0.1834, 0.0085], [-0.0235, -0.0824, -0.0569]],
                ("xpu", 5): [[0.0033, 0.0581, -0.1718], [-0.0559, -0.1834, 0.0085], [-0.0235, -0.0824, -0.0569]],
            }
        )  # fmt: skip
        expected_right = Expectations(
            {
                ("cuda", 8): [[0.0132, -0.0518, -0.1204], [-0.0231, -0.0547, 0.0684], [-0.1406, -0.2664, -0.1904]],
                ("xpu", 5): [[0.0132, -0.0518, -0.1204], [-0.0231, -0.0547, 0.0684], [-0.1406, -0.2664, -0.1904]],
            }
        )  # fmt: skip
        expected_left = torch.tensor(expected_left.get_expectation(), device=torch_device)
        expected_right = torch.tensor(expected_right.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits.float()
        torch.testing.assert_close(logits[0, -3:, -3:], expected_left, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, -3:, -3:], expected_right, atol=1e-3, rtol=1e-3)
