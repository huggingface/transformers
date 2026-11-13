# Copyright 2026 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Granite Switch model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import GraniteSwitchForCausalLM, GraniteSwitchModel


# Two adapters, whose control tokens sit at the top of the tester's small vocabulary. Picked so the
# routing tests can write a readable prompt.
ADAPTER_TOKEN_IDS = [95, 96]
ADAPTER_SUBSTITUTE_TOKEN_IDS = [10, 11]


class GraniteSwitchModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GraniteSwitchModel

    def __init__(self, parent, num_adapters=2, dual_stream=False, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.num_local_experts = 4
        self.num_experts_per_tok = 2
        self.shared_intermediate_size = self.intermediate_size
        self.num_adapters = num_adapters
        self.dual_stream = dual_stream
        if num_adapters > 0:
            self.adapter_token_ids = ADAPTER_TOKEN_IDS[:num_adapters]
            self.adapter_substitute_token_ids = ADAPTER_SUBSTITUTE_TOKEN_IDS[:num_adapters]
            self.adapter_names = [f"adapter_{index}" for index in range(num_adapters)]
            self.max_lora_rank = 4
            self.adapter_ranks = [4] * num_adapters
        if dual_stream:
            self.cross_stream_rank = 4


@require_torch
class GraniteSwitchModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = GraniteSwitchModelTester
    # `lora_B` is zero-initialized, which is what makes a fresh adapter the identity. The chain rule
    # then puts a zero on every `lora_A` gradient, so an untrained model legitimately has LoRA
    # parameters with no gradient.
    test_all_params_have_gradient = False

    def test_control_token_routes_latest_wins(self):
        """Every token takes the adapter of the most recent control token, and 0 before the first."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        switch = GraniteSwitchModel(config).to(torch_device).eval().switch
        first, second = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, first, 7, 8, second, 9, 9]], device=torch_device)
        with torch.no_grad():
            adapter_indices, _ = switch(input_ids)
        expected = torch.tensor([[0, 1, 1, 1, 2, 2, 2]], device=torch_device)
        torch.testing.assert_close(adapter_indices, expected)

    def test_token_exchange_hides_the_control_token(self):
        """The decoder embeds a substitute id, so a control token is never itself embedded."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        switch = GraniteSwitchModel(config).to(torch_device).eval().switch
        first, second = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, first, 7, 8, second, 9, 9]], device=torch_device)
        with torch.no_grad():
            _, modified_input_ids = switch(input_ids)
        first_substitute, second_substitute = ADAPTER_SUBSTITUTE_TOKEN_IDS
        expected = torch.tensor([[5, first_substitute, 7, 8, second_substitute, 9, 9]], device=torch_device)
        torch.testing.assert_close(modified_input_ids, expected)
        self.assertFalse(any(token in modified_input_ids for token in ADAPTER_TOKEN_IDS))

    def test_routing_a_prefix_agrees_with_routing_the_whole_sequence(self):
        """The switch is stateless, so a decode step can be routed by replaying the prefix."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        switch = GraniteSwitchModel(config).to(torch_device).eval().switch
        first, second = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, first, 7, 8, second, 9, 9]], device=torch_device)
        with torch.no_grad():
            whole_sequence, _ = switch(input_ids)
            prefix, _ = switch(input_ids[:, :-1])
        torch.testing.assert_close(whole_sequence[:, :-1], prefix)

    def test_generation_hands_the_switch_the_whole_sequence(self):
        """`prepare_inputs_for_generation` trims `input_ids` for the decoder but not for routing."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSwitchForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, ADAPTER_TOKEN_IDS[0], 7, 8]], device=torch_device)
        model_inputs = model.prepare_inputs_for_generation(input_ids, next_sequence_length=1)
        self.assertEqual(model_inputs["input_ids"].shape, (1, 1))
        torch.testing.assert_close(model_inputs["switch_input_ids"], input_ids)

    def test_inputs_embeds_matches_input_ids(self):
        """Embeddings stand in for ids exactly when token exchange has nothing to do.

        Two things in this model read ids rather than hidden states. Routing can be handed the ids
        separately, as `switch_input_ids`, so it is not what limits the embedding path. Token
        exchange is: it rewrites a control token before the embedding lookup, which cannot be
        applied to an embedding the caller has already computed. The two paths therefore agree
        exactly on a sequence with no control tokens in it, and control tokens are the whole of
        what they may disagree on -- which is what this checks, by clearing them first.
        """
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSwitchModel(config).to(torch_device).eval()

        input_ids = inputs["input_ids"]
        for control_token_id in ADAPTER_TOKEN_IDS:
            input_ids = torch.where(input_ids == control_token_id, torch.ones_like(input_ids), input_ids)

        with torch.no_grad():
            from_ids = model(input_ids=input_ids, attention_mask=inputs.get("attention_mask"))
            from_embeds = model(
                inputs_embeds=model.get_input_embeddings()(input_ids),
                attention_mask=inputs.get("attention_mask"),
                switch_input_ids=input_ids,
            )
        torch.testing.assert_close(from_ids.last_hidden_state, from_embeds.last_hidden_state, rtol=0, atol=0)

    def test_adapters_start_as_the_identity(self):
        """`lora_B` is zero-initialized, so a freshly built model is still the base model."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSwitchModel(config)
        lora_parameters = {name: parameter for name, parameter in model.named_parameters() if "lora_" in name}
        self.assertTrue(lora_parameters)
        for name, parameter in lora_parameters.items():
            if "lora_B" in name:
                self.assertTrue(torch.all(parameter == 0.0), f"{name} should be zero-initialized")
            else:
                self.assertTrue(torch.any(parameter != 0.0), f"{name} should not be zero-initialized")

    def test_no_adapters_builds_a_plain_granite_model(self):
        """`num_adapters=0` skips the switch entirely and leaves every projection a plain Linear."""
        tester = GraniteSwitchModelTester(self, num_adapters=0)
        config, inputs = tester.prepare_config_and_inputs_for_common()
        model = GraniteSwitchModel(config).to(torch_device).eval()
        self.assertIsNone(model.switch)
        self.assertEqual(len(model.layers), config.num_hidden_layers)
        self.assertFalse(any("lora_" in name for name, _ in model.named_parameters()))
        with torch.no_grad():
            model(**inputs)

    def test_shadow_residual_starts_equal_to_the_base_model(self):
        """`cross_stream` is the only path between the streams, and it is zero-initialized."""
        tester = GraniteSwitchModelTester(self, dual_stream=True)
        config, inputs = tester.prepare_config_and_inputs_for_common()
        dual_stream_model = GraniteSwitchForCausalLM(config).to(torch_device).eval()
        for layer in dual_stream_model.model.layers:
            self.assertTrue(torch.all(layer.cross_stream.base_layer.weight == 0.0))

        single_stream_config = config.to_dict()
        single_stream_config["dual_stream"] = False
        single_stream_config["cross_stream_rank"] = None
        single_stream_model = GraniteSwitchForCausalLM(type(config)(**single_stream_config))
        single_stream_model.load_state_dict(
            {
                name: parameter
                for name, parameter in dual_stream_model.state_dict().items()
                if "cross_stream" not in name
            }
        )
        single_stream_model = single_stream_model.to(torch_device).eval()

        with torch.no_grad():
            dual_stream_logits = dual_stream_model(**inputs).logits
            single_stream_logits = single_stream_model(**inputs).logits
        torch.testing.assert_close(dual_stream_logits, single_stream_logits, rtol=0, atol=0)
