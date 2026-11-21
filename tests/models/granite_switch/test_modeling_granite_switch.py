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

import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, GraniteSwitchConfig, GraniteSwitchForCausalLM, GraniteSwitchModel
    from transformers.models.granite_switch.configuration_granite_switch import SWITCH_LAYER_SLOTS
    from transformers.models.granite_switch.modeling_granite_switch import route_control_tokens


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

    def get_config(self):
        # The tester's `num_hidden_layers` is the decoder layer count the common tests check, but the
        # config takes the on-disk count, which includes the switch's layer slots when there are adapters.
        kwargs = {
            key: getattr(self, key)
            for key in self.config_args + self.forced_config_args
            if key != "self" and hasattr(self, key)
        }
        if self.num_adapters > 0:
            kwargs["num_hidden_layers"] = self.num_hidden_layers + SWITCH_LAYER_SLOTS
            kwargs["layer_types"] = ["full_attention"] * SWITCH_LAYER_SLOTS + self.layer_types
        return self.config_class(**kwargs)


@require_torch
class GraniteSwitchModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = GraniteSwitchModelTester
    # `lora_B` is zero-initialized, which is what makes a fresh adapter the identity. The chain rule
    # then puts a zero on every `lora_A` gradient, so an untrained model legitimately has LoRA
    # parameters with no gradient.
    test_all_params_have_gradient = False

    def _route(self, input_ids):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSwitchModel(config).to(torch_device)
        return route_control_tokens(input_ids, model.control_token_ids, model.substitute_token_ids)

    def test_control_token_routes_latest_wins(self):
        """Every token takes the adapter of the most recent control token, and 0 before the first."""
        first, second = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, first, 7, 8, second, 9, 9]], device=torch_device)
        adapter_indices, _ = self._route(input_ids)
        expected = torch.tensor([[0, 1, 1, 1, 2, 2, 2]], device=torch_device)
        torch.testing.assert_close(adapter_indices, expected)

    def test_config_counts_the_switch_layer_slots_on_disk(self):
        """With adapters, `config.json` counts the switch's layer slots; the model builds the rest."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        num_decoder_layers = self.model_tester.num_hidden_layers
        self.assertEqual(config.num_hidden_layers, num_decoder_layers)
        self.assertEqual(len(config.layer_types), num_decoder_layers)
        self.assertEqual(len(GraniteSwitchModel(config).layers), num_decoder_layers)

        on_disk = config.to_dict()
        self.assertEqual(on_disk["num_hidden_layers"], num_decoder_layers + SWITCH_LAYER_SLOTS)
        self.assertEqual(len(on_disk["layer_types"]), num_decoder_layers + SWITCH_LAYER_SLOTS)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded = GraniteSwitchConfig.from_pretrained(tmp_dir)
            from_json = GraniteSwitchConfig.from_json_file(f"{tmp_dir}/config.json")
        for loaded in (reloaded, from_json, GraniteSwitchConfig(**on_disk)):
            self.assertEqual(loaded.num_hidden_layers, num_decoder_layers)
            self.assertEqual(loaded.layer_types, config.layer_types)

        # Without adapters there is no switch, so nothing is counted.
        config_dict = {**on_disk, "num_adapters": 0, "num_hidden_layers": num_decoder_layers}
        for key in ("adapter_token_ids", "adapter_substitute_token_ids", "adapter_names", "adapter_ranks"):
            config_dict[key] = None
        config_dict.update(lora_target_modules=None, layer_types=None)
        plain = GraniteSwitchConfig(**config_dict)
        self.assertEqual(plain.num_hidden_layers, num_decoder_layers)
        self.assertEqual(plain.to_dict()["num_hidden_layers"], num_decoder_layers)

    def test_config_rejects_legacy_checkpoints(self):
        """SingleSwitch and unfused Shadow Residual checkpoints raise instead of loading wrong."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        on_disk = config.to_dict()
        with self.assertRaisesRegex(ValueError, "SingleSwitch"):
            GraniteSwitchConfig.from_dict({**on_disk, "switch_type": "single"})
        # A stale `switch_type: "multi"` is accepted and dropped.
        self.assertFalse(hasattr(GraniteSwitchConfig.from_dict({**on_disk, "switch_type": "multi"}), "switch_type"))
        with self.assertRaisesRegex(ValueError, "unfused_qkv"):
            GraniteSwitchConfig.from_dict({**on_disk, "unfused_qkv": True})

    def test_token_exchange_hides_the_control_token(self):
        """The decoder embeds a substitute id, so a control token is never itself embedded."""
        first, second = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, first, 7, 8, second, 9, 9]], device=torch_device)
        _, modified_input_ids = self._route(input_ids)
        first_substitute, second_substitute = ADAPTER_SUBSTITUTE_TOKEN_IDS
        expected = torch.tensor([[5, first_substitute, 7, 8, second_substitute, 9, 9]], device=torch_device)
        torch.testing.assert_close(modified_input_ids, expected)
        self.assertFalse(any(token in modified_input_ids for token in ADAPTER_TOKEN_IDS))

    def test_routing_a_prefix_agrees_with_routing_the_whole_sequence(self):
        """Routing is stateless, so a decode step can be routed by replaying the prefix."""
        first, second = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, first, 7, 8, second, 9, 9]], device=torch_device)
        whole_sequence, _ = self._route(input_ids)
        prefix, _ = self._route(input_ids[:, :-1])
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
        """`num_adapters=0` skips routing entirely and leaves every projection a plain Linear."""
        tester = GraniteSwitchModelTester(self, num_adapters=0)
        config, inputs = tester.prepare_config_and_inputs_for_common()
        model = GraniteSwitchModel(config).to(torch_device).eval()
        self.assertIsNone(model.control_token_ids)
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


@slow
@require_torch_accelerator
class GraniteSwitchIntegrationTest(unittest.TestCase):
    model_id = "ibm-granite/granite-switch-4.1-3b-preview"
    messages = [{"role": "user", "content": "When was the Eiffel Tower completed?"}]
    documents = [{"doc_id": "1", "text": "The Eiffel Tower was completed in 1889 for the World's Fair."}]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.model = GraniteSwitchForCausalLM.from_pretrained(
            cls.model_id, device_map=torch_device, dtype=torch.bfloat16
        )

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def _inputs(self, adapter_name=None):
        kwargs = {"adapter_name": adapter_name} if adapter_name else {}
        return self.tokenizer.apply_chat_template(
            self.messages,
            documents=self.documents,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **kwargs,
        ).to(torch_device)

    def test_model_logits_bf16(self):
        """The last positions' logits, with the adapter and without. They agree until the control token."""
        # fmt: off
        EXPECTED_MEANS = {
            "answerability": Expectations({
                ("mps", None): torch.tensor([0.7986, -0.6247, -3.5927, -2.3329, -4.7504]),
                ("cuda", 8): torch.tensor([0.7162, -0.7031, -3.6584, -2.3316, -4.7810]),
            }),
            None: Expectations({
                ("mps", None): torch.tensor([0.7986, -0.6247, 1.1428, 0.6827, -0.9518]),
                ("cuda", 8): torch.tensor([0.7162, -0.7031, 1.1583, 0.6546, -0.9134]),
            }),
        }
        EXPECTED_SLICES = {
            "answerability": Expectations({
                ("mps", None): torch.tensor([-1.3828, 34.5000, 4.4375, 4.1562, 3.0781, 1.0781, 18.2500, 6.3750, 6.9375, 4.3438, 5.9375, 4.6875, 7.0000, -5.7500, 2.2031]),
                ("cuda", 8): torch.tensor([-1.1484, 34.7500, 4.6562, 4.2812, 3.2188, 1.2656, 18.6250, 6.5938, 7.0625, 4.5938, 6.1250, 4.6875, 7.1875, -5.5938, 2.3125]),
            }),
            None: Expectations({
                ("mps", None): torch.tensor([9.0625, 24.0000, 18.0000, 17.7500, 15.6250, 13.3125, 20.0000, 21.3750, 12.8750, 20.8750, 15.6250, 11.8750, 21.7500, 13.8750, 16.7500]),
                ("cuda", 8): torch.tensor([9.0625, 24.1250, 18.0000, 17.7500, 15.6250, 13.3125, 19.8750, 21.3750, 12.8750, 21.0000, 15.6250, 12.0000, 21.6250, 13.8750, 16.8750]),
            }),
        }
        # fmt: on
        for adapter_name in ("answerability", None):
            with self.subTest(adapter_name=adapter_name), torch.no_grad():
                logits = self.model(**self._inputs(adapter_name)).logits.float()
                torch.testing.assert_close(
                    logits[0, -5:].mean(-1),
                    EXPECTED_MEANS[adapter_name].get_expectation().to(torch_device),
                    rtol=1e-2,
                    atol=1e-2,
                )
                torch.testing.assert_close(
                    logits[0, -1, :15],
                    EXPECTED_SLICES[adapter_name].get_expectation().to(torch_device),
                    rtol=1e-2,
                    atol=1e-2,
                )

    def test_model_generation(self):
        """The adapter answers in its own format, and the same prompt without it gets the base model's answer."""
        EXPECTED_TEXTS = {
            "answerability": '"answerable"',
            None: "The Eiffel Tower was completed in 1889.",
        }
        for adapter_name in ("answerability", None):
            with self.subTest(adapter_name=adapter_name):
                inputs = self._inputs(adapter_name)
                output = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
                text = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                self.assertEqual(text, EXPECTED_TEXTS[adapter_name])
