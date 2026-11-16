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

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import GraniteSwitchConfig, GraniteSwitchForCausalLM, GraniteSwitchModel
    from transformers.models.granite_switch.configuration_granite_switch import SWITCH_LAYER_SLOTS


# Two adapters, whose control tokens sit at the top of the tester's small vocabulary. Picked so the
# routing tests can write a readable prompt.
ADAPTER_TOKEN_IDS = [95, 96]
ADAPTER_SUBSTITUTE_TOKEN_IDS = [10, 11]
# Label words of the probe slots in the probe tests: three single-token labels.
PROBE_LABEL_TOKEN_IDS = [20, 21, 22]


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
        config_dict.update(adapter_kinds=None, lora_target_modules=None, layer_types=None)
        plain = GraniteSwitchConfig(**config_dict)
        self.assertEqual(plain.num_hidden_layers, num_decoder_layers)
        self.assertEqual(plain.to_dict()["num_hidden_layers"], num_decoder_layers)

    def test_base_reset_token_routes_back_to_base(self):
        """A leading base-reset token sends the tokens after it back to adapter 0."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_reset = 94
        config_dict = config.to_dict()
        config_dict["adapter_token_ids"] = [base_reset] + ADAPTER_TOKEN_IDS
        config_dict["adapter_substitute_token_ids"] = [12] + ADAPTER_SUBSTITUTE_TOKEN_IDS
        switch = GraniteSwitchModel(type(config)(**config_dict)).to(torch_device).eval().switch
        first, second = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, first, 7, base_reset, 8, second, 9, base_reset, 9]], device=torch_device)
        with torch.no_grad():
            adapter_indices, modified_input_ids = switch(input_ids)
        expected = torch.tensor([[0, 1, 1, 0, 0, 2, 2, 0, 0]], device=torch_device)
        torch.testing.assert_close(adapter_indices, expected)
        self.assertFalse(any(token in modified_input_ids for token in [base_reset] + ADAPTER_TOKEN_IDS))

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

    def _build_probe_model(self, adapter_kinds, probe_read_layers=None, dual_stream=False):
        """A causal LM whose slots are `adapter_kinds`, with random probe heads so verdicts differ."""
        tester = GraniteSwitchModelTester(self, dual_stream=dual_stream)
        config, _ = tester.prepare_config_and_inputs_for_common()
        config_dict = config.to_dict()
        config_dict["adapter_kinds"] = adapter_kinds
        config_dict["probe_label_token_ids"] = [
            PROBE_LABEL_TOKEN_IDS if kind == "probe" else None for kind in adapter_kinds
        ]
        config_dict["probe_read_layers"] = probe_read_layers
        model = GraniteSwitchForCausalLM(type(config)(**config_dict))
        torch.manual_seed(0)
        with torch.no_grad():
            model.model.probe_head.weight.normal_()
            model.model.probe_head.bias.normal_()
        return model.to(torch_device).eval()

    def test_probe_head_is_built_only_with_a_probe_slot(self):
        """A checkpoint of LoRA slots alone carries no probe head and no kind table."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        lora_model = GraniteSwitchModel(config)
        self.assertEqual(config.adapter_kinds, ["lora", "lora"])
        self.assertIsNone(lora_model.probe_head)
        self.assertIsNone(lora_model.switch.adapter_kind_lut)

        probe_model = self._build_probe_model(["lora", "probe"])
        self.assertEqual(probe_model.model.probe_head.weight.shape, (2, 3, config.hidden_size))
        torch.testing.assert_close(
            probe_model.model.switch.adapter_kind_lut, torch.tensor([False, False, True], device=torch_device)
        )

    def test_probe_tokens_leave_the_lora_stream(self):
        """A token routed to a probe slot runs the base model; the others keep their adapter."""
        model = self._build_probe_model(["lora", "probe"])
        lora_token, probe_token = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, lora_token, 7, probe_token, 9]], device=torch_device)
        with torch.no_grad():
            adapter_indices, _ = model.model.switch(input_ids)
            lora_indices, probe_indices = model.model.switch.split_indices(adapter_indices)
        torch.testing.assert_close(lora_indices, torch.tensor([[0, 1, 1, 0, 0]], device=torch_device))
        torch.testing.assert_close(probe_indices, torch.tensor([[0, 0, 0, 2, 2]], device=torch_device))

    def test_probe_verdict_is_written_at_the_label_token_ids(self):
        """The control token's logits hold the head's scores at the label ids and nothing else."""
        model = self._build_probe_model(["lora", "probe"])
        input_ids = torch.tensor([[5, 7, 8, ADAPTER_TOKEN_IDS[1]]], device=torch_device)
        with torch.no_grad():
            model_outputs = model.model(input_ids=input_ids)
            logits = model(input_ids=input_ids).logits

        torch.testing.assert_close(model_outputs.probe_indices, torch.tensor([[0, 0, 0, 2]], device=torch_device))
        expected_scores = model.model.probe_head(
            model_outputs.last_hidden_state[0, -1:], torch.tensor([2], device=torch_device)
        )
        torch.testing.assert_close(logits[0, -1, PROBE_LABEL_TOKEN_IDS], expected_scores[0])

        floor = torch.finfo(logits.dtype).min
        is_label = torch.zeros(logits.shape[-1], dtype=torch.bool, device=torch_device)
        is_label[PROBE_LABEL_TOKEN_IDS] = True
        self.assertTrue(torch.all(logits[0, -1, ~is_label] == floor))
        # Every other position is an ordinary language-model row.
        self.assertTrue(torch.all(logits[0, :-1] > floor))

    def test_request_without_a_probe_token_is_untouched(self):
        """Without a probe control token nothing is scored, on a checkpoint that has probe slots."""
        model = self._build_probe_model(["lora", "probe"])
        input_ids = torch.tensor([[5, ADAPTER_TOKEN_IDS[0], 7, 8]], device=torch_device)
        with torch.no_grad():
            model_outputs = model.model(input_ids=input_ids)
            logits = model(input_ids=input_ids).logits
        self.assertIsNone(model_outputs.probe_logits)
        self.assertIsNone(model_outputs.probe_indices)
        self.assertTrue(torch.all(logits > torch.finfo(logits.dtype).min))

    def test_mixed_batch_rewrites_only_the_probe_row(self):
        """Probe and generation requests share a batch; only the probe row's last logits change."""
        model = self._build_probe_model(["lora", "probe"])
        lora_token, probe_token = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, 7, 8, probe_token], [5, lora_token, 7, 8]], device=torch_device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            generation_logits = model(input_ids=input_ids[1:]).logits
        floor = torch.finfo(logits.dtype).min
        self.assertEqual((logits[0, -1] > floor).sum().item(), len(PROBE_LABEL_TOKEN_IDS))
        torch.testing.assert_close(logits[1], generation_logits[0], rtol=1e-5, atol=1e-5)

    def test_probe_token_that_is_not_last_raises(self):
        """The verdict replaces the control token's logits, so nothing may follow that token."""
        model = self._build_probe_model(["lora", "probe"])
        input_ids = torch.tensor([[5, ADAPTER_TOKEN_IDS[1], 7, 8]], device=torch_device)
        with self.assertRaisesRegex(ValueError, "must be the last token"):
            model(input_ids=input_ids)

    def test_probe_token_before_right_padding_is_the_last_token(self):
        """Padding does not count as following the control token, and the verdict stays at it."""
        model = self._build_probe_model(["lora", "probe"])
        input_ids = torch.tensor([[5, 7, ADAPTER_TOKEN_IDS[1], 0]], device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 0]], device=torch_device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        floor = torch.finfo(logits.dtype).min
        self.assertEqual((logits[0, 2] > floor).sum().item(), len(PROBE_LABEL_TOKEN_IDS))
        self.assertTrue(torch.all(logits[0, 3] > floor))

    def test_probe_verdict_survives_logits_to_keep(self):
        """Generation keeps only the last position, which is exactly where the verdict is."""
        model = self._build_probe_model(["lora", "probe"])
        input_ids = torch.tensor([[5, 7, 8, ADAPTER_TOKEN_IDS[1]]], device=torch_device)
        with torch.no_grad():
            all_logits = model(input_ids=input_ids).logits
            last_logits = model(input_ids=input_ids, logits_to_keep=1).logits
        torch.testing.assert_close(last_logits[:, -1], all_logits[:, -1])
        with self.assertRaisesRegex(ValueError, "logits_to_keep"):
            model(input_ids=input_ids, logits_to_keep=torch.tensor([0, 1], device=torch_device))

    def test_probe_reads_its_configured_layer(self):
        """Each slot is scored on its own read layer: a decoder layer's output, or the final norm."""
        model = self._build_probe_model(["probe", "probe"], probe_read_layers=[None, 0])
        first_token, second_token = ADAPTER_TOKEN_IDS
        input_ids = torch.tensor([[5, 7, first_token], [5, 7, second_token]], device=torch_device)
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, output_hidden_states=True)

        probe_head = model.model.probe_head
        final_read = probe_head(outputs.last_hidden_state[:1, -1], torch.tensor([1], device=torch_device))
        # `hidden_states[0]` is the embedding output, so layer 0's output is entry 1.
        layer_read = probe_head(outputs.hidden_states[1][1:, -1], torch.tensor([2], device=torch_device))
        torch.testing.assert_close(outputs.probe_logits[:1, -1], final_read)
        torch.testing.assert_close(outputs.probe_logits[1:, -1], layer_read)

        final_read_of_second = probe_head(outputs.last_hidden_state[1:, -1], torch.tensor([2], device=torch_device))
        self.assertFalse(torch.allclose(layer_read, final_read_of_second))

    def test_probe_generates_a_label_word_then_decodes_as_the_base_model(self):
        """The verdict is the first generated token, and decoding carries on from there."""
        model = self._build_probe_model(["lora", "probe"])
        input_ids = torch.tensor([[5, 7, 8, ADAPTER_TOKEN_IDS[1]]], device=torch_device)
        generated = model.generate(input_ids, max_new_tokens=3, do_sample=False)
        self.assertEqual(generated.shape, (1, 7))
        self.assertIn(generated[0, 4].item(), PROBE_LABEL_TOKEN_IDS)

        # The decode step after the verdict is an ordinary row: nothing is scored there.
        with torch.no_grad():
            step = model.model(input_ids=generated[:, 4:5], switch_input_ids=generated[:, :5])
        self.assertIsNone(step.probe_indices)

    def test_probe_on_a_shadow_residual_checkpoint(self):
        """Shadow Residual scores the probe on its adapter stream, the one the LM head reads."""
        model = self._build_probe_model(["lora", "probe"], dual_stream=True)
        input_ids = torch.tensor([[5, 7, 8, ADAPTER_TOKEN_IDS[1]]], device=torch_device)
        with torch.no_grad():
            model_outputs = model.model(input_ids=input_ids)
            logits = model(input_ids=input_ids).logits
        expected_scores = model.model.probe_head(
            model_outputs.last_hidden_state[0, -1:], torch.tensor([2], device=torch_device)
        )
        torch.testing.assert_close(logits[0, -1, PROBE_LABEL_TOKEN_IDS], expected_scores[0])

    def test_probe_checkpoint_round_trips(self):
        """The probe head and the kind table are in the state dict, so a reload scores identically."""
        model = self._build_probe_model(["lora", "probe"], probe_read_layers=[None, 0])
        input_ids = torch.tensor([[5, 7, 8, ADAPTER_TOKEN_IDS[1]]], device=torch_device)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded = GraniteSwitchForCausalLM.from_pretrained(tmp_dir).to(torch_device).eval()
        self.assertEqual(reloaded.config.adapter_kinds, ["lora", "probe"])
        self.assertEqual(reloaded.config.probe_read_layers, [None, 0])
        with torch.no_grad():
            torch.testing.assert_close(reloaded(input_ids=input_ids).logits, model(input_ids=input_ids).logits)

    def test_probe_config_validation(self):
        """A probe slot that is not fully described is a load-time error, not wrong routing."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config_dict = config.to_dict()
        config_dict["adapter_kinds"] = ["lora", "probe"]
        config_dict["probe_label_token_ids"] = [None, PROBE_LABEL_TOKEN_IDS]

        invalid_overrides = [
            {"probe_label_token_ids": None},
            {"probe_label_token_ids": [PROBE_LABEL_TOKEN_IDS, PROBE_LABEL_TOKEN_IDS]},
            {"probe_label_token_ids": [None, []]},
            {"adapter_kinds": ["lora"]},
            {"adapter_kinds": ["lora", "classifier"]},
            {"probe_read_layers": [0, None]},
            {"probe_read_layers": [None, config.num_hidden_layers]},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(StrictDataclassClassValidationError):
                type(config)(**{**config_dict, **overrides})

        # The last decoder layer is a valid read layer: the switch takes no layer slots.
        type(config)(**{**config_dict, "probe_read_layers": [None, config.num_hidden_layers - 1]})

    def test_base_reset_token_does_not_shift_the_probe_slots(self):
        """With a leading base-reset token, slot `i` is still adapter `i + 1`."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config_dict = config.to_dict()
        config_dict["adapter_kinds"] = ["lora", "probe"]
        config_dict["probe_label_token_ids"] = [None, PROBE_LABEL_TOKEN_IDS]
        config_dict["adapter_token_ids"] = [94] + ADAPTER_TOKEN_IDS
        config_dict["adapter_substitute_token_ids"] = [12] + ADAPTER_SUBSTITUTE_TOKEN_IDS
        model = GraniteSwitchModel(type(config)(**config_dict))
        self.assertEqual(model.probe_control_token_ids, [ADAPTER_TOKEN_IDS[1]])
        torch.testing.assert_close(model.switch.adapter_kind_lut, torch.tensor([False, False, True]))
