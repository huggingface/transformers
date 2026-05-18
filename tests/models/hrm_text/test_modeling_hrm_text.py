# Copyright 2026 The Sapient AI Authors and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HRM-Text model."""

import copy
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

    from transformers import (
        AutoTokenizer,
        HrmTextForCausalLM,
        HrmTextModel,
    )


class HrmTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HrmTextModel

    def __init__(
        self,
        parent,
        prefix_lm=False,
    ):
        super().__init__(parent=parent)
        # False default to enable FA
        self.prefix_lm = prefix_lm


@require_torch
class HrmTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HrmTextModelTester

    # z_L_init does not have any gradients
    test_all_params_have_gradient = False


    @unittest.skip(reason="Higher tols (likely due to different recursion and grad patterns). FIXME later")
    def test_tp_generation_quantized(self):
        pass

    @unittest.skip(reason="Higher tols (likely due to different recursion and grad patterns). FIXME later")
    def test_tp_forward(self):
        pass

    @unittest.skip(reason="Higher tols (likely due to different recursion and grad patterns). FIXME later")
    def test_tp_generation(self):
        pass

    @unittest.skip(reason="Low cycle iterations can have non-grad steps")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_prefix_lm_forward(self):
        """`config.prefix_lm=True` with `token_type_ids` produces a different forward pass than
        the pure-causal default. Guards the PrefixLM mask path that the slow integration tests
        also exercise."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # prefix input
        config.prefix_lm = True
        input_ids = inputs_dict["input_ids"]
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, : input_ids.shape[1] // 2] = 1  # first half is bidirectional prefix

        model = HrmTextForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            causal_logits = model(input_ids, use_cache=False).logits
            prefix_logits = model(input_ids, token_type_ids=token_type_ids, use_cache=False).logits

        self.assertGreater(
            (causal_logits - prefix_logits).abs().max().item(),
            1e-4,
            "PrefixLM logits should differ from causal-only logits when token_type_ids marks a prefix region.",
        )

    def test_flash_attention_rejected_when_prefix_lm(self):
        """`config.prefix_lm=True` + FlashAttention must raise at attention-implementation
        resolution time — FA cannot represent the PrefixLM 4-D mask overlay."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.prefix_lm = True
        model = HrmTextForCausalLM(config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            # 3 different checks -> directly from pretrained, set attn implementation, and on setting directly on config
            with self.assertRaises(ValueError) as ctx:
                model = HrmTextForCausalLM.from_pretrained(tmpdirname, attn_implementation="flash_attention_2")
            with self.assertRaises(ValueError) as ctx:
                model = HrmTextForCausalLM.from_pretrained(tmpdirname)
                model.set_attn_implementation("flash_attention_2")
            with self.assertRaises(ValueError) as ctx:
                model.config._attn_implementation = "flash_attention_2"

            self.assertIn("PrefixLM", str(ctx.exception))

    def test_attention_outputs(self):
        """
        Overriden to account for the proper number of hidden layers that are adjusted
        in the post init of the config.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"

        seq_len = getattr(self.model_tester, "seq_length", None)
        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), config.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            self._set_subconfig_attributes(config, "output_attentions", True)
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), config.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), config.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        """
        Overriden to account for the proper number of hidden layers that are adjusted
        in the post init of the config.
        """

        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = config.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            self._set_subconfig_attributes(config, "output_hidden_states", True)
            check_hidden_states_output(inputs_dict, config, model_class)


# TODO(vasqu) add/unblock integration tests
@unittest.skip(reason="not released yet")
@require_torch_accelerator
class HrmTextIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        self.model_id = "sapientinc/HRM-Text-1B"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_greedy_generation(self):
        expected_texts = Expectations(
            {
                ("cuda", 9): "The capital of France isParis",
            }
        )
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = HrmTextForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        input_text = ["<|im_start|><|object_ref_start|>The capital of France is<|im_end|>"]
        model_inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=4, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)
