# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LFM3 model."""

import unittest

from transformers import AutoTokenizer, Lfm3ForCausalLM, Lfm3Model, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch


class Lfm3ModelTester(CausalLMModelTester):
    base_model_class = Lfm3Model

    def __init__(self, parent):
        super().__init__(parent)
        self.layer_types = ["full_attention", "sliding_attention", "conv"]
        self.num_hidden_layers = len(self.layer_types)
        self.conv_L_cache = 7


@require_torch
class Lfm3ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Lfm3ModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, config.hidden_size, config.conv_L_cache)

    def test_attention_outputs(self):
        """LFM3 records attentions for the full and sliding attention layers, not for the conv layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)
        num_attention_layers = sum(layer_type != "conv" for layer_type in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            config = model.config
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.attentions), num_attention_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(outputs.attentions), num_attention_layers)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len]
            )

    def test_full_attention_reaches_beyond_the_window_for_any_layer_order(self):
        """A sliding layer ahead of the first full-attention layer must not shrink the full-attention mask."""
        for layer_types in (
            ["conv", "sliding_attention", "full_attention"],
            ["sliding_attention", "conv", "full_attention"],
        ):
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            config.layer_types, config.sliding_window, config.conv_L_cache = layer_types, 4, 3
            # eager materializes the masks; sdpa skips an unpadded causal mask and would hide a wrongly sized one
            model = Lfm3ForCausalLM._from_config(config, attn_implementation="eager").to(torch_device).eval()
            input_ids = ids_tensor((1, 17), config.vocab_size).to(torch_device)
            altered = input_ids.clone()
            altered[0, 0] = (input_ids[0, 0] + 1) % config.vocab_size
            with torch.no_grad():
                last = model(input_ids=input_ids, use_cache=False).logits[:, -1]
                altered_last = model(input_ids=altered, use_cache=False).logits[:, -1]
                cached = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=model(input_ids=input_ids[:, :-1], use_cache=True).past_key_values,
                ).logits[:, -1]
            # the first token is far outside the sliding window and the conv taps: only full attention can see it
            self.assertFalse(torch.allclose(last, altered_last, rtol=1e-4, atol=1e-4), layer_types)
            torch.testing.assert_close(cached, last, rtol=1e-4, atol=1e-4)

    def test_static_cache_generation_matches_dynamic_for_any_layer_order(self):
        """Static-cache masks are sized by layer type, also for left-padded rows."""
        for layer_types in (
            ["conv", "sliding_attention", "full_attention"],
            ["sliding_attention", "conv", "full_attention"],
        ):
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            config.layer_types, config.sliding_window, config.conv_L_cache = layer_types, 4, 3
            model = Lfm3ForCausalLM(config).to(torch_device).eval()
            input_ids = ids_tensor((2, 9), config.vocab_size).to(torch_device)
            input_ids[1, :3] = config.pad_token_id
            attention_mask = torch.ones_like(input_ids)
            attention_mask[1, :3] = 0
            with torch.no_grad():
                dynamic = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=6, do_sample=False)
                static = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=6,
                    do_sample=False,
                    cache_implementation="static",
                    disable_compile=True,
                )
            self.assertTrue(torch.equal(dynamic, static), layer_types)


@require_torch_accelerator
@slow
class Lfm3IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "LiquidAI/LFM3-1.7B-Preview"

    def test_model_logits(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = Lfm3ForCausalLM.from_pretrained(self.model_id, device_map=torch_device, dtype=torch.float32)
        input_ids = tokenizer("Simply put, the theory of relativity states that ", return_tensors="pt").to(
            torch_device
        )
        with torch.no_grad():
            logits = model(**input_ids).logits.float()

        expectations = Expectations(
            {
                (None, None): [
                    -1.9214,
                    -1.6632,
                    -1.0323,
                    -1.0694,
                    0.0741,
                    0.2619,
                    0.3455,
                    0.2205,
                    0.2545,
                    0.1654,
                    -1.9167,
                ]
            }
        )
        expected_mean = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0].mean(-1), expected_mean, rtol=1e-3, atol=1e-3)
        expectations = Expectations(
            {
                (None, None): [
                    -0.0203,
                    -2.5232,
                    -0.4502,
                    -1.087,
                    1.0517,
                    0.6892,
                    0.7451,
                    2.0436,
                    9.0642,
                    5.4245,
                    7.058,
                    3.5806,
                    5.2226,
                    2.1968,
                    0.0952,
                    2.6452,
                ]
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0, 0, :16], expected_slice, rtol=1e-3, atol=1e-3)

    def test_model_greedy_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = Lfm3ForCausalLM.from_pretrained(self.model_id, device_map=torch_device, dtype=torch.bfloat16)
        messages = [{"role": "user", "content": "Simply put, what does the theory of relativity state?"}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(torch_device)
        generated_ids = model.generate(**input_ids, max_new_tokens=32, do_sample=False)
        text = tokenizer.decode(generated_ids[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True)

        expectations = Expectations(
            {
                (None, None): (
                    'We need to answer the question: "Simply put, what does the theory of relativity state?" '
                    "This is a straightforward question about the theory of relativity. The"
                )
            }
        )
        self.assertEqual(text, expectations.get_expectation())
