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
"""Testing suite for the PyTorch Nanbeige model."""

import copy
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


if is_torch_available():
    import torch

    from transformers import NanbeigeForCausalLM, NanbeigeModel


class NanbeigeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = NanbeigeModel

    small_config_overrides = {
        "num_loops": 2,
        # Two layers run twice, and the recorder includes the input embeddings.
        "expected_num_hidden_layers": 2 * 2 + 1,
    }


@require_torch
class NanbeigeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = NanbeigeModelTester

    @unittest.skip("New model: `cache` only, no `past_key_values` BC")
    def test_bc_cache_name_forward(self):
        pass

    @unittest.skip("The shared test expects one attention map per layer, and Nanbeige records every layer call")
    def test_attention_outputs(self):
        pass

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        self.assertEqual(len(cache), config.total_num_hidden_layers)
        num_layers = config.num_hidden_layers
        for start in range(0, len(cache), num_layers):
            loop_cache = copy.copy(cache)
            loop_cache.layers = cache.layers[start : start + num_layers]
            super()._check_cache_for_generate(batch_size, loop_cache, seq_length, config)

    def test_two_loop_chunked_prefill_and_decode_match_full_forward(self):
        tester = self.model_tester_class(self, batch_size=2, num_loops=2)
        config, inputs_dict = tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        model = NanbeigeForCausalLM(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        num_cache_layers = config.num_hidden_layers * config.num_loops

        with torch.no_grad():
            expected_logits = model(input_ids, use_cache=False).logits
            outputs = model(input_ids[:, :2], use_cache=True, output_attentions=True)
            logits = [outputs.logits]
            self.assertEqual(len(outputs.attentions), num_cache_layers)
            self.assertEqual(len(outputs.cache.layers), num_cache_layers)
            for layer_idx in range(num_cache_layers):
                self.assertEqual(outputs.cache.get_seq_length(layer_idx), 2)

            outputs = model(input_ids[:, 2:4], cache=outputs.cache, use_cache=True)
            logits.append(outputs.logits)
            for position in range(4, input_ids.shape[1]):
                outputs = model(input_ids[:, position : position + 1], cache=outputs.cache, use_cache=True)
                logits.append(outputs.logits)

        torch.testing.assert_close(torch.cat(logits, dim=1), expected_logits, rtol=1e-4, atol=1e-4)
        for layer_idx in range(num_cache_layers):
            self.assertEqual(outputs.cache.get_seq_length(layer_idx), input_ids.shape[1])


@slow
@require_torch_accelerator
class NanbeigeIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    checkpoint = "Nanbeige/Nanbeige4.2-3B"

    def test_bf16_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint, dtype=torch.bfloat16, device_map=torch_device)
        self.assertIsInstance(model, NanbeigeForCausalLM)
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Which number is bigger, 9.11 or 9.8?"}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        self.assertEqual(
            tokenizer.decode(new_tokens, skip_special_tokens=True),
            "To determine which number is bigger between **9.11** and **9.8**, we compare them step by",
        )
