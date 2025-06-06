# Copyright 2025 The BitNet team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch BitNet model."""

import gc
import unittest

import pytest

from transformers import AutoTokenizer, BitNetConfig, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        BitNetForCausalLM,
        BitNetModel,
    )


class BitNetModelTester(CausalLMModelTester):
    config_class = BitNetConfig
    if is_torch_available():
        base_model_class = BitNetModel
        causal_lm_class = BitNetForCausalLM


@require_torch
class BitNetModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            BitNetModel,
            BitNetForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": BitNetModel,
            "text-generation": BitNetForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = BitNetModelTester

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="BitNet flash attention does not support right padding")


@require_torch
class BitNetIntegrationTest(unittest.TestCase):
    @slow
    def test_model_logits(self):
        input_ids = [128000, 128000, 1502, 25, 2650, 527, 499, 30, 128009, 72803, 25, 220]
        model = BitNetForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor(
            [
                [
                    -1.8665,
                    -1.7681,
                    -1.7043,
                    3.7446,
                    2.7730,
                    4.7133,
                    0.9768,
                    -3.5018,
                    -12.2812,
                    -8.1477,
                    -10.2571,
                    -8.7610,
                ]
            ]
        )
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([5.5815, 4.9154, 1.0478, 4.3869, 3.0112, 0.8235, 3.8412, 2.9233, 8.1140, 1.9406, 1.7973, 10.5025, 4.7796, 8.5926, 4.5196, 3.1549, 3.2656, 3.2588, 2.7356, 2.6032, 2.1454, 1.5683, 1.3465, 1.5329, 1.1886, 7.7902, 5.9326, 1.4737, 3.3319, 1.6291])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_generation(self):
        EXPECTED_TEXT_COMPLETION = """User: What is your favourite food?Assistant: As an AI, I don't have personal preferences or the ability to eat food. However, I"""
        tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is your favourite food?"}], add_generation_prompt=True, tokenize=False
        )
        model = BitNetForCausalLM.from_pretrained(
            "microsoft/bitnet-b1.58-2B-4T", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()
