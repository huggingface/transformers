# Copyright 2024 JetMoe AI and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch JetMoe model."""

import gc
import unittest

import pytest

from transformers import AutoTokenizer, JetMoeConfig, is_torch_available
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
        JetMoeForCausalLM,
        JetMoeForSequenceClassification,
        JetMoeModel,
    )


class JetMoeModelTester(CausalLMModelTester):
    config_class = JetMoeConfig
    forced_config_args = ["pad_token_id"]
    if is_torch_available():
        base_model_class = JetMoeModel
        causal_lm_class = JetMoeForCausalLM
        sequence_class = JetMoeForSequenceClassification
    pipeline_model_mapping = (
        {
            "feature-extraction": JetMoeModel,
            "text-classification": JetMoeForSequenceClassification,
            "text-generation": JetMoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def __init__(self, parent, hidden_act="silu", kv_channels=8, **kwargs):
        super().__init__(parent, hidden_act=hidden_act, **kwargs)
        self.kv_channels = kv_channels


@require_torch
class JetMoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (JetMoeModel, JetMoeForCausalLM, JetMoeForSequenceClassification) if is_torch_available() else ()
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = JetMoeModelTester

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="JetMoe flash attention does not support right padding")


@require_torch
class JetMoeIntegrationTest(unittest.TestCase):
    @slow
    def test_model_8b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = JetMoeForCausalLM.from_pretrained("jetmoe/jetmoe-8b")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[0.2507, -2.7073, -1.3445, -1.9363, -1.7216, -1.7370, -1.9054, -1.9792]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-3.3689,  5.9006,  5.7450, -1.7012, -4.7072, -4.7071, -4.7071, -4.7071, -4.7072, -4.7072, -4.7072, -4.7071,  3.8321,  9.1746, -4.7071, -4.7072, -4.7071, -4.7072, -4.7071, -4.7072, -4.7071, -4.7071, -4.7071, -4.7071, -4.7071, -4.7071, -4.7071, -4.7071, -4.7071, -4.7071])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_8b_generation(self):
        EXPECTED_TEXT_COMPLETION = """My favourite condiment is ....\nI love ketchup. I love"""
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", use_fast=False)
        model = JetMoeForCausalLM.from_pretrained("jetmoe/jetmoe-8b")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=10, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_8b_batched_generation(self):
        EXPECTED_TEXT_COMPLETION = [
            """My favourite condiment is ....\nI love ketchup. I love""",
            """My favourite 2018 Christmas present was a new pair""",
        ]
        prompt = [
            "My favourite condiment is ",
            "My favourite ",
        ]
        tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", use_fast=False)
        model = JetMoeForCausalLM.from_pretrained("jetmoe/jetmoe-8b")
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(model.model.embed_tokens.weight.device)
        print(input_ids)

        # greedy generation outputs
        generated_ids = model.generate(**input_ids, max_new_tokens=10, temperature=0)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(text)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()
