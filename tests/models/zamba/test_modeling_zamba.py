# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Zamba model."""

import unittest

import pytest

from transformers import AutoTokenizer, ZambaConfig, is_torch_available
from transformers.testing_utils import (
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
        ZambaForCausalLM,
        ZambaForSequenceClassification,
        ZambaModel,
    )


class ZambaModelTester(CausalLMModelTester):
    config_class = ZambaConfig
    if is_torch_available():
        base_model_class = ZambaModel
        causal_lm_class = ZambaForCausalLM
        sequence_classification_class = ZambaForSequenceClassification

    def __init__(self, parent, num_hidden_layers=6, attn_layer_period=1, **kwargs):
        super().__init__(parent=parent, num_hidden_layers=num_hidden_layers, **kwargs)
        self.attn_layer_period = attn_layer_period


@require_torch
class ZambaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            ZambaModel,
            ZambaForCausalLM,
            ZambaForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ZambaModel,
            "text-classification": ZambaForSequenceClassification,
            "text-generation": ZambaForCausalLM,
            "zero-shot": ZambaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = ZambaModelTester

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        Mamba block are initialized differently and we tested that in test_initialization
        """
        self.skipTest("Cumbersome and redundant for Zamba")

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        r"""
        Overriding the test_flash_attn_2_inference_padding_right test as the Zamba model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        self.skipTest(reason="Zamba flash attention does not support right padding")

    @unittest.skip("Model has unusual initializations!")
    def test_initialization(self):
        pass

    @unittest.skip("Does not output attentions for all layers")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Zamba does not support weight tying")
    def test_tied_weights_keys(self):
        pass


@require_torch
class ZambaModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    @slow
    def setUpClass(cls):
        model_id = "Zyphra/Zamba-7B-v1"
        cls.model = ZambaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_mamba_kernels=False
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @slow
    def test_simple_generate(self):
        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        self.assertEqual(
            output_sentence,
            "<s> Hey how are you doing on this lovely evening? I hope you are all doing well. I am",
        )

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits

        EXPECTED_LOGITS_NO_GRAD = torch.tensor(
            [
                -7.9375,  8.1875,  1.3984, -6.0000, -7.9375, -7.9375, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375,  2.7500, 13.0625, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375
            ]
            , dtype=torch.float32)  # fmt: skip

        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)

    @slow
    def test_simple_batched_generate_with_padding(self):
        self.model.to(torch_device)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.resize_token_embeddings(len(self.tokenizer))

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "Tell me a story"], padding=True, return_tensors="pt"
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(
            output_sentences[0],
            "<s> Hey how are you doing on this lovely evening? I hope you are all doing well. I am",
        )
        self.assertEqual(
            output_sentences[1],
            "[PAD][PAD][PAD][PAD][PAD][PAD]<s> Tell me a story about a time when you were in a difficult situation",
        )

        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

        EXPECTED_LOGITS_NO_GRAD_0 = torch.tensor(
            [
                -7.9375,  8.1250,  1.3594, -6.0000, -7.9375, -7.9375, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375,  2.7344, 13.0625, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375,
                -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375, -7.9375
            ]
            , dtype=torch.float32)  # fmt: skip

        EXPECTED_LOGITS_NO_GRAD_1 = torch.tensor(
            [
               -6.3750,  3.4219,  0.6719, -5.0312, -8.5000, -8.5000, -8.5000, -8.5000,
               -8.5000, -8.5000, -8.5000, -8.5000,  2.0625, 10.3750, -8.5000, -8.5000,
               -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000,
               -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000,
               -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000
            ]
            , dtype=torch.float32)  # fmt: skip

        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits[1, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_1, rtol=1e-3, atol=1e-3)
