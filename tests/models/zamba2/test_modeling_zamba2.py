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
from parameterized import parameterized

from transformers import AutoTokenizer, Zamba2Config, is_torch_available
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
        Zamba2ForCausalLM,
        Zamba2ForSequenceClassification,
        Zamba2Model,
    )


class Zamba2ModelTester(CausalLMModelTester):
    config_class = Zamba2Config
    if is_torch_available():
        base_model_class = Zamba2Model
        causal_lm_class = Zamba2ForCausalLM
        sequence_classification_class = Zamba2ForSequenceClassification


@require_torch
class Zamba2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Zamba2Model,
            Zamba2ForCausalLM,
            Zamba2ForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Zamba2Model,
            "text-classification": Zamba2ForSequenceClassification,
            "text-generation": Zamba2ForCausalLM,
            "zero-shot": Zamba2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = Zamba2ModelTester

    @unittest.skip("position_ids cannot be used to pad due to Mamba2 layers")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="Zamba2 has hybrid cache.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="A large mamba2 would be necessary (and costly) for that")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="Cumbersome and redundant for Zamba2")
    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        Mamba block are initialized differently and we tested that in test_initialization
        """
        pass

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        r"""
        Overriding the test_flash_attn_2_inference_padding_right test as the Zamba2 model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        self.skipTest(reason="Zamba2 flash attention does not support right padding")

    @unittest.skip(reason="Zamba2 has its own special cache type")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass


@require_torch
class Zamba2ModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    @slow
    def setUpClass(cls):
        model_id = "Zyphra/Zamba2-1.2B"
        cls.model = Zamba2ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, revision="PR"
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="PR")

    @parameterized.expand([(torch_device,), ("cpu",)])
    @slow
    def test_simple_generate(self, torch_device):
        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        self.assertEqual(
            output_sentence,
            "<s> Hey how are you doing on this lovely evening?\n\nI'm doing well, thanks for",
        )

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits.to(dtype=torch.float32)

        EXPECTED_LOGITS_NO_GRAD = torch.tensor(
            [
               -5.9587, 10.5152,  7.0382, -2.8728, -4.8143, -4.8142, -4.8142, -4.8144,
               -4.8143, -4.8143, -4.8142, -4.8142,  6.0185, 18.0037, -4.8142, -4.8144,
               -4.8143, -4.8142, -4.8143, -4.8143, -4.8143, -4.8143, -4.8142, -4.8143,
               -4.8144, -4.8143, -4.8143, -4.8141, -4.8142, -4.8142, -4.8142, -4.8144,
               -4.8143, -4.8143, -4.8143, -4.8142, -4.8144, -4.8144, -4.8142, -4.8142
            ]
            , dtype=torch.float32)  # fmt: skip
        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)

    @parameterized.expand([(torch_device,), ("cpu",)])
    @slow
    def test_simple_batched_generate_with_padding(self, torch_device):
        self.model.to(torch_device)

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "When did the Roman empire "],
            padding=True,
            return_tensors="pt",
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(
            output_sentences[0],
            "<s> Hey how are you doing on this lovely evening?\n\nI'm doing well, thanks for",
        )

        self.assertEqual(
            output_sentences[1],
            "[PAD][PAD][PAD][PAD]<s> When did the Roman empire 1st fall?\nThe Roman Empire fell in",
        )

        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits.to(
                dtype=torch.float32
            )

        EXPECTED_LOGITS_NO_GRAD_0 = torch.tensor(
            [
                -5.9611, 10.5208,  7.0411, -2.8743, -4.8167, -4.8167, -4.8167, -4.8168,
                -4.8167, -4.8167, -4.8167, -4.8166,  6.0218, 18.0062, -4.8167, -4.8168,
                -4.8167, -4.8167, -4.8167, -4.8168, -4.8168, -4.8168, -4.8167, -4.8167,
                -4.8168, -4.8167, -4.8167, -4.8165, -4.8167, -4.8167, -4.8167, -4.8169,
                -4.8168, -4.8168, -4.8168, -4.8166, -4.8169, -4.8168, -4.8167, -4.8167
            ]
            , dtype=torch.float32)  # fmt: skip

        EXPECTED_LOGITS_NO_GRAD_1 = torch.tensor(
            [
               0.1966,  6.3449,  3.8350, -5.7291, -6.5106, -6.5104, -6.5103, -6.5104,
               -6.5103, -6.5104, -6.5106, -6.5105,  7.8700, 13.5434, -6.5104, -6.5096,
               -6.5106, -6.5102, -6.5106, -6.5106, -6.5105, -6.5106, -6.5104, -6.5106,
               -6.5105, -6.5106, -6.5106, -6.5113, -6.5102, -6.5105, -6.5108, -6.5105,
               -6.5104, -6.5106, -6.5106, -6.5104, -6.5106, -6.5107, -6.5103, -6.5105          ]
            , dtype=torch.float32)  # fmt: skip

        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            logits[1, -1, :40].cpu(),
            EXPECTED_LOGITS_NO_GRAD_1,
            rtol=1e-3,
            atol=6e-3 if torch_device == "cpu" else 1e-3,
        )
