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
"""Testing suite for the PyTorch Jamba model."""

import unittest

import pytest

from transformers import AutoTokenizer, JambaConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    get_device_properties,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        JambaForCausalLM,
        JambaForSequenceClassification,
        JambaModel,
    )


class JambaConfigTester(ConfigTester):
    def _create_attn_config(self, attn_layer_offset: int, attn_layer_period: int):
        _input_dict = self.inputs_dict.copy()
        _input_dict["attn_layer_offset"] = attn_layer_offset
        _input_dict["attn_layer_period"] = attn_layer_period
        return self.config_class(**_input_dict)

    def _create_expert_config(self, expert_layer_offset: int, expert_layer_period: int):
        _input_dict = self.inputs_dict.copy()
        _input_dict["expert_layer_offset"] = expert_layer_offset
        _input_dict["expert_layer_period"] = expert_layer_period
        return self.config_class(**_input_dict)

    def test_attn_offsets(self):
        self._create_attn_config(attn_layer_offset=0, attn_layer_period=4)
        self._create_attn_config(attn_layer_offset=1, attn_layer_period=4)
        self._create_attn_config(attn_layer_offset=2, attn_layer_period=4)
        self._create_attn_config(attn_layer_offset=3, attn_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_attn_config(attn_layer_offset=4, attn_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_attn_config(attn_layer_offset=5, attn_layer_period=4)

    def test_expert_offsets(self):
        self._create_expert_config(expert_layer_offset=0, expert_layer_period=4)
        self._create_expert_config(expert_layer_offset=1, expert_layer_period=4)
        self._create_expert_config(expert_layer_offset=2, expert_layer_period=4)
        self._create_expert_config(expert_layer_offset=3, expert_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_expert_config(expert_layer_offset=4, expert_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_expert_config(expert_layer_offset=5, expert_layer_period=4)

    def test_jamba_offset_properties(self):
        self.test_attn_offsets()
        self.test_expert_offsets()

    def run_common_tests(self):
        self.test_jamba_offset_properties()
        return super().run_common_tests()


class JambaModelTester(CausalLMModelTester):
    config_class = JambaConfig
    if is_torch_available():
        base_model_class = JambaModel
        causal_lm_class = JambaForCausalLM
        sequence_classification_class = JambaForSequenceClassification

    def __init__(self, parent, use_mamba_kernels=False, **kwargs):
        self.use_mamba_kernels = use_mamba_kernels
        super().__init__(parent=parent, **kwargs)


@require_torch
class JambaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            JambaModel,
            JambaForCausalLM,
            JambaForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": JambaModel,
            "text-classification": JambaForSequenceClassification,
            "text-generation": JambaForCausalLM,
            "zero-shot": JambaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = JambaModelTester

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        Mamba block are initialized differently and we tested that in test_initialization
        """
        self.skipTest(reason="Cumbersome and redundant for Jamba")

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        r"""
        Overriding the test_flash_attn_2_inference_padding_right test as the Jamba model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        self.skipTest(reason="Jamba flash attention does not support right padding")


@require_torch
class JambaModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None
    # This variable is used to determine which acclerator are we using for our runners (e.g. A10 or T4)
    # Depending on the hardware we get different logits / generations
    device_properties = None

    @classmethod
    def setUpClass(cls):
        model_id = "ai21labs/Jamba-tiny-dev"
        cls.model = JambaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.device_properties = get_device_properties()

    @slow
    def test_simple_generate(self):
        # ("cuda", 8) for A100/A10, and ("cuda", 7) for T4.
        #
        # considering differences in hardware processing and potential deviations in generated text.
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 7): "<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh<|reserved_797|>cw algunas",
                ("cuda", 8): "<|startoftext|>Hey how are you doing on this lovely evening? I'm so glad you're here.",
                ("rocm", 9): "<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh Hebrew llam bb",
            }
        )
        # fmt: on
        expected_sentence = EXPECTED_TEXTS.get_expectation()

        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        self.assertEqual(output_sentence, expected_sentence)

        # TODO: there are significant differences in the logits across major cuda versions, which shouldn't exist
        if self.device_properties == ("cuda", 8):
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits

            EXPECTED_LOGITS_NO_GRAD = torch.tensor(
                [
                    -7.6875, -7.6562,  8.9375, -7.7812, -7.4062, -7.9688, -8.3125, -7.4062,
                    -7.8125, -8.1250, -7.8125, -7.3750, -7.8438, -7.5000, -8.0625, -8.0625,
                    -7.5938, -7.9688, -8.2500, -7.5625, -7.7500, -7.7500, -7.6562, -7.6250,
                    -8.1250, -8.0625, -8.1250, -7.8750, -8.1875, -8.2500, -7.5938, -8.0000,
                    -7.5000, -7.7500, -7.9375, -7.4688, -8.0625, -7.3438, -8.0000, -7.5000
                ]
                , dtype=torch.float32)  # fmt: skip

            torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)

    @slow
    def test_simple_batched_generate_with_padding(self):
        # ("cuda", 8) for A100/A10, and ("cuda", 7) for T4.
        #
        # considering differences in hardware processing and potential deviations in generated text.
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 7): ["<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh Hebrew cases Cats", "<|pad|><|pad|><|pad|><|pad|><|pad|><|pad|><|startoftext|>Tell me a storyptus Nets Madison El chamadamodern updximVaparsed",],
                ("cuda", 8): ["<|startoftext|>Hey how are you doing on this lovely evening? I'm so glad you're here.", "<|pad|><|pad|><|pad|><|pad|><|pad|><|pad|><|startoftext|>Tell me a story about a woman who was born in the United States",],
                ("rocm", 9): ["<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh<|reserved_797|>cw algunas", "<|pad|><|pad|><|pad|><|pad|><|pad|><|pad|><|startoftext|>Tell me a storyptus Nets Madison El chamadamodern updximVaparsed",],
            }
        )
        # fmt: on
        expected_sentences = EXPECTED_TEXTS.get_expectation()

        self.model.to(torch_device)

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "Tell me a story"], padding=True, return_tensors="pt"
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(output_sentences[0], expected_sentences[0])
        self.assertEqual(output_sentences[1], expected_sentences[1])

        # TODO: there are significant differences in the logits across major cuda versions, which shouldn't exist
        if self.device_properties == ("cuda", 8):
            with torch.no_grad():
                logits = self.model(input_ids=inputs["input_ids"]).logits

            # TODO fix logits
            EXPECTED_LOGITS_NO_GRAD_0 = torch.tensor(
                [
                    -7.7188, -7.6875,  8.8750, -7.8125, -7.4062, -8.0000, -8.3125, -7.4375,
                    -7.8125, -8.1250, -7.8125, -7.4062, -7.8438, -7.5312, -8.0625, -8.0625,
                    -7.6250, -8.0000, -8.3125, -7.5938, -7.7500, -7.7500, -7.6562, -7.6562,
                    -8.1250, -8.0625, -8.1250, -7.8750, -8.1875, -8.2500, -7.5938, -8.0625,
                     -7.5000, -7.7812, -7.9375, -7.4688, -8.0625, -7.3750, -8.0000, -7.50003
                ]
                , dtype=torch.float32)  # fmt: skip

            EXPECTED_LOGITS_NO_GRAD_1 = torch.tensor(
                [
                    -3.5469, -4.0625,  8.5000, -3.8125, -3.6406, -3.7969, -3.8125, -3.3594,
                     -3.7188, -3.7500, -3.7656, -3.5469, -3.7969, -4.0000, -3.5625, -3.6406,
                    -3.7188, -3.6094, -4.0938, -3.6719, -3.8906, -3.9844, -3.8594, -3.4219,
                    -3.2031, -3.4375, -3.7500, -3.6562, -3.9688, -4.1250, -3.6406, -3.57811,
                    -3.0312, -3.4844, -3.6094, -3.5938, -3.7656, -3.8125, -3.7500, -3.8594
                ]
                , dtype=torch.float32)  # fmt: skip

            torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(logits[1, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_1, rtol=1e-3, atol=1e-3)
