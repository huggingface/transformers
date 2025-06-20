# Copyright 2023 The HuggingFace Team. All rights reserved.
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
#
import unittest

from transformers import MptConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_bitsandbytes,
    require_deterministic_for_xpu,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        MptForCausalLM,
        MptForQuestionAnswering,
        MptForSequenceClassification,
        MptForTokenClassification,
        MptModel,
    )


@require_torch
class MptModelTester(CausalLMModelTester):
    config_class = MptConfig
    if is_torch_available():
        base_model_class = MptModel
        causal_lm_class = MptForCausalLM
        question_answering_class = MptForQuestionAnswering
        sequence_classification_class = MptForSequenceClassification
        token_classification_class = MptForTokenClassification


class MptConfigTester(ConfigTester):
    def __init__(self, parent, config_class=None, has_text_modality=True, common_properties=None, **kwargs):
        super().__init__(parent, config_class, has_text_modality, common_properties, **kwargs)

    def test_attn_config_as_dict(self):
        config = self.config_class(**self.inputs_dict, attn_config={"attn_impl": "flash", "softmax_scale": None})
        self.parent.assertTrue(config.attn_config.attn_impl == "flash")
        self.parent.assertTrue(config.attn_config.softmax_scale is None)

    def run_common_tests(self):
        self.test_attn_config_as_dict()
        return super().run_common_tests()


@require_torch
class MptModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            MptModel,
            MptForCausalLM,
            MptForSequenceClassification,
            MptForTokenClassification,
            MptForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MptModel,
            "question-answering": MptForQuestionAnswering,
            "text-classification": MptForSequenceClassification,
            "text-generation": MptForCausalLM,
            "token-classification": MptForTokenClassification,
            "zero-shot": MptForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = MptModelTester

    @unittest.skip(reason="For backward compatibility the lm_head is not in the model's state dict on the Hub.")
    def test_model_weights_reload_no_missing_tied_weights(self):
        pass


@slow
@require_torch_accelerator
@require_bitsandbytes
class MptIntegrationTests(unittest.TestCase):
    def test_generation_8k(self):
        model_id = "mosaicml/mpt-7b-8k"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load in 4bit to fit the daily CI runner GPU RAM
        model = MptForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": 0}, load_in_4bit=True
        )

        input_text = "Hello"
        expected_outputs = Expectations({
            ("cuda", None): "Hello, I'm a new user of the forum. I have a question about the \"Solaris",
            ("rocm", (9, 5)): "Hello, I'm a newbie to the forum. I have a question about the \"B\" in",
        })  # fmt: off
        expected_output = expected_outputs.get_expectation()

        inputs = tokenizer(input_text, return_tensors="pt").to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=20)

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertEqual(decoded_output, expected_output)

    def test_generation(self):
        model_id = "mosaicml/mpt-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load in 4bit to fit the daily CI runner GPU RAM
        model = MptForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": 0}, load_in_4bit=True
        )

        input_text = "Hello"
        expected_outputs = Expectations({
            ("rocm", (9, 5)): "Hello and welcome to the first day of the new release at The Stamp Man!\nToday we are",
            ("xpu", 3): "Hello and welcome to the first ever episode of the new and improved, and hopefully improved, podcast.\n",
            ("cuda", 7): "Hello and welcome to the first episode of the new podcast, The Frugal Feminist.\n",
            ("cuda", 8): "Hello and welcome to the first day of the new release countdown for the month of May!\nToday",
        })  # fmt: off
        expected_output = expected_outputs.get_expectation()

        inputs = tokenizer(input_text, return_tensors="pt").to(torch_device)
        outputs = model.generate(**inputs, max_new_tokens=20)

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertEqual(decoded_output, expected_output)

    @require_deterministic_for_xpu
    def test_generation_batched(self):
        model_id = "mosaicml/mpt-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load in 4bit to fit the daily CI runner GPU RAM
        model = MptForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": 0}, load_in_4bit=True
        )

        input_texts = ["Hello my name is", "Today I am going at the gym and"]
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(torch_device)

        expected_outputs = Expectations(
            {
                ("xpu", 3): [
                    "Hello my name is Tiffany. I am a mother of two beautiful children. I have been a nanny for over",
                    "Today I am going at the gym and then I am going to go to the mall with my mom. I am going to go to the",
                ],
                ("cuda", 7): [
                    "Hello my name is Tiffany and I am a mother of two beautiful children. I have been a nanny for the",
                    "Today I am going at the gym and then I am going to go to the grocery store. I am going to buy some food and some",
                ],
                ("rocm", (9, 5)): [
                    "Hello my name is Jasmine and I am a very sweet and loving dog. I am a very playful dog and I",
                    "Today I am going at the gym and then I am going to go to the mall. I am going to buy a new pair of jeans",
                ],
            }
        )
        expected_output = expected_outputs.get_expectation()
        outputs = model.generate(**inputs, max_new_tokens=20)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, predicted_output in enumerate(decoded_outputs):
            self.assertEqual(predicted_output, expected_output[i])

    def test_model_logits(self):
        model_id = "mosaicml/mpt-7b"

        # Load in 4bit to fit the daily CI runner GPU RAM
        model = MptForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": 0}, load_in_4bit=True
        )

        dummy_input = torch.LongTensor([[1, 2, 3, 4, 5]]).to(torch_device)

        outputs = model(dummy_input, output_hidden_states=True)

        expected_slices = Expectations(
            {
                ("xpu", 3): torch.Tensor([-0.2090, -0.2061, -0.1465]),
                ("cuda", 7): torch.Tensor([-0.2520, -0.2178, -0.1953]),
                # TODO: This is quite a bit off, check BnB
                ("rocm", (9, 5)): torch.Tensor([-0.3008, -0.1309, -0.1562]),
            }
        )
        expected_slice = expected_slices.get_expectation().to(torch_device, torch.bfloat16)
        predicted_slice = outputs.hidden_states[-1][0, 0, :3]
        torch.testing.assert_close(expected_slice, predicted_slice, rtol=1e-3, atol=1e-3)
