# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GPT Neo model."""

import unittest

from transformers import GPTNeoConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import cached_property

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        GPT2Tokenizer,
        GPTNeoForCausalLM,
        GPTNeoForQuestionAnswering,
        GPTNeoForSequenceClassification,
        GPTNeoForTokenClassification,
        GPTNeoModel,
    )


class GPTNeoModelTester(CausalLMModelTester):
    config_class = GPTNeoConfig
    if is_torch_available():
        base_model_class = GPTNeoModel
        causal_lm_class = GPTNeoForCausalLM
        question_answering_class = GPTNeoForQuestionAnswering
        sequence_classification_class = GPTNeoForSequenceClassification
        token_classification_class = GPTNeoForTokenClassification


@require_torch
class GPTNeoModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            GPTNeoModel,
            GPTNeoForCausalLM,
            GPTNeoForQuestionAnswering,
            GPTNeoForSequenceClassification,
            GPTNeoForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GPTNeoModel,
            "question-answering": GPTNeoForQuestionAnswering,
            "text-classification": GPTNeoForSequenceClassification,
            "text-generation": GPTNeoForCausalLM,
            "token-classification": GPTNeoForTokenClassification,
            "zero-shot": GPTNeoForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = GPTNeoModelTester


@require_torch
class GPTNeoModelLanguageGenerationTest(unittest.TestCase):
    @cached_property
    def model(self):
        return GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(torch_device)

    @cached_property
    def tokenizer(self):
        return GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    @slow
    def test_lm_generate_gpt_neo(self):
        for checkpointing in [True, False]:
            model = self.model
            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            input_ids = torch.tensor([[464, 3290]], dtype=torch.long, device=torch_device)  # The dog
            # The dog-eared copy of the book, which is a collection of essays by the late author,
            expected_output_ids = [464, 3290, 12, 3380, 4866, 286, 262, 1492, 11, 543, 318, 257, 4947, 286, 27126, 416, 262, 2739, 1772, 11]  # fmt: skip
            output_ids = model.generate(input_ids, do_sample=False)
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_gpt_neo_sample(self):
        model = self.model
        tokenizer = self.tokenizer

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)
        output_ids = model.generate(input_ids, do_sample=True)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = "Today is a nice day and if you don’t get the memo here is what you can"
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)

    @slow
    def test_batch_generation(self):
        model = self.model
        tokenizer = self.tokenizer

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I am",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded)

        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].long().sum().item()
        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a kitty. She is a very sweet and loving",
            "Today, I am going to talk about the best way to get a job in the",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        model_name = "EleutherAI/gpt-neo-1.3B"
        model = GPTNeoModel.from_pretrained(model_name)
        self.assertIsNotNone(model)
