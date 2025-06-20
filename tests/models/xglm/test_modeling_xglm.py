# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import unittest

from transformers import XGLMConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_torch_greater_or_equal,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import XGLMForCausalLM, XGLMModel, XGLMTokenizer


class XGLMModelTester(CausalLMModelTester):
    config_class = XGLMConfig
    if is_torch_available():
        base_model_class = XGLMModel
        causal_lm_class = XGLMForCausalLM


@require_torch
class XGLMModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (XGLMModel, XGLMForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": XGLMModel, "text-generation": XGLMForCausalLM} if is_torch_available() else {}
    )
    model_tester_class = XGLMModelTester

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        super().test_model_parallelism()


@require_torch
class XGLMModelLanguageGenerationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        cleanup(torch_device, gc_collect=True)

    def _test_lm_generate_xglm_helper(
        self,
        gradient_checkpointing=False,
        verify_outputs=True,
    ):
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.to(torch_device)
        input_ids = torch.tensor([[2, 268, 9865]], dtype=torch.long, device=torch_device)  # The dog
        # </s> The dog is a very friendly dog. He is very affectionate and loves to play with other
        expected_output_ids = [2, 268, 9865, 67, 11, 1988, 57252, 9865, 5, 984, 67, 1988, 213838, 1658, 53, 70446, 33, 6657, 278, 1581, 72616, 5, 984]  # fmt: skip
        output_ids = model.generate(input_ids, do_sample=False, num_beams=1)
        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_batch_generation(self):
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
        model.to(torch_device)
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")

        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            "This is an extremely long sentence that only exists to test the ability of the model to cope with "
            "left-padding, such as in batched generation. The output for the sequence below should be the same "
            "regardless of whether left padding is applied or not. When",
            "Hello, my dog is a little",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)

        outputs = model.generate(
            input_ids=input_ids, attention_mask=inputs["attention_mask"].to(torch_device), max_new_tokens=12
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded, max_new_tokens=12)

        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_new_tokens=12)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "This is an extremely long sentence that only exists to test the ability of the model to cope with "
            "left-padding, such as in batched generation. The output for the sequence below should be the same "
            "regardless of whether left padding is applied or not. When left padding is applied, the sequence will be "
            "a single",
            "Hello, my dog is a little bit of a shy one, but he is very friendly",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_lm_generate_xglm(self):
        self._test_lm_generate_xglm_helper()

    @slow
    def test_lm_generate_xglm_with_gradient_checkpointing(self):
        self._test_lm_generate_xglm_helper(gradient_checkpointing=True)

    @slow
    def test_xglm_sample(self):
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt")
        input_ids = tokenized.input_ids
        output_ids = model.generate(input_ids, do_sample=True, num_beams=1)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if is_torch_greater_or_equal("2.7.0"):
            cuda_expectation = (
                "Today is a nice day and the sun is shining. A nice day with warm rainy and windy weather today."
            )
        else:
            cuda_expectation = "Today is a nice day and the water is still cold. We just stopped off for some fresh coffee. This place looks like a"

        expected_output_strings = Expectations(
            {
                ("rocm", (9, 5)): "Today is a nice day and the sun is shining. A nice day with warm rainy and windy weather today.",
                ("cuda", None): cuda_expectation,
            }
        )  # fmt: skip
        EXPECTED_OUTPUT_STR = expected_output_strings.get_expectation()
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)

    @require_torch_accelerator
    @require_torch_fp16
    def test_batched_nan_fp16(self):
        model_name = "facebook/xglm-564M"
        tokenizer = XGLMTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")

        model = XGLMForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).to(torch_device)
        model = model.eval()

        batch = tokenizer(["Who are you?", "Joe Biden is the president of"], padding=True, return_tensors="pt")

        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            self.assertFalse(
                torch.isnan(outputs.logits[0]).any().item()
            )  # the first logits could contain NaNs if it fails

    @slow
    def test_loss_with_padding(self):
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
        model.to(torch_device)

        tokenizer.padding_side = "right"

        sequence = "Sequence"

        tokenized_non_padded = tokenizer(sequence, return_tensors="pt")
        tokenized_non_padded.to(torch_device)
        labels_non_padded = tokenized_non_padded.input_ids.clone()
        loss_non_padded = model(**tokenized_non_padded, labels=labels_non_padded).loss

        tokenized_padded = tokenizer(sequence, padding="max_length", max_length=16, return_tensors="pt")
        tokenized_padded.to(torch_device)
        labels_padded = tokenized_padded.input_ids.clone()
        labels_padded[labels_padded == tokenizer.pad_token_id] = -100
        loss_padded = model(**tokenized_padded, labels=labels_padded).loss

        torch.testing.assert_close(loss_non_padded, loss_padded, rtol=1e-3, atol=1e-3)
