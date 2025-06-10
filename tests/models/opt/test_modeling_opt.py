# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch OPT model."""

import unittest

import timeout_decorator  # noqa

from transformers import OPTConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        GPT2Tokenizer,
        OPTForCausalLM,
        OPTForQuestionAnswering,
        OPTForSequenceClassification,
        OPTModel,
    )


def prepare_opt_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class OPTModelTester(CausalLMModelTester):
    config_class = OPTConfig
    if is_torch_available():
        base_model_class = OPTModel
        causal_lm_class = OPTForCausalLM
        question_answering_class = OPTForQuestionAnswering
        sequence_classification_class = OPTForSequenceClassification


@require_torch
class OPTModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (OPTModel, OPTForCausalLM, OPTForSequenceClassification, OPTForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": OPTModel,
            "question-answering": OPTForQuestionAnswering,
            "text-classification": OPTForSequenceClassification,
            "text-generation": OPTForCausalLM,
            "zero-shot": OPTForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = OPTModelTester

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        super().test_model_parallelism()


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


@require_torch
class OPTModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = OPTModel.from_pretrained("facebook/opt-350m").to(torch_device)
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])

        with torch.no_grad():
            output = model(input_ids=input_ids).last_hidden_state

        expected_shape = torch.Size((1, 11, 512))
        self.assertEqual(output.shape, expected_shape)
        # expected value works for CPU, as well as GPU (with TF32 disabled)
        expected_slice = torch.tensor(
            [
                [-0.28726277, -1.9241608, -0.3058734],
                [-1.2737825, -0.13332152, -0.18766522],
                [0.41159445, 0.1191957, -1.3107123],
            ],
            device=torch_device,
        )
        assert_tensors_close(output[0, :3, :3], expected_slice, atol=5e-5)


@require_torch
@slow
class OPTEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path_model = "facebook/opt-350m"

    def test_load_model(self):
        try:
            _ = OPTForCausalLM.from_pretrained(self.path_model)
        except BaseException:
            self.fail("Failed loading model")

    def test_logits(self):
        model = OPTForCausalLM.from_pretrained(self.path_model)
        model = model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained(self.path_model)

        prompts = [
            "Today is a beautiful day and I want to",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]
        # verify that prompt without BOS token is identical to Metaseq -> add_special_tokens=False
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)[0].mean(dim=-1)
        logits_meta = torch.Tensor(
            [
                [1.3851, -13.8923, -10.5229, -10.7533, -0.2309, -10.2384, -0.5365, -9.0947, -5.1670],
                [-4.7073, -10.6276, -3.9415, -21.5242, -0.2822, -0.2822, -0.2822, -0.2822, -0.2822],
                [0.6247, -3.4229, -8.9179, -1.4297, -14.1650, 1.4146, -9.0218, -0.2703, -0.2703],
                [6.4783, -1.9913, -10.7926, -2.3336, 1.5092, -0.9974, -6.8213, 1.3477, 1.3477],
            ]
        )
        assert torch.allclose(logits, logits_meta, atol=1e-4)


@slow
class OPTGenerationTest(unittest.TestCase):
    @property
    def prompts(self):
        return [
            "Today is a beautiful day and I want",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]

    def test_generation_pre_attn_layer_norm(self):
        model_id = "facebook/opt-125m"

        EXPECTED_OUTPUTS = [
            "Today is a beautiful day and I want to",
            "In the city of New York, the city",
            "Paris is the capital of France and the capital",
            "Computers and mobile phones have taken over the",
        ]

        predicted_outputs = []
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = OPTForCausalLM.from_pretrained(model_id)

        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            generated_ids = model.generate(input_ids, max_length=10)

            generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predicted_outputs += generated_string

        self.assertListEqual(predicted_outputs, EXPECTED_OUTPUTS)

    def test_batch_generation(self):
        model_id = "facebook/opt-350m"

        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = OPTForCausalLM.from_pretrained(model_id)
        model.to(torch_device)

        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
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
            "Hello, my dog is a little bit of a dork.\nI'm a little bit",
            "Today, I was in the middle of a conversation with a friend about the",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(batch_out_sentence, [non_padded_sentence, padded_sentence])

    def test_generation_post_attn_layer_norm(self):
        model_id = "facebook/opt-350m"

        EXPECTED_OUTPUTS = [
            "Today is a beautiful day and I want to",
            "In the city of San Francisco, the city",
            "Paris is the capital of France and the capital",
            "Computers and mobile phones have taken over the",
        ]

        predicted_outputs = []
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = OPTForCausalLM.from_pretrained(model_id)

        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            generated_ids = model.generate(input_ids, max_length=10)

            generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predicted_outputs += generated_string

        self.assertListEqual(predicted_outputs, EXPECTED_OUTPUTS)

    @require_torch_accelerator
    @require_torch_fp16
    def test_batched_nan_fp16(self):
        # a bug manifested starting at models facebook/opt-1.3 and larger when running batched generations,
        # therefore not using a tiny model, but the smallest model the problem was seen with which is opt-1.3b.
        # please refer to this github thread: https://github.com/huggingface/transformers/pull/17437 for more details
        model_name = "facebook/opt-1.3b"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")

        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).to(torch_device)
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
    def test_contrastive_search_opt(self):
        article = (
            "A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
            "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
            "there?"
        )

        opt_tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
        opt_model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b").to(torch_device)
        input_ids = opt_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        outputs = opt_model.generate(input_ids, penalty_alpha=0.6, top_k=5, max_length=256)
        generated_text = opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I "
                "am the Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have "
                "you lived there?\nStatue: A hundred years.\nHuman: And you’re from what country?\nStatue: The United "
                "States of America.\nHuman: Why did you come to America?\nStatue: I came to escape the tyranny of my "
                "country.\nHuman: What tyranny?\nStatue: They didn’t let me speak my mind.\nHuman: What was your "
                "country?\nStatue: It was a country of immigrants.\nHuman: Who were the immigrants?\nStatue: They "
                "were from all over the world.\nHuman: What language did they speak?\nStatue: French, Spanish, "
                "Italian, German, English—you name it.\nHuman: And where did they come from?\nStatue: They came from "
                "every country in the world.\nHuman: And you were born in what country?\nStatue: I was born in "
                "France.\nHuman: And your parents were French?\nStatue"
            ],
        )
