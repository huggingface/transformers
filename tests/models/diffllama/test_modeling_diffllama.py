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
"""Testing suite for the PyTorch DiffLlama model."""

import gc
import unittest

from packaging import version

from transformers import AutoTokenizer, DiffLlamaConfig, StaticCache, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        DiffLlamaForCausalLM,
        DiffLlamaForQuestionAnswering,
        DiffLlamaForSequenceClassification,
        DiffLlamaForTokenClassification,
        DiffLlamaModel,
    )


class DiffLlamaModelTester(CausalLMModelTester):
    config_class = DiffLlamaConfig
    if is_torch_available():
        base_model_class = DiffLlamaModel
        causal_lm_class = DiffLlamaForCausalLM
        question_answering_class = DiffLlamaForQuestionAnswering
        sequence_classification_class = DiffLlamaForSequenceClassification
        token_classification_class = DiffLlamaForTokenClassification


@require_torch
class DiffLlamaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            DiffLlamaModel,
            DiffLlamaForCausalLM,
            DiffLlamaForSequenceClassification,
            DiffLlamaForQuestionAnswering,
            DiffLlamaForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": DiffLlamaModel,
            "text-classification": DiffLlamaForSequenceClassification,
            "text-generation": DiffLlamaForCausalLM,
            "zero-shot": DiffLlamaForSequenceClassification,
            "question-answering": DiffLlamaForQuestionAnswering,
            "token-classification": DiffLlamaForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = DiffLlamaModelTester


@require_torch_accelerator
class DiffLlamaIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # See LlamaIntegrationTest.tearDown(). Can be removed once LlamaIntegrationTest.tearDown() is removed.
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_torch_accelerator
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
        # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the speed of light is constant in all inertial "
            "reference frames, and 2) the laws of physics are the same for all inertial reference frames.\nThe "
            "theory of relativ",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, "
            "my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "kajuma/DiffLlama-0.3B-handcut", pad_token="</s>", padding_side="right"
        )
        model = DiffLlamaForCausalLM.from_pretrained(
            "kajuma/DiffLlama-0.3B-handcut", device_map=torch_device, torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)


@slow
@require_torch_accelerator
class Mask4DTestHard(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def setUp(self):
        model_name = "kajuma/DiffLlama-0.3B-handcut"
        self.model_dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DiffLlamaForCausalLM.from_pretrained(model_name, torch_dtype=self.model_dtype).to(torch_device)

    def get_test_data(self):
        template = "my favorite {}"
        items = ("pet is a", "artist plays a", "name is L")  # same number of tokens in each item

        batch_separate = [template.format(x) for x in items]  # 3 separate lines
        batch_shared_prefix = template.format(" ".join(items))  # 1 line with options concatenated

        input_ids = self.tokenizer(batch_separate, return_tensors="pt").input_ids.to(torch_device)
        input_ids_shared_prefix = self.tokenizer(batch_shared_prefix, return_tensors="pt").input_ids.to(torch_device)

        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                ]
            ],
            device=torch_device,
        )

        position_ids = torch.arange(input_ids.shape[1]).tile(input_ids.shape[0], 1).to(torch_device)

        # building custom positions ids based on custom mask
        position_ids_shared_prefix = (mask_shared_prefix.sum(dim=-1) - 1).reshape(1, -1)
        # effectively: position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]]).to(device)

        # inverting the mask
        min_dtype = torch.finfo(self.model_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=self.model_dtype) * min_dtype

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix

    def test_stacked_causal_mask(self):
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # single forward run with 4D custom mask
        logits_shared_prefix = self.model.forward(
            input_ids_shared_prefix, attention_mask=mask_shared_prefix, position_ids=position_ids_shared_prefix
        ).logits
        logits_shared_prefix_last = logits_shared_prefix[
            0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1], :
        ]  # last three tokens
        decoded_shared_prefix = [self.tokenizer.decode(t) for t in logits_shared_prefix_last.argmax(dim=-1)]

        self.assertEqual(decoded, decoded_shared_prefix)

    def test_partial_stacked_causal_mask(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention masks

        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # 2 forward runs with custom 4D masks
        part_a = 3  # split point

        input_1a = input_ids_shared_prefix[:, :part_a]
        position_ids_1a = position_ids_shared_prefix[:, :part_a]
        mask_1a = mask_shared_prefix[:, :, :part_a, :part_a]

        outs_1a = self.model.forward(input_1a, attention_mask=mask_1a, position_ids=position_ids_1a)
        past_key_values_a = outs_1a["past_key_values"]

        # Case 1: we pass a 4D attention mask regarding the current sequence length (i.e. [..., seq_len, full_len])
        input_1b = input_ids_shared_prefix[:, part_a:]
        position_ids_1b = position_ids_shared_prefix[:, part_a:]
        mask_1b = mask_shared_prefix[:, :, part_a:, :]
        outs_1b = self.model.forward(
            input_1b,
            attention_mask=mask_1b,
            position_ids=position_ids_1b,
            past_key_values=past_key_values_a,
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        self.assertEqual(decoded, decoded_1b)

    def test_stacked_causal_mask_static_cache(self):
        """same as above but with StaticCache"""
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # upgrade the model with StaticCache
        max_cache_len = 16  # note that max_cache_len is greater than the attention_mask.shape[-1]
        past_key_values = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=torch_device,
            dtype=self.model.dtype,
        )

        padded_attention_mask = torch.nn.functional.pad(
            input=mask_shared_prefix,
            pad=(0, max_cache_len - mask_shared_prefix.shape[-1]),
            mode="constant",
            value=torch.finfo(self.model_dtype).min,
        )

        # single forward run with 4D custom mask
        logits_shared_prefix = self.model.forward(
            input_ids_shared_prefix,
            attention_mask=padded_attention_mask,
            position_ids=position_ids_shared_prefix,
            cache_position=torch.arange(input_ids_shared_prefix.shape[-1], device=torch_device),
            past_key_values=past_key_values,
        ).logits
        logits_shared_prefix_last = logits_shared_prefix[
            0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1], :
        ]  # last three tokens
        decoded_shared_prefix = [self.tokenizer.decode(t) for t in logits_shared_prefix_last.argmax(dim=-1)]

        self.assertEqual(decoded, decoded_shared_prefix)

    def test_partial_stacked_causal_mask_static_cache(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention masks
        # we pass a 4D attention mask shaped [..., seq_len, full_static_cache_len])
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # upgrade the model with StaticCache
        max_cache_len = 16  # note that max_cache_len is greater than the attention_mask.shape[-1]
        past_key_values = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=torch_device,
            dtype=self.model.dtype,
        )

        # forward run for the first part of input
        part_a = 3  # split point

        input_1a = input_ids_shared_prefix[:, :part_a]
        position_ids_1a = position_ids_shared_prefix[:, :part_a]
        mask_1a = mask_shared_prefix[:, :, :part_a, :part_a]

        padded_mask_1a = torch.nn.functional.pad(
            input=mask_1a,
            pad=(0, max_cache_len - mask_1a.shape[-1]),
            mode="constant",
            value=torch.finfo(self.model_dtype).min,
        )

        _ = self.model.forward(
            input_1a,
            attention_mask=padded_mask_1a,
            position_ids=position_ids_1a,
            cache_position=torch.arange(part_a, device=torch_device),
            past_key_values=past_key_values,
        )

        # forward run for the second part of input
        input_1b = input_ids_shared_prefix[:, part_a:]
        position_ids_1b = position_ids_shared_prefix[:, part_a:]
        mask_1b = mask_shared_prefix[:, :, part_a:, :]

        padded_mask_1b = torch.nn.functional.pad(
            input=mask_1b, pad=(0, max_cache_len - mask_1b.shape[-1]), mode="constant", value=0
        )

        outs_1b = self.model.forward(
            input_1b,
            attention_mask=padded_mask_1b,
            position_ids=position_ids_1b,
            cache_position=torch.arange(
                part_a,
                input_ids_shared_prefix.shape[-1],
                device=torch_device,
            ),
            past_key_values=past_key_values,
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        self.assertEqual(decoded, decoded_1b)
