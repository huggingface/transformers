# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team and the Swiss AI Initiative. All rights reserved.
#
# This code is based on HuggingFace's LLaMA implementation in this library.
# It has been modified from its original forms to accommodate minor architectural
# differences compared to LLaMA used by the Swiss AI Initiative that trained the model.
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
"""Testing suite for the PyTorch Apertus model."""

import unittest

from transformers import AutoTokenizer, StaticCache, is_torch_available
from transformers.testing_utils import (
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
        ApertusConfig,
        ApertusForCausalLM,
        ApertusForTokenClassification,
        ApertusModel,
    )
    from transformers.models.apertus.modeling_apertus import ApertusRotaryEmbedding


class ApertusModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = ApertusConfig
        base_model_class = ApertusModel
        causal_lm_class = ApertusForCausalLM
        token_class = ApertusForTokenClassification


@require_torch
class ApertusModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            ApertusModel,
            ApertusForCausalLM,
            ApertusForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ApertusModel,
            "text-generation": ApertusForCausalLM,
            "token-classification": ApertusForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = ApertusModelTester
    rotary_embedding_layer = ApertusRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = ApertusForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
@slow
class ApertusIntegrationTest(unittest.TestCase):
    pass


@slow
@require_torch_accelerator
class Mask4DTestHard(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        model_name = "swiss-ai/Apertus-8B"
        self.model_dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ApertusForCausalLM.from_pretrained(model_name, torch_dtype=self.model_dtype).to(torch_device)

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
