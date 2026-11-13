# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import GPT2Config, GPT2LMHeadModel


@require_torch
class ClassifierFreeGuidanceBatchExpansionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=23,
                n_positions=32,
                n_embd=16,
                n_layer=2,
                n_head=2,
                bos_token_id=1,
                pad_token_id=0,
                eos_token_id=None,
                attn_pdrop=0.0,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
            )
        ).eval()

    def check_expanded_generation(self, batch_size, num_return_sequences, use_cache, guidance_scale, negative):
        input_ids = torch.tensor([[1, 5, 6], [1, 7, 8]])[:batch_size]
        saved_negative = {key: value.clone() for key, value in negative.items()}
        options = {
            "do_sample": True,
            "top_k": 0,
            "max_new_tokens": 4,
            "use_cache": use_cache,
            "guidance_scale": guidance_scale,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        torch.manual_seed(91)
        actual = self.model.generate(
            input_ids,
            num_return_sequences=num_return_sequences,
            **negative,
            **options,
        )
        torch.manual_seed(91)
        expected = self.model.generate(
            input_ids.repeat_interleave(num_return_sequences, dim=0),
            **{key: value.repeat_interleave(num_return_sequences, dim=0) for key, value in negative.items()},
            **options,
        )
        torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
        self.assertEqual(actual.sequences.shape, (batch_size * num_return_sequences, 7))
        self.assertEqual(len(actual.scores), 4)
        for actual_scores, expected_scores in zip(actual.scores, expected.scores):
            torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
        for key, value in negative.items():
            torch.testing.assert_close(value, saved_negative[key], rtol=0, atol=0)

    @parameterized.expand(
        [
            (batch_size, returns, use_cache, explicit_mask, scale)
            for batch_size in (1, 2)
            for returns in (1, 3)
            for use_cache in (False, True)
            for explicit_mask in (False, True)
            for scale in (0.0, 1.5)
        ]
    )
    def test_negative_prompt_expansion(self, batch_size, returns, use_cache, explicit_mask, scale):
        negative_ids = torch.tensor([[9, 10], [11, 12]])[:batch_size]
        negative = {"negative_prompt_ids": negative_ids}
        if explicit_mask:
            negative["negative_prompt_attention_mask"] = torch.ones_like(negative_ids)
        self.check_expanded_generation(batch_size, returns, use_cache, scale, negative)

    @parameterized.expand([(False,), (True,)])
    def test_mask_without_negative_ids(self, use_cache):
        negative = {"negative_prompt_attention_mask": torch.ones(2, 1, dtype=torch.long)}
        self.check_expanded_generation(2, 3, use_cache, 1.5, negative)

    @parameterized.expand([(False,), (True,)])
    def test_implicit_negative_prompt(self, use_cache):
        self.check_expanded_generation(2, 3, use_cache, 1.5, {})
