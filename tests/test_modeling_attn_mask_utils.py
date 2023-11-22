# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

from transformers.testing_utils import (
    require_torch,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.modeling_attn_mask_utils import AttentionMaskConverter


class TestAttnMaskConverter(unittest.TestCase):
    @require_torch
    @slow
    def test_unmask_unattended_left_padding(self):
        attention_mask = torch.Tensor([[0, 0, 1], [1, 1, 1], [0, 1, 1]]).to(torch.int64)

        expanded_mask = torch.Tensor(
            [
                [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[0, 0, 0], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(torch.int64)

        reference_output = torch.Tensor(
            [
                [[[1, 1, 1], [1, 1, 1], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[1, 1, 1], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(torch.int64)

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=1)

        self.assertTrue(torch.equal(result, reference_output))

        attention_mask = torch.Tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)
        min_inf = torch.finfo(torch.float32).min
        reference_output = torch.Tensor(
            [
                [
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [min_inf, min_inf, 0, min_inf, min_inf],
                        [min_inf, min_inf, 0, 0, min_inf],
                        [min_inf, min_inf, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, min_inf, min_inf, min_inf, min_inf],
                        [0, 0, min_inf, min_inf, min_inf],
                        [0, 0, 0, min_inf, min_inf],
                        [0, 0, 0, 0, min_inf],
                        [0, 0, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0, 0, 0],
                        [min_inf, 0, min_inf, min_inf, min_inf],
                        [min_inf, 0, 0, min_inf, min_inf],
                        [min_inf, 0, 0, 0, min_inf],
                        [min_inf, 0, 0, 0, 0],
                    ]
                ],
            ]
        )

        self.assertTrue(torch.equal(reference_output, result))

    @require_torch
    @slow
    def test_unmask_unattended_right_padding(self):
        attention_mask = torch.Tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(torch.equal(expanded_mask, result))

    @require_torch
    @slow
    def test_unmask_unattended_random_mask(self):
        attention_mask = torch.Tensor([[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(torch.equal(expanded_mask, result))
