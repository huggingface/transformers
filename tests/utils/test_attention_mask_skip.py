# Copyright 2026 HuggingFace Inc.
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
from unittest.mock import patch

from transformers.testing_utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.masking_utils import (
        _ignore_bidirectional_mask_sdpa,
        _ignore_causal_mask_sdpa,
    )


@unittest.skipUnless(is_torch_available(), "torch is required")
class CausalMaskSkipTest(unittest.TestCase):
    def test_skip_when_no_padding_and_single_query(self):
        padding_mask = None
        self.assertTrue(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=1,
                kv_length=10,
                kv_offset=0,
                local_attention_size=None,
            )
        )

    def test_skip_when_no_padding_and_equal_lengths(self):
        padding_mask = None
        self.assertTrue(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=10,
                kv_length=10,
                kv_offset=0,
                local_attention_size=None,
            )
        )

    def test_no_skip_when_padding_exists(self):
        padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
        self.assertFalse(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=6,
                kv_length=6,
                kv_offset=0,
                local_attention_size=None,
            )
        )

    def test_skip_when_padding_is_all_ones(self):
        padding_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        self.assertTrue(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=6,
                kv_length=6,
                kv_offset=0,
                local_attention_size=None,
            )
        )

    def test_no_skip_when_local_attention_active(self):
        padding_mask = None
        self.assertFalse(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=10,
                kv_length=20,
                kv_offset=0,
                local_attention_size=16,
            )
        )

    def test_skip_when_kv_less_than_local_window(self):
        padding_mask = None
        self.assertTrue(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=10,
                kv_length=10,
                kv_offset=0,
                local_attention_size=16,
            )
        )

    def test_no_skip_when_query_gt_one_and_kv_gt_query(self):
        padding_mask = None
        self.assertFalse(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=5,
                kv_length=10,
                kv_offset=5,
                local_attention_size=None,
            )
        )

    def test_skip_with_offset_and_no_padding(self):
        padding_mask = None
        self.assertTrue(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=1,
                kv_length=10,
                kv_offset=5,
                local_attention_size=None,
            )
        )


@unittest.skipUnless(is_torch_available(), "torch is required")
class BidirectionalMaskSkipTest(unittest.TestCase):
    def test_skip_when_no_padding(self):
        padding_mask = None
        self.assertTrue(
            _ignore_bidirectional_mask_sdpa(
                padding_mask=padding_mask,
                kv_length=10,
                local_attention_size=None,
            )
        )

    def test_skip_when_all_ones(self):
        padding_mask = torch.tensor([[1, 1, 1, 1, 1]])
        self.assertTrue(
            _ignore_bidirectional_mask_sdpa(
                padding_mask=padding_mask,
                kv_length=5,
                local_attention_size=None,
            )
        )

    def test_no_skip_when_padding_exists(self):
        padding_mask = torch.tensor([[1, 1, 1, 0, 0]])
        self.assertFalse(
            _ignore_bidirectional_mask_sdpa(
                padding_mask=padding_mask,
                kv_length=5,
                local_attention_size=None,
            )
        )

    def test_no_skip_when_local_attention(self):
        padding_mask = None
        self.assertFalse(
            _ignore_bidirectional_mask_sdpa(
                padding_mask=padding_mask,
                kv_length=20,
                local_attention_size=16,
            )
        )


@unittest.skipUnless(is_torch_available(), "torch is required")
class MaskSkipWithTracingTest(unittest.TestCase):
    @patch("transformers.masking_utils.is_tracing", return_value=True)
    def test_no_skip_when_tracing_no_padding(self, mock_is_tracing):
        padding_mask = torch.tensor([[1, 1, 1, 1, 1]])
        self.assertFalse(
            _ignore_causal_mask_sdpa(
                padding_mask=padding_mask,
                query_length=1,
                kv_length=5,
                kv_offset=0,
                local_attention_size=None,
            )
        )

    @patch("transformers.masking_utils.is_tracing", return_value=True)
    def test_bidirectional_no_skip_when_tracing(self, mock_is_tracing):
        padding_mask = torch.tensor([[1, 1, 1, 1, 1]])
        self.assertFalse(
            _ignore_bidirectional_mask_sdpa(
                padding_mask=padding_mask,
                kv_length=5,
                local_attention_size=None,
            )
        )
