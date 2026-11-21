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

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.modeling_flash_attention_utils import _pad_input, _unpad_input, _upad_input


@require_torch
class UpadInputTest(unittest.TestCase):
    """`_upad_input` only does tensor bookkeeping, so it can be checked on CPU without flash attention kernels."""

    def setUp(self):
        # 2 sequences of 6 keys, the second right-padded to 4 real keys; 3 queries per sequence
        self.query = torch.randn(2, 3, 2, 4)
        self.key = torch.randn(2, 6, 2, 4)
        self.value = torch.randn(2, 6, 2, 4)
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])

    def test_non_causal_cross_attention_keeps_all_queries(self):
        query, key, _, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_q, max_k) = _upad_input(
            self.query, self.key, self.value, self.mask, query_length=3, unpad_input_func=_unpad_input, is_causal=False
        )
        # The key mask says nothing about the queries: every query is kept, even for the padded sequence
        self.assertEqual(query.shape[0], 6)
        self.assertEqual(cu_seqlens_q.tolist(), [0, 3, 6])
        self.assertEqual(max_q, 3)
        self.assertEqual(cu_seqlens_k.tolist(), [0, 6, 10])
        self.assertEqual(key.shape[0], 10)
        torch.testing.assert_close(_pad_input(query, indices_q, 2, 3), self.query)

    def test_causal_keeps_left_padding_assumption(self):
        # Causal attention (e.g. decoding with a cache): the queries are the last positions of the keys, so they are
        # unpadded with the end of the key mask, as before
        query, _, _, _, (cu_seqlens_q, _), (max_q, _) = _upad_input(
            self.query, self.key, self.value, self.mask, query_length=3, unpad_input_func=_unpad_input
        )
        self.assertEqual(cu_seqlens_q.tolist(), [0, 3, 4])
        self.assertEqual(query.shape[0], 4)
        self.assertEqual(max_q, 3)
