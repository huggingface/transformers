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
from types import SimpleNamespace

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch

    from transformers.vision_utils import get_vision_attention_seqlens, get_vision_cu_seqlens


@require_torch
class VisionUtilsTest(unittest.TestCase):
    def test_attention_seqlens_computes_python_max_for_flash_attention(self):
        grid_thw = torch.tensor([[1, 2, 3], [2, 1, 4]])
        config = SimpleNamespace(_attn_implementation="flash_attention_2")

        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, config)

        torch.testing.assert_close(cu_seqlens, torch.tensor([0, 6, 10, 14], dtype=torch.int32))
        self.assertEqual(max_seqlen, 6)
        self.assertIsInstance(max_seqlen, int)

    def test_attention_seqlens_skips_max_for_non_flash_attention(self):
        grid_thw = torch.tensor([[1, 2, 3]])
        config = SimpleNamespace(_attn_implementation="sdpa")

        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, config)

        torch.testing.assert_close(cu_seqlens, get_vision_cu_seqlens(grid_thw))
        self.assertIsNone(max_seqlen)

    def test_attention_seqlens_pops_precomputed_kwargs(self):
        grid_thw = torch.tensor([[1, 2, 3]])
        config = SimpleNamespace(_attn_implementation="sdpa")
        precomputed_cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
        kwargs = {"cu_seqlens": precomputed_cu_seqlens, "max_seqlen": 8}

        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, config, kwargs=kwargs)

        self.assertIs(cu_seqlens, precomputed_cu_seqlens)
        self.assertEqual(max_seqlen, 8)
        self.assertEqual(kwargs, {})
