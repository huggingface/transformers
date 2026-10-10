# Copyright 2026 The HuggingFace Inc. team.
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

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.integrations import sdpa_attention
    from transformers.integrations.sdpa_attention import sdpa_attention_forward


@require_torch
class SdpaNpuAttentionMaskTest(unittest.TestCase):
    """
    On Ascend NPU, a non-boolean `attention_mask` is booleanized so that sdpa can dispatch to
    `FlashAttentionScore`. That is only valid for masks with boolean semantics (`0` keeps a position,
    `-inf` / `torch.finfo(dtype).min` masks it out). Masks that are an additive attention bias carry
    information in the magnitude of every entry and must reach sdpa unchanged.

    These tests patch the NPU flag, so they exercise the branch on any device.
    """

    def _mask_seen_by_sdpa(self, attention_mask):
        batch, num_heads, seq_len, head_dim = 1, 2, 4, 8
        query = torch.randn(batch, num_heads, seq_len, head_dim)
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.randn(batch, num_heads, seq_len, head_dim)
        captured = {}

        def _capture(query, key, value, attn_mask=None, **kwargs):
            captured["attn_mask"] = attn_mask
            return torch.zeros_like(query)

        with patch.object(sdpa_attention, "_is_torch_npu_available", True):
            with patch.object(torch.nn.functional, "scaled_dot_product_attention", _capture):
                sdpa_attention_forward(torch.nn.Module(), query, key, value, attention_mask, is_causal=False)
        return captured["attn_mask"]

    def test_additive_bias_mask_is_left_unchanged(self):
        # Relative position biases (e.g. Parakeet) are passed through `attention_mask`; booleanizing
        # them maps every non-zero entry to the same value and silently drops the bias.
        bias = torch.randn(1, 2, 4, 4) * 10.0
        seen = self._mask_seen_by_sdpa(bias.clone())

        self.assertEqual(seen.dtype, bias.dtype)
        torch.testing.assert_close(seen, bias)

    def test_neg_inf_mask_is_booleanized(self):
        mask = torch.zeros(1, 2, 4, 4)
        mask[..., 2:] = float("-inf")
        seen = self._mask_seen_by_sdpa(mask.clone())

        self.assertEqual(seen.dtype, torch.bool)
        # `0` means "attend" and maps to True, `-inf` means "masked out" and maps to False.
        self.assertTrue(seen[..., :2].all())
        self.assertFalse(seen[..., 2:].any())

    def test_dtype_min_mask_is_booleanized(self):
        mask = torch.zeros(1, 2, 4, 4)
        mask[..., 3:] = torch.finfo(mask.dtype).min
        seen = self._mask_seen_by_sdpa(mask.clone())

        self.assertEqual(seen.dtype, torch.bool)
        self.assertTrue(seen[..., :3].all())
        self.assertFalse(seen[..., 3:].any())
