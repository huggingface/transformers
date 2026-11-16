# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Tests for the MPS SDPA float-mask workaround (pytorch/pytorch#174861, #44247).

The buggy `sdpa_vector_2pass_mps` kernel is selected only inside a narrow,
enumerated window, so the gate predicate is tested against an independent
encoding of that window over a grid of shapes, dtypes, masks and head
configurations. The on-device dispatch test is MPS-only and skips cleanly
everywhere else.
"""

import itertools
import unittest

import torch

from transformers.integrations.sdpa_attention import (
    _mps_sdpa_general_path_mask,
    _mps_sdpa_needs_general_path,
    sdpa_attention_forward,
)
from transformers.testing_utils import require_torch, require_torch_mps


def _expected_needs_general_path(query, key, value, attention_mask, is_causal, num_key_value_groups):
    """Independent encoding of the upstream bug window, written from the issue text."""
    if is_causal:
        return False
    if query.dtype == torch.float32:
        return False
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        return False
    q_len, kv_len = query.shape[2], key.shape[2]
    if q_len > 8 or q_len > kv_len:
        return False
    if query.shape[-1] != value.shape[-1] or query.shape[-1] not in (64, 96, 128):
        return False
    # Issue text, verbatim: "key sequence length >= 1024 OR (number of key
    # heads < number of query heads AND key sequence length >= 4096)".
    return kv_len >= 1024 or (num_key_value_groups > 1 and kv_len >= 4096)


@require_torch
class MpsSdpaGatePredicateTest(unittest.TestCase):
    def test_predicate_matches_issue_window_on_grid(self):
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        mask_kinds = ["none", "bool", "float"]
        q_lens = [1, 4, 8, 9]
        kv_lens = [512, 1023, 1024, 2048, 4095, 4096, 8192]
        head_dims = [32, 64, 96, 128, 256]
        value_dims = [64, 128]
        groups = [1, 2, 4]
        checked = 0
        for dtype, causal, mask_kind, q_len, kv_len, head_dim, value_dim, group in itertools.product(
            dtypes, [False, True], mask_kinds, q_lens, kv_lens, head_dims, value_dims, groups
        ):
            query = torch.empty(1, 4, q_len, head_dim, dtype=dtype)
            key = torch.empty(1, 4 // group, kv_len, head_dim, dtype=dtype)
            value = torch.empty(1, 4 // group, kv_len, value_dim, dtype=dtype)
            if mask_kind == "none":
                mask = None
            elif mask_kind == "bool":
                mask = torch.ones(1, 1, q_len, kv_len, dtype=torch.bool)
            else:
                mask = torch.zeros(1, 1, q_len, kv_len, dtype=dtype)
            expected = _expected_needs_general_path(query, key, value, mask, causal, group)
            actual = _mps_sdpa_needs_general_path(query, key, value, mask, causal, group)
            self.assertEqual(
                actual,
                expected,
                f"dtype={dtype} causal={causal} mask={mask_kind} q={q_len} kv={kv_len} "
                f"head_dim={head_dim} value_dim={value_dim} groups={group}",
            )
            checked += 1
        self.assertEqual(checked, 3 * 2 * 3 * 4 * 7 * 5 * 2 * 3)

    def test_general_path_mask_is_equivalent(self):
        # A materialized float mask must not change the attention result:
        # zeros mask == no mask, and the additive form of a bool mask == the bool mask.
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            query = torch.randn(2, 4, 4, 64, dtype=dtype)
            key = torch.randn(2, 4, 128, 64, dtype=dtype)
            value = torch.randn(2, 4, 128, 64, dtype=dtype)
            no_mask = torch.nn.functional.scaled_dot_product_attention(query, key, value)
            zeros_mask = _mps_sdpa_general_path_mask(query, key, None)
            self.assertEqual(zeros_mask.dtype, dtype)
            with_zeros = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=zeros_mask
            )
            self.assertTrue(torch.equal(no_mask, with_zeros))

            bool_mask = torch.ones(2, 1, 4, 128, dtype=torch.bool)
            bool_mask[..., 100:] = False
            additive = _mps_sdpa_general_path_mask(query, key, bool_mask)
            with_bool = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=bool_mask
            )
            with_additive = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=additive
            )
            self.assertTrue(torch.equal(with_bool, with_additive))


class _DummyModule(torch.nn.Module):
    def __init__(self, num_key_value_groups=1):
        super().__init__()
        self.num_key_value_groups = num_key_value_groups
        self.config = None


@require_torch
class MpsSdpaDispatchTest(unittest.TestCase):
    def _dispatch(self, device, dtype=torch.bfloat16, kv_len=1024):
        torch.manual_seed(0)
        query = torch.randn(1, 4, 4, 64, dtype=dtype, device=device)
        key = torch.randn(1, 4, kv_len, 64, dtype=dtype, device=device)
        value = torch.randn(1, 4, kv_len, 64, dtype=dtype, device=device)
        module = _DummyModule()
        output, _ = sdpa_attention_forward(
            module, query, key, value, None, dropout=0.0, scaling=None, is_causal=False
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float().cpu(), key.float().cpu(), value.float().cpu()
        )
        # sdpa_attention_forward returns (batch, q_len, heads, head_dim)
        return output.transpose(1, 2), reference

    def test_dispatch_on_cpu_matches_reference(self):
        # The workaround is MPS-gated, so on CPU the dispatch must be inert
        # and agree with the plain SDPA result even inside the bug window.
        output, reference = self._dispatch("cpu")
        self.assertLess((output.float() - reference).abs().max().item(), 0.05)

    @require_torch_mps
    def test_dispatch_on_mps_matches_reference_in_bug_window(self):
        # On an affected torch build (>= 2.8, < 2.11) the unpatched dispatch
        # diverges from the fp32 reference by ~0.5 in this window; the
        # workaround must bring it back to the healthy-kernel noise floor.
        output, reference = self._dispatch("mps")
        self.assertLess((output.float().cpu() - reference).abs().max().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
