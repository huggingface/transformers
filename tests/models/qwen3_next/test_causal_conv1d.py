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

import itertools
import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers.models.qwen3_next.modeling_qwen3_next import causal_conv1d_fn, causal_conv1d_update


@require_torch
class CausalConv1dCpuTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("torch.backends.mkldnn.is_available", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_onednn_build_retains_conv1d(self):
        hidden_states = torch.randn(2, 7, 3)
        weight = torch.randn(7, 4)
        with (
            patch("torch.backends.mkldnn.is_available", return_value=True),
            patch("transformers.models.qwen3_next.modeling_qwen3_next.F.conv1d", wraps=F.conv1d) as conv,
        ):
            causal_conv1d_fn(hidden_states, weight)
            conv.assert_called_once()
            conv.reset_mock()
            causal_conv1d_update(hidden_states, torch.zeros(2, 7, 4), weight)
            conv.assert_called_once()

    def test_cpu_autocast_retains_conv1d(self):
        hidden_states = torch.randn(2, 7, 3)
        weight = torch.randn(7, 4)
        with (
            torch.autocast("cpu", dtype=torch.bfloat16),
            patch("transformers.models.qwen3_next.modeling_qwen3_next.F.conv1d", wraps=F.conv1d) as conv,
        ):
            expected = F.conv1d(hidden_states, weight[:, None], padding=3, groups=7)[:, :, :3].float()
            conv.reset_mock()
            actual = causal_conv1d_fn(hidden_states, weight)
            conv.assert_called_once()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            conv.reset_mock()
            causal_conv1d_update(hidden_states, torch.zeros(2, 7, 4), weight)
            conv.assert_called_once()

    @parameterized.expand(itertools.product([1, 2, 4, 5], [1, 7, 17], [False, True]))
    def test_forward_matches_conv1d(self, kernel_size, seq_len, bias):
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                # Exercise non-contiguous inputs and weights, including inputs shorter than the kernel.
                hidden_states = torch.randn(2, 7, seq_len * 2, dtype=dtype)[:, :, ::2]
                weight = torch.randn(7, kernel_size * 2, dtype=dtype)[:, ::2]
                bias_tensor = torch.randn(7, dtype=dtype) if bias else None
                expected = F.conv1d(hidden_states, weight[:, None], bias_tensor, padding=kernel_size - 1, groups=7)[
                    :, :, :seq_len
                ]
                for activation in [None, "silu"]:
                    actual = causal_conv1d_fn(hidden_states, weight, bias_tensor, activation)
                    reference = F.silu(expected) if activation else expected
                    torch.testing.assert_close(actual, reference)

    @parameterized.expand(itertools.product([1, 2, 4, 5], [1, 3, 7], [False, True]))
    def test_update_matches_conv1d_and_cache(self, kernel_size, seq_len, bias):
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                torch.manual_seed(1)
                hidden_states = torch.randn(2, 7, seq_len, dtype=dtype)
                weight = torch.randn(7, kernel_size, dtype=dtype)
                bias_tensor = torch.randn(7, dtype=dtype) if bias else None
                for state_len in {kernel_size, kernel_size + 2}:
                    initial_state = torch.randn(2, 7, state_len, dtype=dtype)
                    joined = torch.cat([initial_state, hidden_states], dim=-1)
                    expected = F.conv1d(joined, weight[:, None], bias_tensor, groups=7)[:, :, -seq_len:]
                    for activation in [None, "silu"]:
                        state = initial_state.clone()
                        actual = causal_conv1d_update(hidden_states, state, weight, bias_tensor, activation)
                        reference = F.silu(expected) if activation else expected
                        torch.testing.assert_close(actual, reference)
                        torch.testing.assert_close(state, joined[:, :, -state_len:], rtol=0, atol=0)

    def test_prefill_and_cached_chunks_match_full_sequence(self):
        torch.manual_seed(2)
        hidden_states = torch.randn(2, 7, 17)
        weight = torch.randn(7, 4)
        expected = causal_conv1d_fn(hidden_states, weight, activation="silu")
        parts = [causal_conv1d_fn(hidden_states[:, :, :5], weight, activation="silu")]
        state = hidden_states[:, :, 1:5].clone()
        start = 5
        for length in [1, 3, 8]:
            parts.append(
                causal_conv1d_update(hidden_states[:, :, start : start + length], state, weight, activation="silu")
            )
            start += length
        torch.testing.assert_close(torch.cat(parts, dim=-1), expected)

    @parameterized.expand([("prefill",), ("update",)])
    def test_gradients_match_conv1d(self, mode):
        torch.manual_seed(3)
        hidden_states = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(3, dtype=torch.float64, requires_grad=True)
        if mode == "update":
            initial_state = torch.randn(2, 3, 4, dtype=torch.float64)
            actual = causal_conv1d_update(hidden_states, initial_state.clone(), weight, bias, "silu")
            expected = F.silu(
                F.conv1d(torch.cat([initial_state, hidden_states], dim=-1), weight[:, None], bias, groups=3)
            )[:, :, -7:]
        else:
            actual = causal_conv1d_fn(hidden_states, weight, bias, "silu")
            expected = F.silu(F.conv1d(hidden_states, weight[:, None], bias, padding=3, groups=3)[:, :, :7])
        grad_output = torch.randn_like(actual)
        inputs = (hidden_states, weight, bias)
        actual_grads = torch.autograd.grad(actual, inputs, grad_output)
        expected_grads = torch.autograd.grad(expected, inputs, grad_output)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
