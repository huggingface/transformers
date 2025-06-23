# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
from parameterized import parameterized

from transformers import is_flash_attn_v2_available, is_torch_available
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)


@require_torch
@require_flash_attn
@require_torch_gpu
@slow
class FlashAttentionV2Test(unittest.TestCase):
    """Tests for Flash Attention v2 implementation."""

    def setUp(self):
        if not is_torch_available() or not is_flash_attn_v2_available():
            self.skipTest("PyTorch or Flash Attention v2 is not available")

        # Set deterministic mode for testing
        self.deterministic = True
        self.original_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = self.deterministic

    def tearDown(self):
        # Restore original deterministic setting
        torch.backends.cudnn.deterministic = self.original_deterministic

    def test_flash_attn_v2_forward(self):
        """Test that Flash Attention v2 forward pass matches the expected output."""
        from transformers.modeling_flash_attention_utils import _flash_attention_v2_forward

        batch_size = 2
        seq_len = 32
        num_heads = 4
        head_dim = 64
        hidden_size = num_heads * head_dim

        # Create random input tensors
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )

        # Run Flash Attention v2 forward pass
        output, _ = _flash_attention_v2_forward(
            query,
            key,
            value,
            dropout=0.0,
            causal=True,
            window_size=(-1, -1),  # No windowing
            alibi_slopes=None,
            deterministic=self.deterministic,
        )

        # Verify output shape
        self.assertEqual(output.shape, (batch_size, seq_len, num_heads, head_dim))

    def test_flash_attn_v2_with_padding(self):
        """Test Flash Attention v2 with padded sequences."""
        from transformers.modeling_flash_attention_utils import _flash_attention_v2_forward

        batch_size = 2
        seq_len = 32
        num_heads = 4
        head_dim = 64
        hidden_size = num_heads * head_dim

        # Create random input tensors with padding
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )

        # Create attention mask (second example has padding at the end)
        attention_mask = torch.ones(batch_size, seq_len, device=torch_device, dtype=torch.bool)
        attention_mask[1, -4:] = 0  # Mask out last 4 tokens of second example

        # Run Flash Attention v2 forward pass with padding
        output, _ = _flash_attention_v2_forward(
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=self.deterministic,
        )

        # Verify output shape
        self.assertEqual(output.shape, (batch_size, seq_len, num_heads, head_dim))

    def test_flash_attn_v2_with_sliding_window(self):
        """Test Flash Attention v2 with sliding window attention."""
        from transformers.modeling_flash_attention_utils import _flash_attention_v2_forward

        batch_size = 2
        seq_len = 32
        num_heads = 4
        head_dim = 64
        window_size = 8  # Smaller window size

        # Create random input tensors
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )

        # Run Flash Attention v2 forward pass with sliding window
        output, _ = _flash_attention_v2_forward(
            query,
            key,
            value,
            dropout=0.0,
            causal=True,
            window_size=(window_size, window_size),
            alibi_slopes=None,
            deterministic=self.deterministic,
        )

        # Verify output shape
        self.assertEqual(output.shape, (batch_size, seq_len, num_heads, head_dim))

    @parameterized.expand([(True,), (False,)])
    def test_flash_attn_v2_deterministic(self, deterministic):
        """Test that Flash Attention v2 produces deterministic results when requested."""
        from transformers.modeling_flash_attention_utils import _flash_attention_v2_forward

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 64

        # Create random input tensors
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        key = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=torch_device, dtype=torch.float16
        )

        # Set dropout to non-zero to test non-deterministic behavior
        dropout = 0.1 if not deterministic else 0.0

        # Run forward pass twice
        torch.manual_seed(42)
        output1, _ = _flash_attention_v2_forward(
            query,
            key,
            value,
            dropout=dropout,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
        )

        torch.manual_seed(42)
        output2, _ = _flash_attention_v2_forward(
            query,
            key,
            value,
            dropout=dropout,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
        )

        # Check if outputs match (should match if deterministic=True or dropout=0.0)
        if deterministic or dropout == 0.0:
            self.assertTrue(torch.allclose(output1, output2, atol=1e-5, rtol=1e-5))
        else:
            # With dropout and non-deterministic, they should be different
            self.assertFalse(torch.allclose(output1, output2))
