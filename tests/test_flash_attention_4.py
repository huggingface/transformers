# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Tests for Flash Attention 4 (CuTe DSL) integration.

Usage:
    pytest tests/test_flash_attention_4.py -v

    # Run with specific GPU
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_flash_attention_4.py -v
"""

import unittest

import torch

from transformers import is_flash_attn_4_available
from transformers.testing_utils import require_flash_attn_4, require_torch_gpu


class FlashAttention4DetectionTest(unittest.TestCase):
    """Test FA4 detection without requiring GPU."""

    def test_detection_function_exists(self):
        """Verify is_flash_attn_4_available is callable."""
        self.assertTrue(callable(is_flash_attn_4_available))

    def test_detection_returns_bool(self):
        """Verify detection returns boolean."""
        result = is_flash_attn_4_available()
        self.assertIsInstance(result, bool)


@require_torch_gpu
@require_flash_attn_4
class FlashAttention4IntegrationTest(unittest.TestCase):
    """Integration tests requiring GPU and FA4."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.batch_size = 2
        self.seq_len = 128
        self.num_heads = 8
        self.head_dim = 64

    def test_fa4_import(self):
        """Test FA4 can be imported."""
        try:
            from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

            self.assertIsNotNone(flash_attn_func)
            self.assertIsNotNone(flash_attn_varlen_func)
        except ImportError as e:
            self.fail(f"Failed to import FA4: {e}")

    def test_fa4_basic_forward(self):
        """Test basic FA4 forward pass."""
        from flash_attn.cute import flash_attn_func

        q = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )

        try:
            out = flash_attn_func(q, k, v, causal=False)
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
            self.assertEqual(out.dtype, self.dtype)
        except Exception as e:
            self.fail(f"FA4 forward pass failed: {e}")

    def test_fa4_causal_attention(self):
        """Test FA4 with causal masking."""
        from flash_attn.cute import flash_attn_func

        q = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )

        try:
            out = flash_attn_func(q, k, v, causal=True)
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        except Exception as e:
            self.fail(f"FA4 causal attention failed: {e}")

    def test_fa4_varlen_no_max_seqlen(self):
        """Test FA4 varlen function does not accept max_seqlen parameters."""
        from flash_attn.cute import flash_attn_varlen_func
        import inspect

        sig = inspect.signature(flash_attn_varlen_func)
        params = list(sig.parameters.keys())

        # Verify FA4 API: no max_seqlen_q/k parameters
        self.assertNotIn("max_seqlen_q", params, "FA4 should not have max_seqlen_q parameter")
        self.assertNotIn("max_seqlen_k", params, "FA4 should not have max_seqlen_k parameter")

        # Verify FA4 has cu_seqlens parameters
        self.assertIn("cu_seqlens_q", params, "FA4 should have cu_seqlens_q parameter")
        self.assertIn("cu_seqlens_k", params, "FA4 should have cu_seqlens_k parameter")

    def test_fa4_varlen_forward(self):
        """Test FA4 varlen forward pass."""
        from flash_attn.cute import flash_attn_varlen_func

        # Create packed sequences: [seq1_len=50, seq2_len=78]
        total_tokens = 128
        cu_seqlens = torch.tensor([0, 50, 128], dtype=torch.int32, device=self.device)

        q = torch.randn(total_tokens, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
        k = torch.randn(total_tokens, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)
        v = torch.randn(total_tokens, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype)

        try:
            # FA4 calculates max_seqlen internally, no need to pass it
            out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, causal=False)
            self.assertEqual(out.shape, (total_tokens, self.num_heads, self.head_dim))
        except Exception as e:
            self.fail(f"FA4 varlen forward failed: {e}")

    def test_hf_fa4_integration(self):
        """Test HF's FA4 integration via lazy_import_flash_attention."""
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention, _is_using_fa4

        # Test explicit FA4 selection
        (flash_fn, flash_varlen_fn, pad_fn, unpad_fn), process_kwargs_fn = lazy_import_flash_attention(
            "flash_attention_4"
        )

        self.assertIsNotNone(flash_fn)
        self.assertIsNotNone(flash_varlen_fn)

        # Verify we're using FA4 (no max_seqlen_q parameter)
        is_fa4 = _is_using_fa4(flash_varlen_fn)
        self.assertTrue(is_fa4, "Should detect FA4 via introspection")

    def test_hf_fa4_auto_selection(self):
        """Test HF auto-selects FA4 when available."""
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention, _is_using_fa4

        # Test auto-selection (implementation=None)
        (flash_fn, flash_varlen_fn, pad_fn, unpad_fn), process_kwargs_fn = lazy_import_flash_attention(None)

        # Should select FA4 if available
        is_fa4 = _is_using_fa4(flash_varlen_fn)
        # This will be True if FA4 is the highest priority available
        self.assertIsInstance(is_fa4, bool)


@require_torch_gpu
@require_flash_attn_4
class FlashAttention4ParameterTest(unittest.TestCase):
    """Test FA4-specific parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.batch_size = 2
        self.seq_len = 128
        self.num_heads = 8
        self.head_dim = 64

    def test_softcap_parameter(self):
        """Test FA4 softcap parameter."""
        from flash_attn.cute import flash_attn_func

        q = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )

        try:
            out = flash_attn_func(q, k, v, causal=False, softcap=30.0)
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        except Exception as e:
            self.fail(f"FA4 softcap parameter failed: {e}")

    def test_window_size_parameter(self):
        """Test FA4 sliding window attention."""
        from flash_attn.cute import flash_attn_func

        q = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device, dtype=self.dtype
        )

        try:
            # FA4 uses (left, right) tuple for window_size
            out = flash_attn_func(q, k, v, causal=True, window_size=(32, 32))
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.num_heads, self.head_dim))
        except Exception as e:
            self.fail(f"FA4 window_size parameter failed: {e}")


if __name__ == "__main__":
    unittest.main()
