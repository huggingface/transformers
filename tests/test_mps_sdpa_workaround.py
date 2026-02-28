"""
Tests for the MPS SDPA correctness workaround (pytorch/pytorch#174861).

This workaround addresses a silent correctness bug in PyTorch's
sdpa_vector_2pass_mps kernel affecting versions 2.8.0 through 2.10.x.
The bug produces wrong results for bidirectional attention under specific
conditions on Apple Silicon (MPS backend).

See: https://github.com/huggingface/transformers/issues/44247
"""

import unittest
from unittest.mock import patch

import torch

from transformers.integrations.sdpa_attention import (
    _MPS_SDPA_BUG_HEAD_DIMS,
    _needs_mps_sdpa_workaround,
    sdpa_attention_forward,
)


class TestNeedsMpsSdpaWorkaround(unittest.TestCase):
    """Test the condition-checking function for the MPS SDPA workaround."""

    def _make_tensors(
        self,
        batch=1,
        num_query_heads=8,
        num_key_heads=8,
        query_len=1,
        key_len=2048,
        head_dim=128,
        dtype=torch.float16,
        device="mps",
    ):
        """Helper to create query and key tensors with specified shapes."""
        query = torch.randn(batch, num_query_heads, query_len, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, num_key_heads, key_len, head_dim, dtype=dtype, device=device)
        return query, key

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_triggers_on_mps_with_bug_conditions(self):
        """Workaround should trigger when all bug conditions are met on MPS."""
        query, key = self._make_tensors(
            query_len=1, key_len=2048, head_dim=128, dtype=torch.float16
        )
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertTrue(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_no_trigger_float32(self):
        """float32 dtype should not trigger the workaround (bug only affects non-float32)."""
        query, key = self._make_tensors(dtype=torch.float32)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_no_trigger_causal(self):
        """Causal attention should not trigger the workaround."""
        query, key = self._make_tensors()
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=True))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_no_trigger_float_mask(self):
        """A float attention mask already routes to sdpa_general_mps, no workaround needed."""
        query, key = self._make_tensors()
        float_mask = torch.zeros(1, 1, 1, 2048, dtype=torch.float16, device="mps")
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, float_mask, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_triggers_with_bool_mask(self):
        """Boolean masks trigger the buggy kernel, so workaround should activate."""
        query, key = self._make_tensors()
        bool_mask = torch.ones(1, 1, 1, 2048, dtype=torch.bool, device="mps")
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertTrue(_needs_mps_sdpa_workaround(query, key, bool_mask, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_no_trigger_query_len_too_large(self):
        """query_len > 8 should not trigger (bug only affects query_len <= 8)."""
        query, key = self._make_tensors(query_len=16, key_len=2048)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_no_trigger_query_longer_than_key(self):
        """query_len > key_len should not trigger."""
        query, key = self._make_tensors(query_len=8, key_len=4)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_no_trigger_wrong_head_dim(self):
        """Head dims not in {64, 96, 128} should not trigger."""
        query, key = self._make_tensors(head_dim=32)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_no_trigger_short_key_len(self):
        """key_len < 1024 with equal heads should not trigger."""
        query, key = self._make_tensors(key_len=512)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_triggers_gqa_long_key(self):
        """GQA (num_key_heads < num_query_heads) with key_len >= 4096 should trigger."""
        query, key = self._make_tensors(num_query_heads=8, num_key_heads=2, key_len=4096)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertTrue(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_all_affected_head_dims(self):
        """All affected head dims should trigger the workaround."""
        for hd in _MPS_SDPA_BUG_HEAD_DIMS:
            query, key = self._make_tensors(head_dim=hd)
            with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
                self.assertTrue(
                    _needs_mps_sdpa_workaround(query, key, None, is_causal=False),
                    f"Should trigger for head_dim={hd}",
                )

    def test_no_trigger_when_not_affected_version(self):
        """Should not trigger when PyTorch version is not affected."""
        query = torch.randn(1, 8, 1, 128, dtype=torch.float16)
        key = torch.randn(1, 8, 2048, 128, dtype=torch.float16)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", False):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))

    def test_no_trigger_non_mps_device(self):
        """Should not trigger on CPU even with all other conditions met."""
        query = torch.randn(1, 8, 1, 128, dtype=torch.float16)
        key = torch.randn(1, 8, 2048, 128, dtype=torch.float16)
        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            self.assertFalse(_needs_mps_sdpa_workaround(query, key, None, is_causal=False))


class TestSdpaAttentionForwardMpsWorkaround(unittest.TestCase):
    """End-to-end tests for the SDPA forward function with MPS workaround."""

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_workaround_produces_correct_output(self):
        """
        Verify that the workaround produces the same output as float32 reference.

        This is the core correctness test: run SDPA with the workaround active
        and compare against a float32 reference (which is not affected by the bug).
        """
        torch.manual_seed(42)
        batch, heads, q_len, kv_len, head_dim = 1, 8, 1, 2048, 128
        dtype = torch.float16

        # Create inputs
        query_fp32 = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float32, device="mps")
        key_fp32 = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float32, device="mps")
        value_fp32 = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float32, device="mps")

        # Float32 reference (not affected by the bug)
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            query_fp32, key_fp32, value_fp32, is_causal=False
        )

        # Run with workaround using fp16
        query_fp16 = query_fp32.to(dtype)
        key_fp16 = key_fp32.to(dtype)
        value_fp16 = value_fp32.to(dtype)

        module = torch.nn.Module()
        module.is_causal = False

        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            output, _ = sdpa_attention_forward(
                module, query_fp16, key_fp16, value_fp16,
                attention_mask=None, is_causal=False,
            )

        # output shape: (batch, q_len, heads, head_dim) due to transpose in sdpa_attention_forward
        output_compare = output.transpose(1, 2).to(torch.float32)  # back to (batch, heads, q_len, head_dim)
        ref_compare = ref_output

        # Check outputs are close (allowing for fp16 precision loss)
        # fp16 SDPA vs fp32 reference: typical error is ~0.01, allow 0.02 for margin
        torch.testing.assert_close(output_compare, ref_compare, atol=0.02, rtol=0.02)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_workaround_creates_float_mask(self):
        """Verify that the workaround creates a float mask when conditions are met."""
        torch.manual_seed(42)
        batch, heads, q_len, kv_len, head_dim = 1, 8, 1, 2048, 128

        query = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float16, device="mps")
        key = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float16, device="mps")
        value = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float16, device="mps")

        module = torch.nn.Module()
        module.is_causal = False

        # Patch SDPA to capture the mask argument
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        captured_masks = []

        def capturing_sdpa(*args, **kwargs):
            captured_masks.append(kwargs.get("attn_mask"))
            return original_sdpa(*args, **kwargs)

        with (
            patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True),
            patch("torch.nn.functional.scaled_dot_product_attention", capturing_sdpa),
        ):
            sdpa_attention_forward(module, query, key, value, attention_mask=None, is_causal=False)

        self.assertEqual(len(captured_masks), 1)
        mask = captured_masks[0]
        self.assertIsNotNone(mask, "Workaround should have created a float mask")
        self.assertEqual(mask.dtype, torch.float16)
        self.assertEqual(mask.shape, (batch, 1, q_len, kv_len))
        # All zeros (semantically equivalent to no mask)
        self.assertTrue(torch.all(mask == 0))


if __name__ == "__main__":
    unittest.main()

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_workaround_produces_correct_output_bfloat16(self):
        """
        Verify correctness with bfloat16 (upstream MRE uses bfloat16).
        """
        torch.manual_seed(42)
        batch, heads, q_len, kv_len, head_dim = 1, 8, 1, 2048, 128
        dtype = torch.bfloat16

        query_fp32 = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float32, device="mps")
        key_fp32 = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float32, device="mps")
        value_fp32 = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float32, device="mps")

        ref_output = torch.nn.functional.scaled_dot_product_attention(
            query_fp32, key_fp32, value_fp32, is_causal=False
        )

        query_bf16 = query_fp32.to(dtype)
        key_bf16 = key_fp32.to(dtype)
        value_bf16 = value_fp32.to(dtype)

        module = torch.nn.Module()
        module.is_causal = False

        with patch("transformers.integrations.sdpa_attention._mps_sdpa_bug_affected", True):
            output, _ = sdpa_attention_forward(
                module, query_bf16, key_bf16, value_bf16,
                attention_mask=None, is_causal=False,
            )

        output_compare = output.transpose(1, 2).to(torch.float32)
        # bfloat16 has less precision than float16, slightly wider tolerance
        torch.testing.assert_close(output_compare, ref_output, atol=0.05, rtol=0.05)
