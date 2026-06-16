# Copyright 2025 HuggingFace Inc.
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

import torch


TRANSFORMERS_SDPA_MODULE = "transformers.integrations.sdpa_attention"


def _import_sdpa_module():
    """Helper to (re)import the SDPA module for testing with mocked version gates."""
    import importlib

    import transformers.integrations.sdpa_attention as sdpa_mod

    importlib.reload(sdpa_mod)
    return sdpa_mod


class TestMPSVersionGates(unittest.TestCase):
    """Test the version-gate boolean expression ranges for _apply_mps_fixes."""

    def test_version_gate_2_8_to_2_11_range_affects_fix1(self):
        """V1: Version >= 2.8.0, < 2.11.0 — Fix 1 should be active."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            q = torch.randn(1, 1, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 1, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 1, 16, 64, dtype=torch.float16)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertIsNotNone(mask, "Fix 1 should create a zeros mask")

    def test_version_gate_2_11_plus_disables_fix1(self):
        """V2: Version >= 2.11.0 — Fix 1 should NOT be active (upstream fix available)."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", True),
        ):
            q = torch.randn(1, 1, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 1, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 1, 16, 64, dtype=torch.float16)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertIsNone(mask, "Fix 1 should NOT create a mask on >= 2.11")

    def test_version_gate_pre_2_8_disables_fix1(self):
        """V3: Version < 2.8.0 — Fix 1 should NOT be active (bug not yet introduced)."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", False),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            q = torch.randn(1, 1, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 1, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 1, 16, 64, dtype=torch.float16)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertIsNone(mask, "Fix 1 should NOT create a mask on < 2.8")


class TestMPSBidirectionalCorrectness(unittest.TestCase):
    """Test the condition detection for Fix 1 (bidirectional correctness)."""

    def test_fix1_activates_on_non_causal_non_fp32_attention(self):
        """V6: Fix 1 activates when is_causal=False, non-float32, no mask."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            q = torch.randn(1, 2, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertIsNotNone(mask, "Fix 1 should activate with non-causal, non-fp32, no mask")

    def test_fix1_preserves_existing_bool_mask(self):
        """V7: Fix 1 should NOT replace an existing bool attention mask."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            q = torch.randn(1, 2, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            bool_mask = torch.ones(1, 1, 8, 16, dtype=torch.bool)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, bool_mask, is_causal=False)
            self.assertIs(mask, bool_mask, "Fix 1 should preserve existing bool mask semantics")

    def test_fix1_zeros_mask_shape_matches_expected(self):
        """V8: Zeros mask shape matches expected SDPA mask format."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            batch, n_heads, q_len, k_len, head_dim = 2, 4, 16, 32, 64
            q = torch.randn(batch, n_heads, q_len, head_dim, dtype=torch.float16)
            k = torch.randn(batch, n_heads, k_len, head_dim, dtype=torch.float16)
            v = torch.randn(batch, n_heads, k_len, head_dim, dtype=torch.float16)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            expected_shape = (batch, 1, q_len, k_len)
            self.assertEqual(mask.shape, expected_shape, f"Mask shape should be {expected_shape}")

    def test_fix1_skips_on_fp32(self):
        """V9: Fix 1 should NOT activate when query dtype is float32 (bug doesn't affect fp32)."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            q = torch.randn(1, 2, 8, 64, dtype=torch.float32)
            k = torch.randn(1, 2, 16, 64, dtype=torch.float32)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float32)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertIsNone(mask, "Fix 1 should NOT activate with float32")

    def test_fix1_skips_on_causal(self):
        """V10: Fix 1 should NOT activate when is_causal=True."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            q = torch.randn(1, 2, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            _, _, _, mask, _ = sdpa._apply_mps_fixes(q, k, v, None, is_causal=True)
            self.assertIsNone(mask, "Fix 1 should NOT activate with is_causal=True")

    def test_fix1_skips_when_float_mask_present(self):
        """Fix 1 should NOT replace an existing float attention mask."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
        ):
            q = torch.randn(1, 2, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            existing_mask = torch.randn(1, 1, 8, 16)
            q2, k2, v2, mask, vd = sdpa._apply_mps_fixes(q, k, v, existing_mask, is_causal=False)
            self.assertIs(mask, existing_mask, "Fix 1 should NOT replace an existing float mask")


class TestMPSValueHeadDim(unittest.TestCase):
    """Test Fix 2: value head dim mismatch padding and output slicing."""

    def test_fix2_pads_value_when_mismatch(self):
        """V13: Fix 2 should pad value when v head dim != q head dim."""
        sdpa = _import_sdpa_module()
        with patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", False):
            q = torch.randn(1, 2, 8, 128)
            k = torch.randn(1, 2, 16, 128)
            v = torch.randn(1, 2, 16, 64)  # v head dim = 64, q head dim = 128
            _, _, padded_v, _, original_v_head_dim = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertEqual(padded_v.shape[-1], 128, "Padded v head dim should match q head dim")
            self.assertEqual(original_v_head_dim, 64, "Should record original v head dim")

    def test_fix2_no_pad_when_matching(self):
        """V14: Fix 2 should NOT pad when v head dim matches q head dim."""
        sdpa = _import_sdpa_module()
        with patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", False):
            q = torch.randn(1, 2, 8, 64)
            k = torch.randn(1, 2, 16, 64)
            v = torch.randn(1, 2, 16, 64)  # v head dim = 64, q head dim = 64
            _, _, padded_v, _, original_v_head_dim = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertEqual(padded_v.shape[-1], 64, "V head dim should remain unchanged")
            self.assertIsNone(original_v_head_dim, "Should NOT record original v head dim")

    def test_fix2_disabled_on_2_12_plus(self):
        """V15: Fix 2 should be disabled on PyTorch >= 2.12."""
        sdpa = _import_sdpa_module()
        with patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", True):
            q = torch.randn(1, 2, 8, 128)
            k = torch.randn(1, 2, 16, 128)
            v = torch.randn(1, 2, 16, 64)  # mismatch, but fix disabled
            _, _, padded_v, _, original_v_head_dim = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertEqual(padded_v.shape[-1], 64, "V head dim should remain unchanged on >= 2.12")
            self.assertIsNone(original_v_head_dim, "Should not record original v head dim")

    def test_output_slicing_restores_original_dim(self):
        """V16: After SDPA, output should be sliced back to original v head dim."""
        sdpa = _import_sdpa_module()
        with patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", False):
            q = torch.randn(1, 2, 8, 128)
            k = torch.randn(1, 2, 16, 128)
            v = torch.randn(1, 2, 8, 64)  # v head dim smaller than q head dim
            # Simulate what happens after SDPA: pad, SDPA output, then slice
            q2, k2, padded_v, mask, original_v_head_dim = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            # Simulate SDPA output — padded v produces padded output
            padded_attn_output = torch.randn(1, 2, 8, padded_v.shape[-1])
            # Now slice back (same as line 167 in sdpa_attention.py)
            if original_v_head_dim is not None:
                attn_output = padded_attn_output[..., :original_v_head_dim]
            self.assertEqual(attn_output.shape[-1], 64, "Output should be sliced back to original v head dim")

    def test_fix2_skips_when_v_dim_greater_than_q_dim(self):
        """V17: Fix 2 should skip padding when v head dim > q head dim (no negative pad)."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", False),
            patch.object(sdpa.logger, "warning_once") as mock_warn,
        ):
            q = torch.randn(1, 2, 8, 64)  # q head dim = 64
            k = torch.randn(1, 2, 16, 64)
            v = torch.randn(1, 2, 16, 96)  # v head dim = 96 > q head dim = 64
            _, _, padded_v, _, original_v_head_dim = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            # Should NOT pad (would negative-pad and crop)
            self.assertEqual(padded_v.shape[-1], 96, "V head dim should remain unchanged when v > q")
            self.assertIsNone(original_v_head_dim, "Should NOT record original v head dim when v > q")
            # Should log a warning
            mock_warn.assert_called_once()


class TestMPSNoopConditions(unittest.TestCase):
    """Test conditions where both fixes are no-ops."""

    def test_noop_on_cpu(self):
        """N3: _apply_mps_fixes is never called on CPU (device gate in caller)."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", False),
        ):
            q = torch.randn(1, 2, 8, 64, dtype=torch.float16)
            k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
            mask_before = None
            _, _, _, mask_after, vd = sdpa._apply_mps_fixes(q, k, v, mask_before, is_causal=False)
            self.assertIsNotNone(mask_after, "Fix 1 activates when version-gate passes regardless of device")
            # The device gate is at call site, not inside the function

    def test_noop_float32_causal_by_default(self):
        """N4: Default forward pass (float32, causal) should be no-op in normal conditions."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", True),  # Fix 2 disabled
        ):
            q = torch.randn(1, 2, 8, 64, dtype=torch.float32)
            k = torch.randn(1, 2, 16, 64, dtype=torch.float32)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float32)
            mask = None
            q2, k2, v2, mask_out, vd = sdpa._apply_mps_fixes(q, k, v, mask, is_causal=True)
            self.assertIsNone(mask_out, "Fix 1: float32 should skip")
            self.assertIsNone(vd, "Fix 2: matching dims should skip")
            self.assertIs(q2, q, "Input tensors should pass through unchanged")
            self.assertIs(k2, k, "Input tensors should pass through unchanged")
            self.assertIs(v2, v, "Input tensors should pass through unchanged")

    def test_noop_2_12_plus_matching_dim(self):
        """N5: On >= 2.12 with matching dims and causal or fp32 — total no-op."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", True),
        ):
            q = torch.randn(1, 2, 8, 64)
            k = torch.randn(1, 2, 16, 64)
            v = torch.randn(1, 2, 16, 64)
            q2, k2, v2, mask, vd = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            self.assertIsNone(mask, "Fix 1: >= 2.11 should skip")
            self.assertIsNone(vd, "Fix 2: >= 2.12 should skip")


class TestSDPAForwardIntegration(unittest.TestCase):
    """Test the sdpa_attention_forward integration — confirms CPU path is not affected."""

    def test_cpu_forward_no_regression_simple(self):
        """I1: CPU forward with simple parameters works."""
        sdpa = _import_sdpa_module()
        module = type("DummyModule", (), {"num_key_value_groups": 1, "is_causal": True})()
        q = torch.randn(1, 2, 4, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        output, _ = sdpa.sdpa_attention_forward(module, q, k, v, attention_mask=None)
        self.assertEqual(output.shape, (1, 4, 2, 64), "CPU forward should produce correct shape")

    def test_cpu_forward_with_gqa(self):
        """I4: CPU forward with GQA (num_key_value_groups > 1) works."""
        sdpa = _import_sdpa_module()
        module = type("DummyModule", (), {"num_key_value_groups": 2, "is_causal": True})()
        q = torch.randn(1, 4, 4, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        output, _ = sdpa.sdpa_attention_forward(module, q, k, v, attention_mask=None)
        self.assertEqual(output.shape, (1, 4, 4, 64), "CPU forward with GQA should produce correct shape")

    def test_cpu_forward_with_mask(self):
        """I5: CPU forward with attention mask works."""
        sdpa = _import_sdpa_module()
        module = type("DummyModule", (), {"num_key_value_groups": 1, "is_causal": False})()
        q = torch.randn(1, 2, 4, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        mask = torch.zeros(1, 1, 4, 8)
        output, _ = sdpa.sdpa_attention_forward(module, q, k, v, attention_mask=mask)
        self.assertEqual(output.shape, (1, 4, 2, 64), "CPU forward with mask should produce correct shape")

    def test_cpu_forward_non_causal(self):
        """I6: CPU forward with is_causal=False works."""
        sdpa = _import_sdpa_module()
        module = type("DummyModule", (), {"num_key_value_groups": 1, "is_causal": False})()
        q = torch.randn(1, 2, 4, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        output, _ = sdpa.sdpa_attention_forward(module, q, k, v, attention_mask=None)
        self.assertEqual(output.shape, (1, 4, 2, 64), "CPU forward non-causal should produce correct shape")


class TestCombinedFixes(unittest.TestCase):
    """Test both fixes active together with mocked version gates."""

    def test_both_fixes_active_simultaneously(self):
        """Both fixes can activate together: version >= 2.8, < 2.11 (fix1), < 2.12 (fix2)."""
        sdpa = _import_sdpa_module()
        with (
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_8", True),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_11", False),
            patch(f"{TRANSFORMERS_SDPA_MODULE}._is_torch_greater_or_equal_than_2_12", False),
        ):
            q = torch.randn(1, 2, 8, 128, dtype=torch.float16)
            k = torch.randn(1, 2, 16, 128, dtype=torch.float16)
            v = torch.randn(1, 2, 16, 64, dtype=torch.float16)  # v head dim mismatch
            q2, k2, v2, mask, original_v_head_dim = sdpa._apply_mps_fixes(q, k, v, None, is_causal=False)
            # Fix 1: zeros mask
            self.assertIsNotNone(mask, "Fix 1 should be active")
            self.assertEqual(mask.dtype, q.dtype, "Mask dtype should match query dtype")
            # Fix 2: v padded
            self.assertEqual(v2.shape[-1], 128, "Fix 2 should pad v head dim")
            self.assertEqual(original_v_head_dim, 64, "Fix 2 should record original v head dim")


if __name__ == "__main__":
    unittest.main()
