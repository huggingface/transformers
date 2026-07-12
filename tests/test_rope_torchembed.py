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
"""Tests for the torchembed-accelerated apply_rotary_pos_emb in modeling_rope_utils."""

import unittest

import torch

from transformers.modeling_rope_utils import (
    _TORCHEMBED_AVAILABLE,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.testing_utils import require_torch_gpu


def _make_cos_sin(B, S, D, dtype, device):
    """Build (B, S, D) cos/sin tensors matching LlamaRotaryEmbedding's output format.

    Frequencies are doubled (cat([freqs, freqs], dim=-1)) as in the Llama implementation.
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float, device=device) / D))
    t = torch.arange(S, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (S, D//2)
    freqs_doubled = torch.cat([freqs, freqs], dim=-1)  # (S, D)
    cos = freqs_doubled.cos().to(dtype).unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, S, D)
    sin = freqs_doubled.sin().to(dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    return cos, sin


def _ref_apply_rotary(q, k, cos, sin, unsqueeze_dim=1):
    """Pure-PyTorch reference (the original rotate_half path)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TestRotateHalf(unittest.TestCase):
    def test_shape_preserved(self):
        x = torch.randn(2, 8, 16, 64)
        out = rotate_half(x)
        self.assertEqual(out.shape, x.shape)

    def test_double_application_is_identity(self):
        x = torch.randn(2, 8, 16, 64)
        self.assertTrue(torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6))

    def test_values(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(rotate_half(x), expected))


class TestApplyRotaryPosEmb(unittest.TestCase):
    """Tests for apply_rotary_pos_emb — covers both the reference and (if available) fused path."""

    B, H, KVH, S, D = 2, 32, 8, 512, 128

    def _tensors(self, dtype=torch.float32, device="cpu"):
        q = torch.randn(self.B, self.H, self.S, self.D, dtype=dtype, device=device)
        k = torch.randn(self.B, self.KVH, self.S, self.D, dtype=dtype, device=device)
        cos, sin = _make_cos_sin(self.B, self.S, self.D, dtype, device)
        return q, k, cos, sin

    def test_output_shapes(self):
        q, k, cos, sin = self._tensors()
        qo, ko = apply_rotary_pos_emb(q, k, cos, sin)
        self.assertEqual(qo.shape, q.shape)
        self.assertEqual(ko.shape, k.shape)

    def test_reference_path_float32(self):
        q, k, cos, sin = self._tensors(dtype=torch.float32)
        qo, ko = apply_rotary_pos_emb(q, k, cos, sin)
        qr, kr = _ref_apply_rotary(q, k, cos, sin)
        self.assertTrue(torch.allclose(qo, qr, atol=1e-5))
        self.assertTrue(torch.allclose(ko, kr, atol=1e-5))

    @require_torch_gpu
    @unittest.skipUnless(_TORCHEMBED_AVAILABLE, "torchembed + triton not installed")
    def test_fused_matches_reference_float16(self):
        device = "cuda"
        q, k, cos, sin = self._tensors(dtype=torch.float16, device=device)
        qo, ko = apply_rotary_pos_emb(q, k, cos, sin)
        qr, kr = _ref_apply_rotary(q, k, cos, sin)
        # fp16 arithmetic: allow small absolute tolerance
        self.assertTrue(torch.allclose(qo, qr, atol=0.01), f"max q diff: {(qo-qr).abs().max()}")
        self.assertTrue(torch.allclose(ko, kr, atol=0.01), f"max k diff: {(ko-kr).abs().max()}")

    @require_torch_gpu
    @unittest.skipUnless(_TORCHEMBED_AVAILABLE, "torchembed + triton not installed")
    def test_fused_matches_reference_bfloat16(self):
        device = "cuda"
        q, k, cos, sin = self._tensors(dtype=torch.bfloat16, device=device)
        qo, ko = apply_rotary_pos_emb(q, k, cos, sin)
        qr, kr = _ref_apply_rotary(q, k, cos, sin)
        self.assertTrue(torch.allclose(qo, qr, atol=0.04), f"max q diff: {(qo-qr).abs().max()}")

    @require_torch_gpu
    @unittest.skipUnless(_TORCHEMBED_AVAILABLE, "torchembed + triton not installed")
    def test_gradient_flows(self):
        device = "cuda"
        q, k, cos, sin = self._tensors(dtype=torch.float16, device=device)
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        qo, ko = apply_rotary_pos_emb(q, k, cos, sin)
        (qo.sum() + ko.sum()).backward()
        self.assertIsNotNone(q.grad)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertIsNotNone(k.grad)
        self.assertTrue(torch.isfinite(k.grad).all())

    @require_torch_gpu
    @unittest.skipUnless(_TORCHEMBED_AVAILABLE, "torchembed + triton not installed")
    def test_gradient_matches_reference(self):
        """Fused backward gradient is close to the float32 reference backward."""
        device = "cuda"
        q32, k32, cos32, sin32 = self._tensors(dtype=torch.float32, device=device)
        q32, k32 = q32.requires_grad_(True), k32.requires_grad_(True)
        (_ref_apply_rotary(q32, k32, cos32, sin32)[0].sum() +
         _ref_apply_rotary(q32, k32, cos32, sin32)[1].sum()).backward()

        q16 = q32.detach().half().requires_grad_(True)
        k16 = k32.detach().half().requires_grad_(True)
        cos16, sin16 = cos32.half(), sin32.half()
        qo, ko = apply_rotary_pos_emb(q16, k16, cos16, sin16)
        (qo.sum() + ko.sum()).backward()

        torch.testing.assert_close(q16.grad.float(), q32.grad, atol=0.02, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
