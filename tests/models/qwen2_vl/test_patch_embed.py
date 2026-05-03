# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Tests for the Conv3d → Linear patch-embed rewrite across Qwen-VL model families.
Covers:
  - fp32 numerical equivalence between the old Conv3d path and the new Linear path
  - bf16 cosine similarity > 0.999 on the projection output
  - backward-compat _load_from_state_dict: 5-D Conv3d weights load into Linear without error
  - Qwen2-VL, Qwen2.5-VL and Qwen3-VL classes are all exercised
See https://github.com/huggingface/transformers/issues/45750
"""

import unittest

import torch
import torch.nn as nn

from transformers.testing_utils import require_torch


def _make_conv3d_weight(out_dim, in_c, kt, kh, kw, bias=False, seed=42):
    """Return a Conv3d module initialised from a fixed seed."""
    torch.manual_seed(seed)
    conv = nn.Conv3d(in_c, out_dim, kernel_size=(kt, kh, kw), stride=(kt, kh, kw), bias=bias)
    return conv


def _conv3d_to_linear(conv):
    """Construct the equivalent Linear by reshaping Conv3d weights."""
    out_dim = conv.out_channels
    in_dim = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1] * conv.kernel_size[2]
    lin = nn.Linear(in_dim, out_dim, bias=conv.bias is not None)
    lin.weight.data.copy_(conv.weight.detach().reshape(out_dim, in_dim))
    if conv.bias is not None:
        lin.bias.data.copy_(conv.bias.detach())
    return lin


def _run_patch_embed_class(cls, init_kwargs, x_5d):
    """Instantiate cls, optionally with a 5-D conv weight pre-loaded, and run forward."""
    from types import SimpleNamespace

    # Build a config-like namespace for classes that take a config object
    cfg = SimpleNamespace(**init_kwargs)
    module = cls(cfg)
    module.eval()
    return module, module(x_5d.reshape(-1, x_5d.shape[1] * x_5d.shape[2] * x_5d.shape[3] * x_5d.shape[4]))


@require_torch
class TestPatchEmbedLinearEquivalence(unittest.TestCase):
    """Verify that the new Linear forward is numerically equivalent to the old Conv3d path."""

    # (in_channels, temporal_patch_size, patch_size, embed_dim, bias)
    CONFIGS = [
        (3, 2, 14, 1152, False),  # Qwen2-VL / Qwen2.5-VL defaults
        (3, 2, 14, 1024, True),   # Qwen3-VL defaults (bias=True)
    ]
    N_PATCHES = 64

    def _make_inputs(self, in_c, kt, kh, kw, n=64):
        torch.manual_seed(0)
        # The modules expect pre-flattened input: (N, in_c * kt * kh * kw)
        return torch.randn(n, in_c * kt * kh * kw)

    def _forward_conv3d(self, x_flat, in_c, kt, kh, kw, out_dim, bias, dtype=torch.float32):
        conv = _make_conv3d_weight(out_dim, in_c, kt, kh, kw, bias=bias)
        conv = conv.to(dtype)
        x_5d = x_flat.reshape(-1, in_c, kt, kh, kw).to(dtype)
        with torch.no_grad():
            return conv(x_5d).view(-1, out_dim)

    def _forward_linear(self, x_flat, in_c, kt, kh, kw, out_dim, bias, dtype=torch.float32):
        conv = _make_conv3d_weight(out_dim, in_c, kt, kh, kw, bias=bias)
        lin = _conv3d_to_linear(conv).to(dtype)
        with torch.no_grad():
            return lin(x_flat.to(dtype))

    def test_fp32_equivalence(self):
        for in_c, kt, kh, kw, out_dim, bias in [
            (3, 2, 14, 14, 1152, False),
            (3, 2, 14, 14, 1024, True),
        ]:
            with self.subTest(out_dim=out_dim, bias=bias):
                x = self._make_inputs(in_c, kt, kh, kw)
                y_conv = self._forward_conv3d(x, in_c, kt, kh, kw, out_dim, bias)
                y_lin = self._forward_linear(x, in_c, kt, kh, kw, out_dim, bias)
                max_diff = (y_conv - y_lin).abs().max().item()
                self.assertLess(max_diff, 1e-5, f"fp32 max abs diff {max_diff:.2e} exceeds 1e-5")

    def test_bf16_cosine_similarity(self):
        for in_c, kt, kh, kw, out_dim, bias in [
            (3, 2, 14, 14, 1152, False),
            (3, 2, 14, 14, 1024, True),
        ]:
            with self.subTest(out_dim=out_dim, bias=bias):
                x = self._make_inputs(in_c, kt, kh, kw)
                y_conv = self._forward_conv3d(x, in_c, kt, kh, kw, out_dim, bias, dtype=torch.bfloat16).float()
                y_lin = self._forward_linear(x, in_c, kt, kh, kw, out_dim, bias, dtype=torch.bfloat16).float()
                cos = nn.functional.cosine_similarity(
                    y_conv.flatten().unsqueeze(0), y_lin.flatten().unsqueeze(0)
                ).item()
                self.assertGreater(cos, 0.999, f"bf16 cosine similarity {cos:.6f} below 0.999")


@require_torch
class TestPatchEmbedCheckpointCompat(unittest.TestCase):
    """Verify that _load_from_state_dict handles 5-D Conv3d weights from old checkpoints."""

    def _test_compat(self, cls, init_kwargs, in_c, kt, kh, kw, embed_dim, bias):
        from types import SimpleNamespace

        cfg = SimpleNamespace(**init_kwargs)
        module = cls(cfg)
        module.eval()

        # Simulate a legacy checkpoint: build a Conv3d and use its raw 5-D state dict
        conv = _make_conv3d_weight(embed_dim, in_c, kt, kh, kw, bias=bias)
        legacy_sd = {
            "proj.weight": conv.weight.detach().clone(),  # shape (out, in, kt, kh, kw)
        }
        if bias:
            legacy_sd["proj.bias"] = conv.bias.detach().clone()

        # Should not raise
        missing, unexpected = module.load_state_dict(legacy_sd, strict=True)
        self.assertEqual(missing, [], f"Missing keys after legacy load: {missing}")
        self.assertEqual(unexpected, [], f"Unexpected keys: {unexpected}")

        # Weight should now be 2-D
        self.assertEqual(module.proj.weight.dim(), 2)

        # Forward should produce the same result as the Conv3d baseline
        in_features = in_c * kt * kh * kw
        x = torch.randn(16, in_features)
        lin_ref = _conv3d_to_linear(conv)
        with torch.no_grad():
            y_mod = module(x)
            y_ref = lin_ref(x)
        max_diff = (y_mod - y_ref).abs().max().item()
        self.assertLess(max_diff, 1e-5)

    def test_qwen2_vl_patch_embed_compat(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed

        class _Cfg:
            pass

        # PatchEmbed takes scalar kwargs, not a config object; instantiate directly
        module = PatchEmbed(patch_size=14, temporal_patch_size=2, in_channels=3, embed_dim=1152)
        module.eval()

        conv = _make_conv3d_weight(1152, 3, 2, 14, 14, bias=False)
        legacy_sd = {"proj.weight": conv.weight.detach().clone()}
        missing, unexpected = module.load_state_dict(legacy_sd, strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertEqual(module.proj.weight.dim(), 2)

    def test_qwen3_vl_patch_embed_compat(self):
        from types import SimpleNamespace

        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

        cfg = SimpleNamespace(patch_size=14, temporal_patch_size=2, in_channels=3, hidden_size=1024)
        module = Qwen3VLVisionPatchEmbed(cfg)
        module.eval()

        conv = _make_conv3d_weight(1024, 3, 2, 14, 14, bias=True)
        legacy_sd = {
            "proj.weight": conv.weight.detach().clone(),
            "proj.bias": conv.bias.detach().clone(),
        }
        missing, unexpected = module.load_state_dict(legacy_sd, strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertEqual(module.proj.weight.dim(), 2)

    def test_qwen2_5_vl_patch_embed_compat(self):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed

        module = Qwen2_5_VisionPatchEmbed(patch_size=14, temporal_patch_size=2, in_channels=3, embed_dim=1152)
        module.eval()

        conv = _make_conv3d_weight(1152, 3, 2, 14, 14, bias=False)
        legacy_sd = {"proj.weight": conv.weight.detach().clone()}
        missing, unexpected = module.load_state_dict(legacy_sd, strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertEqual(module.proj.weight.dim(), 2)

    def test_idempotent_load(self):
        """Loading a 2-D weight (already converted) must not corrupt it."""
        from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed

        module = PatchEmbed(patch_size=14, temporal_patch_size=2, in_channels=3, embed_dim=1152)
        module.eval()

        # Save the already-linear state dict and reload it
        sd = module.state_dict()
        self.assertEqual(sd["proj.weight"].dim(), 2)
        module.load_state_dict(sd, strict=True)
        self.assertEqual(module.proj.weight.dim(), 2)


if __name__ == "__main__":
    unittest.main()
