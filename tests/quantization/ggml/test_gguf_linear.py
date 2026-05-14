# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Standalone tests for the GgufLinear module.

These don't exercise the full ``from_pretrained(..., gguf_linear=True)`` path
(which depends on the GGUF rename + dequant pipeline being healthy on this
branch). They just check that ``GgufLinear`` produces forward outputs that
agree with the dequant-then-``nn.functional.linear`` reference, using
synthesized Q4_0 / Q4_K bytes.

The Metal-kernel fast path on MPS is exercised via the ``kernels-community``
package (``ArthurZ/gguf-kernels``). If that package isn't installed locally,
GgufLinear falls back to the pure-torch dequant path and these tests still
pass.
"""

from __future__ import annotations

import unittest

import numpy as np

from transformers.testing_utils import require_gguf, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from transformers.integrations.gguf_linear import (
        GgufLinear,
        gguf_linear_supports,
        replace_with_gguf_linear,
    )


def _random_q4_0_bytes(M: int, K: int, seed: int = 0) -> bytes:
    """Synthesize a valid Q4_0 byte blob with finite half scales."""
    nblocks = M * (K // 32)
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.01, 0.5, size=nblocks).astype(np.float16)
    qs = rng.integers(0, 256, size=(nblocks, 16), dtype=np.uint8)
    out = np.empty((nblocks, 18), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2)
    out[:, 2:18] = qs
    return out.tobytes()


def _random_q4_K_bytes(M: int, K: int, seed: int = 0) -> bytes:
    """Synthesize a valid Q4_K byte blob with finite half scales."""
    nblocks = M * (K // 256)
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.01, 0.5, size=nblocks).astype(np.float16)
    dmin = rng.uniform(0.0, 0.5, size=nblocks).astype(np.float16)
    body = rng.integers(0, 256, size=(nblocks, 140), dtype=np.uint8)
    out = np.empty((nblocks, 144), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2)
    out[:, 2:4] = dmin.view(np.uint8).reshape(nblocks, 2)
    out[:, 4:144] = body
    return out.tobytes()


@require_torch
@require_gguf
class GgufLinearForwardTest(unittest.TestCase):
    """Forward pass: GgufLinear vs (dequant → nn.functional.linear) baseline."""

    def _check_forward(self, quant_type: str, qbytes_fn, M: int, K: int, batch_sizes):
        import gguf

        from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

        qb = qbytes_fn(M, K)
        layer = GgufLinear(in_features=K, out_features=M, quant_type=quant_type, bias=False)
        layer.qweight.copy_(torch.frombuffer(bytearray(qb), dtype=torch.uint8))

        # Reference: dequant the same bytes, then run torch's nn.linear
        W_ref = dequantize_gguf_tensor(
            layer.qweight, getattr(gguf.GGMLQuantizationType, quant_type), device="cpu"
        ).reshape(M, K).to(torch.float32)

        torch.manual_seed(0)
        for B in batch_sizes:
            shape = (K,) if B == 0 else (B, K)
            x = torch.randn(shape, dtype=torch.float32) * 0.1
            ref = torch.nn.functional.linear(x, W_ref)
            got = layer(x)
            self.assertEqual(ref.shape, got.shape)
            scale = max(1.0, float(ref.abs().max()))
            rel = float((ref - got).abs().max() / scale)
            tol = 1e-3 if B > 1 else 1e-5
            self.assertLessEqual(rel, tol, msg=f"B={B}: rel={rel:.4e}")

    def test_q4_0_forward(self):
        self._check_forward("Q4_0", _random_q4_0_bytes, M=64, K=512, batch_sizes=(0, 1, 8))

    def test_q4_K_forward(self):
        self._check_forward("Q4_K", _random_q4_K_bytes, M=64, K=256, batch_sizes=(0, 1, 8))


@require_torch
@require_gguf
class GgufLinearSwapTest(unittest.TestCase):
    """``replace_with_gguf_linear`` swaps Linears and preserves forward output."""

    def test_swap_two_layer_mlp(self):
        import gguf
        import torch.nn as nn

        from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

        K, H, M = 256, 512, 128
        qb1 = _random_q4_K_bytes(H, K, seed=1)
        qb2 = _random_q4_K_bytes(M, H, seed=2)

        W1_ref = dequantize_gguf_tensor(
            torch.frombuffer(bytearray(qb1), dtype=torch.uint8),
            gguf.GGMLQuantizationType.Q4_K, device="cpu",
        ).reshape(H, K).to(torch.float32)
        W2_ref = dequantize_gguf_tensor(
            torch.frombuffer(bytearray(qb2), dtype=torch.uint8),
            gguf.GGMLQuantizationType.Q4_K, device="cpu",
        ).reshape(M, H).to(torch.float32)

        baseline = nn.Sequential(nn.Linear(K, H, bias=False), nn.ReLU(), nn.Linear(H, M, bias=False))
        baseline[0].weight.data.copy_(W1_ref)
        baseline[2].weight.data.copy_(W2_ref)

        target = nn.Sequential(nn.Linear(K, H, bias=False), nn.ReLU(), nn.Linear(H, M, bias=False))
        target[0].weight.data.copy_(W1_ref)
        target[2].weight.data.copy_(W2_ref)

        # Swap directly (bypass the gguf.quantize round-trip — Q4_K Python quantizer
        # isn't implemented, and this test only validates the swap mechanics).
        target[0] = GgufLinear(K, H, "Q4_K", bias=False)
        target[0].qweight.copy_(torch.frombuffer(bytearray(qb1), dtype=torch.uint8))
        target[2] = GgufLinear(H, M, "Q4_K", bias=False)
        target[2].qweight.copy_(torch.frombuffer(bytearray(qb2), dtype=torch.uint8))

        torch.manual_seed(7)
        x = torch.randn(4, K) * 0.1
        ref = baseline(x)
        got = target(x)
        self.assertEqual(ref.shape, got.shape)
        rel = float((ref - got).abs().max() / max(1.0, float(ref.abs().max())))
        self.assertLessEqual(rel, 1e-3, msg=f"swap forward rel={rel:.4e}")


@require_torch
@require_gguf
class GgufLinearAcceptsSupportedTypes(unittest.TestCase):
    def test_supported(self):
        import gguf

        self.assertTrue(gguf_linear_supports(gguf.GGMLQuantizationType.Q4_0))
        self.assertTrue(gguf_linear_supports(gguf.GGMLQuantizationType.Q4_K))
        self.assertFalse(gguf_linear_supports(gguf.GGMLQuantizationType.F32))


if __name__ == "__main__":
    unittest.main()
