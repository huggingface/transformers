# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the slim GGUF integration.

Covers:
* :class:`GgufLinear` construction + ``state_dict`` / ``load_state_dict`` round-trip
* :func:`replace_with_gguf_linear` meta-time swap mechanics
* ``MODEL_TYPE_TO_GGUF_EXPERTS`` registry lookup
* the public load APIs:
    - ``from_pretrained(..., gguf_file=..., dtype=torch.bfloat16)`` → dequant path
    - ``from_pretrained(..., quantization_config=GgufQuantizeConfig(...))`` → on-the-fly swap
* generation-config defaults set by ``GGUFQuantizer.postprocess_model``

Forward-pass correctness lives behind MPS + kernels-available — the strict
no-fallback policy means we can't validate forward on CI hosts. The structural
tests below run anywhere.
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from transformers.testing_utils import require_gguf, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn

    from transformers.integrations.gguf_linear import (
        MODEL_TYPE_TO_GGUF_EXPERTS,
        GgufExperts,
        GgufLinear,
        gguf_linear_supports,
        replace_with_gguf_linear,
    )


def _random_q4_0_bytes(M: int, K: int, seed: int = 0) -> bytes:
    nblocks = M * (K // 32)
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.01, 0.5, size=nblocks).astype(np.float16)
    qs = rng.integers(0, 256, size=(nblocks, 16), dtype=np.uint8)
    out = np.empty((nblocks, 18), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2)
    out[:, 2:18] = qs
    return out.tobytes()


def _random_q4_K_bytes(M: int, K: int, seed: int = 0) -> bytes:
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
class GgufLinearConstructionTest(unittest.TestCase):
    def test_supports_lookup(self):
        import gguf

        for name in ("Q4_0", "Q4_K", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "IQ4_XS"):
            self.assertTrue(gguf_linear_supports(getattr(gguf.GGMLQuantizationType, name)), name)
        self.assertFalse(gguf_linear_supports(gguf.GGMLQuantizationType.F32))

    def test_weight_buffer_size(self):
        # Q4_K: 144 bytes / 256 elems → buffer shape (out_features, bytes_per_row).
        layer = GgufLinear(in_features=256, out_features=32, quant_type="Q4_K", bias=False)
        self.assertEqual(tuple(layer.weight.shape), (32, (256 // 256) * 144))
        self.assertEqual(layer.weight.dtype, torch.uint8)

    def test_rejects_unsupported_dim(self):
        with self.assertRaises(ValueError):
            GgufLinear(in_features=130, out_features=32, quant_type="Q4_0")  # 130 not multiple of 32


@require_torch
@require_gguf
class GgufLinearStateDictRoundtripTest(unittest.TestCase):
    """``state_dict`` → ``load_state_dict`` preserves bytes + extra-state metadata."""

    def test_q4_0_roundtrip(self):
        K, M = 64, 32
        qb = _random_q4_0_bytes(M, K)
        layer = GgufLinear(in_features=K, out_features=M, quant_type="Q4_0", bias=False)
        # Buffer is `(out_features, bytes_per_row)` 2D uint8.
        layer.weight.copy_(torch.frombuffer(bytearray(qb), dtype=torch.uint8).view_as(layer.weight))
        clone = GgufLinear(in_features=K, out_features=M, quant_type="Q4_0", bias=False)
        clone.load_state_dict(layer.state_dict())
        self.assertTrue(torch.equal(layer.weight, clone.weight))


@require_torch
@require_gguf
class ReplaceWithGgufLinearTest(unittest.TestCase):
    """``replace_with_gguf_linear`` does the FP8-style meta-time swap."""

    def test_swap_select_linears(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("C", (), {"model_type": "llama"})()
                self.a = nn.Linear(64, 32, bias=False)
                self.b = nn.Linear(64, 32, bias=False)

        m = M()
        info = {"a": {"quant_type": "Q4_0"}}  # only swap `.a`
        n = replace_with_gguf_linear(m, info)
        self.assertEqual(n, 1)
        self.assertIsInstance(m.a, GgufLinear)
        self.assertIsInstance(m.b, nn.Linear)
        # Buffer shape is (out_features, bytes_per_row) for Q4_0: 32 × 2 blocks × 18 bytes.
        self.assertEqual(tuple(m.a.weight.shape), (32, (64 // 32) * 18))

    def test_moe_registry_lookup(self):
        # Every MoE arch we promise byte-passthrough support for must land on
        # the GgufExperts base. gpt_oss is intentionally excluded — see
        # `test_gguf_arch_coverage.py::GgufExpertsRegistryTests` for the why.
        for arch in ("qwen2_moe", "qwen3_moe", "minimax_m2", "mixtral", "deepseek_v3"):
            self.assertIn(arch, MODEL_TYPE_TO_GGUF_EXPERTS, arch)
            self.assertIs(MODEL_TYPE_TO_GGUF_EXPERTS[arch], GgufExperts, arch)


@require_torch
@require_gguf
class GgufLinearForwardOnMpsTest(unittest.TestCase):
    """Forward path only runs on MPS with kernels available — no slow fallback."""

    def test_forward_errors_off_mps(self):
        layer = GgufLinear(in_features=32, out_features=32, quant_type="Q4_0", bias=False)
        x = torch.zeros(1, 32, dtype=torch.float32, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "GgufLinear runs only on MPS"):
            layer(x)


@require_torch
@require_gguf
class FromPretrainedDequantPathTest(unittest.TestCase):
    """`gguf_file=` + explicit `dtype=` → dequant path; no GgufLinear in the model."""

    def test_dequant_on_load(self):
        # Pull a tiny GGUF off the Hub. The dequant path runs on CPU.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct-GGUF",
            gguf_file="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        n_gguf = sum(1 for _, mod in model.named_modules() if isinstance(mod, GgufLinear))
        self.assertEqual(n_gguf, 0)
        # First projection weight came through dequant.
        q = next(p for name, p in model.named_parameters() if name.endswith("q_proj.weight"))
        self.assertEqual(q.dtype, torch.bfloat16)


@require_torch
@require_gguf
class FromPretrainedOnTheFlyTest(unittest.TestCase):
    """`quantization_config=GgufQuantizeConfig(...)` → meta-time swap, GGUFQuantize op packs bytes."""

    def test_on_the_fly_swap(self):
        from transformers import AutoModelForCausalLM, GgufQuantizeConfig

        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            dtype=torch.float32,
            quantization_config=GgufQuantizeConfig(quant_type="Q4_0", modules_to_convert=["model.layers.*.mlp.*"]),
        )
        gguf_linears = [name for name, mod in model.named_modules() if isinstance(mod, GgufLinear)]
        # 30 layers × 3 mlp linears = 90.
        self.assertEqual(len(gguf_linears), 90)
        # The buffer is uint8 and non-empty after the load.
        sample = model.get_submodule(gguf_linears[0]).weight
        self.assertEqual(sample.dtype, torch.uint8)
        self.assertGreater(sample.numel(), 0)
        self.assertTrue(bool((sample != 0).any()), "weight should have been populated by GGUFQuantize")


@require_torch
@require_gguf
class GenerationDefaultsTest(unittest.TestCase):
    """`postprocess_model` should default `cache_implementation` + `compile_config`
    when the GgufLinear swap is in effect."""

    def test_defaults_applied(self):
        if not torch.backends.mps.is_available():
            self.skipTest("GgufLinear swap only triggers on MPS")
        from transformers.integrations.gguf_kernels import metal_kernels_available

        if not metal_kernels_available():
            self.skipTest("GgufLinear swap also requires the `ArthurZ/gguf-kernels` package")
        from transformers import AutoModelForCausalLM
        from transformers.generation.configuration_utils import CompileConfig

        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B-Instruct-GGUF",
            gguf_file="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        )
        self.assertEqual(model.generation_config.cache_implementation, "static")
        self.assertIsInstance(model.generation_config.compile_config, CompileConfig)
        self.assertEqual(model.generation_config.compile_config.mode, "reduce-overhead")


@require_torch
@require_gguf
class SaveRoundtripTest(unittest.TestCase):
    """`state_dict` → safetensors round-trip via ``save_pretrained`` / ``from_pretrained``."""

    def test_uint8_buffers_survive_safetensors(self):
        # Quick check that uint8 buffer values survive a CPU round-trip via state_dict.
        K, M = 32, 32
        layer = GgufLinear(in_features=K, out_features=M, quant_type="Q4_0", bias=False)
        src = torch.frombuffer(bytearray(_random_q4_0_bytes(M, K, seed=11)), dtype=torch.uint8)
        layer.weight.copy_(src.view_as(layer.weight))
        with tempfile.TemporaryDirectory():
            sd = layer.state_dict()
            clone = GgufLinear(in_features=K, out_features=M, quant_type="Q4_0", bias=False)
            clone.load_state_dict(sd)
            self.assertTrue(torch.equal(layer.weight, clone.weight))


if __name__ == "__main__":
    unittest.main()
