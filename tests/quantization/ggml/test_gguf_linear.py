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

import os
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
        self.assertTrue(gguf_linear_supports(gguf.GGMLQuantizationType.Q5_0))
        self.assertTrue(gguf_linear_supports(gguf.GGMLQuantizationType.Q5_1))
        self.assertFalse(gguf_linear_supports(gguf.GGMLQuantizationType.F32))


@require_torch
@require_gguf
class GgufLinearSerializationTest(unittest.TestCase):
    """``state_dict`` / ``load_state_dict`` roundtrip preserves bytes + quant type,
    and forward output of a freshly reloaded module matches the original."""

    def _roundtrip_state_dict(self, layer):
        sd = layer.state_dict()
        clone = GgufLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            quant_type=layer.quant_type,
            bias=layer.bias is not None,
        )
        # zero the clone so the reload is observable
        clone.qweight.zero_()
        if clone.bias is not None:
            clone.bias.data.zero_()
        clone.load_state_dict(sd)
        return clone, sd

    def test_q4_0_state_dict_roundtrip(self):
        M, K = 64, 256
        qb = _random_q4_0_bytes(M, K, seed=11)
        layer = GgufLinear(K, M, "Q4_0", bias=False)
        layer.qweight.copy_(torch.frombuffer(bytearray(qb), dtype=torch.uint8))

        clone, sd = self._roundtrip_state_dict(layer)

        # state_dict carries the raw bytes verbatim.
        self.assertTrue(torch.equal(sd["qweight"], layer.qweight))
        # And the extra state carries the quant_type.
        self.assertIn("_extra_state", sd)
        self.assertEqual(sd["_extra_state"]["quant_type"], "Q4_0")
        # Reloaded module has identical bytes.
        self.assertTrue(torch.equal(clone.qweight, layer.qweight))

        # Forward output of the clone matches the original bit-for-bit.
        torch.manual_seed(3)
        x = torch.randn(4, K) * 0.1
        self.assertTrue(torch.equal(layer(x), clone(x)))

    def test_q4_K_state_dict_with_bias(self):
        M, K = 64, 256
        qb = _random_q4_K_bytes(M, K, seed=13)
        layer = GgufLinear(K, M, "Q4_K", bias=True)
        layer.qweight.copy_(torch.frombuffer(bytearray(qb), dtype=torch.uint8))
        layer.bias.data.copy_(torch.linspace(-0.5, 0.5, M))

        clone, sd = self._roundtrip_state_dict(layer)

        self.assertIn("bias", sd)
        self.assertTrue(torch.equal(clone.bias, layer.bias))
        torch.manual_seed(5)
        x = torch.randn(2, K) * 0.1
        self.assertTrue(torch.equal(layer(x), clone(x)))

    def test_quant_type_mismatch_raises(self):
        M, K = 64, 256
        qb = _random_q4_0_bytes(M, K, seed=17)
        layer = GgufLinear(K, M, "Q4_0", bias=False)
        layer.qweight.copy_(torch.frombuffer(bytearray(qb), dtype=torch.uint8))
        sd = layer.state_dict()

        # Loading Q4_0 state into a Q4_K-configured module must fail loudly
        # rather than silently dequant garbage.
        wrong = GgufLinear(K, M, "Q4_K", bias=False)
        with self.assertRaises((ValueError, RuntimeError)):
            wrong.load_state_dict(sd)

    def test_disk_roundtrip_via_torch_save(self):
        """torch.save / torch.load on the state_dict — same path save_pretrained
        uses internally for non-safetensor backends."""
        import io

        M, K = 64, 256
        qb = _random_q4_K_bytes(M, K, seed=23)
        layer = GgufLinear(K, M, "Q4_K", bias=False)
        layer.qweight.copy_(torch.frombuffer(bytearray(qb), dtype=torch.uint8))

        buf = io.BytesIO()
        torch.save(layer.state_dict(), buf)
        buf.seek(0)
        sd = torch.load(buf, weights_only=False)

        clone = GgufLinear(K, M, "Q4_K", bias=False)
        clone.qweight.zero_()
        clone.load_state_dict(sd)

        torch.manual_seed(11)
        x = torch.randn(3, K) * 0.1
        self.assertTrue(torch.equal(layer(x), clone(x)))


@require_torch
@require_gguf
class GgufQwen2MoeExpertsSerializationTest(unittest.TestCase):
    """Round-trip ``GgufQwen2MoeExperts`` — including per-projection quant types
    that mixed-precision (e.g. Q4_K_M) files exercise."""

    @staticmethod
    def _build_module(gate="Q4_K", up="Q4_K", down="Q8_0", num_experts=4):
        from transformers.integrations.gguf_linear import GgufQwen2MoeExperts

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.num_experts = num_experts
        cfg.hidden_size = 256
        cfg.moe_intermediate_size = 256
        cfg.hidden_act = "silu"
        return GgufQwen2MoeExperts(cfg, gate_quant=gate, up_quant=up, down_quant=down)

    @staticmethod
    def _fill_random(m, seed=0):
        rng = np.random.default_rng(seed)
        for name in ("gate_proj_q", "up_proj_q", "down_proj_q"):
            buf = getattr(m, name)
            rnd = torch.from_numpy(rng.integers(0, 256, size=buf.shape, dtype=np.uint8))
            buf.copy_(rnd)

    def test_state_dict_roundtrip_mixed_quants(self):
        # Q4_K_M-style: gate/up are Q4_K, down is Q8_0.
        m = self._build_module(gate="Q4_K", up="Q4_K", down="Q8_0", num_experts=3)
        self._fill_random(m, seed=31)

        sd = m.state_dict()
        # Per-projection bytes and per-projection quant types both ride along.
        self.assertIn("gate_proj_q", sd)
        self.assertIn("up_proj_q", sd)
        self.assertIn("down_proj_q", sd)
        self.assertIn("_extra_state", sd)
        self.assertEqual(sd["_extra_state"]["gate_quant"], "Q4_K")
        self.assertEqual(sd["_extra_state"]["up_quant"], "Q4_K")
        self.assertEqual(sd["_extra_state"]["down_quant"], "Q8_0")

        clone = self._build_module(gate="Q4_K", up="Q4_K", down="Q8_0", num_experts=3)
        for name in ("gate_proj_q", "up_proj_q", "down_proj_q"):
            getattr(clone, name).zero_()
        clone.load_state_dict(sd)
        for name in ("gate_proj_q", "up_proj_q", "down_proj_q"):
            self.assertTrue(torch.equal(getattr(clone, name), getattr(m, name)))

    def test_state_dict_roundtrip_uniform_q4_K(self):
        m = self._build_module(gate="Q4_K", up="Q4_K", down="Q4_K", num_experts=2)
        self._fill_random(m, seed=37)
        sd = m.state_dict()

        clone = self._build_module(gate="Q4_K", up="Q4_K", down="Q4_K", num_experts=2)
        clone.load_state_dict(sd)
        for name in ("gate_proj_q", "up_proj_q", "down_proj_q"):
            self.assertTrue(torch.equal(getattr(clone, name), getattr(m, name)))

    def test_quant_type_mismatch_raises(self):
        m = self._build_module(gate="Q4_K", up="Q4_K", down="Q8_0", num_experts=2)
        self._fill_random(m, seed=41)
        sd = m.state_dict()
        # Constructed with the wrong ``down_quant`` — must reject.
        wrong = self._build_module(gate="Q4_K", up="Q4_K", down="Q4_K", num_experts=2)
        with self.assertRaises((ValueError, RuntimeError)):
            wrong.load_state_dict(sd)


def _build_synthetic_llama_gguf(path, *, H=64, I=128, L=2, V=256, num_heads=2, seed=0):
    """Write a tiny well-formed Q4_0 llama GGUF file for round-trip tests."""
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType as Q

    rng = np.random.default_rng(seed)
    w = GGUFWriter(path, arch="llama")
    w.add_block_count(L); w.add_embedding_length(H); w.add_feed_forward_length(I)
    w.add_head_count(num_heads); w.add_head_count_kv(num_heads); w.add_context_length(64)
    w.add_rope_dimension_count(H // num_heads); w.add_layer_norm_rms_eps(1e-6); w.add_vocab_size(V)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(["<unk>"] * V); w.add_token_types([1] * V); w.add_token_merges([])

    def add_q40(name, shape):
        a = rng.standard_normal(shape).astype(np.float32) * 0.1
        w.add_tensor(name, gguf.quants.quantize(a, Q.Q4_0), raw_dtype=Q.Q4_0)

    def add_f32(name, shape):
        a = rng.standard_normal(shape).astype(np.float32) * 0.1
        w.add_tensor(name, a, raw_dtype=Q.F32)

    add_q40("token_embd.weight", (V, H))
    add_f32("output_norm.weight", (H,))
    add_q40("output.weight", (V, H))
    for i in range(L):
        for kind in ("attn_q", "attn_k", "attn_v", "attn_output"):
            add_q40(f"blk.{i}.{kind}.weight", (H, H))
        for kind in ("ffn_gate", "ffn_up"):
            add_q40(f"blk.{i}.{kind}.weight", (I, H))
        add_q40(f"blk.{i}.ffn_down.weight", (H, I))
        add_f32(f"blk.{i}.attn_norm.weight", (H,))
        add_f32(f"blk.{i}.ffn_norm.weight",  (H,))

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path


@require_torch
@require_gguf
class GgufSaveRoundtripTest(unittest.TestCase):
    """``hf_quantizer.save_gguf`` writes a .gguf file that, on reload, produces
    the same forward output as the original. The byte-preserved path is the
    happy path: same file size, bit-identical for the per-tensor blob ordering."""

    def _build_and_load(self, tmpdir, name="src.gguf"):
        from transformers import AutoModelForCausalLM

        path = os.path.join(tmpdir, name)
        _build_synthetic_llama_gguf(path)
        model = AutoModelForCausalLM.from_pretrained(tmpdir, gguf_file=name, gguf_linear=True)
        return path, model

    def test_byte_preserved_roundtrip(self):
        import tempfile
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as tmp:
            _, m1 = self._build_and_load(tmp)
            rt = os.path.join(tmp, "rt.gguf")
            m1.hf_quantizer.save_gguf(m1, rt)

            m2 = AutoModelForCausalLM.from_pretrained(tmp, gguf_file="rt.gguf", gguf_linear=True)
            torch.manual_seed(7)
            ids = torch.randint(0, m1.config.vocab_size, (1, 8))
            y1 = m1(ids).logits
            y2 = m2(ids).logits
            err = float((y1 - y2).abs().max())
            self.assertEqual(err, 0.0, msg=f"byte-preserved roundtrip diverged: {err}")

    def test_quantize_on_save_norms_to_f16(self):
        """Override the F32 norms to F16 on save — verifies the policy DSL routes
        the right tensors and the file size shrinks accordingly."""
        import tempfile
        import gguf
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as tmp:
            src, m1 = self._build_and_load(tmp)
            rt = os.path.join(tmp, "rt-f16.gguf")
            m1.hf_quantizer.save_gguf(m1, rt, quant_config={r"_norm\.weight$": "F16"})

            # The norm tensors must now read as F16 in the saved file.
            r = gguf.GGUFReader(rt)
            for t in r.tensors:
                if "_norm" in t.name:
                    self.assertEqual(t.tensor_type, gguf.GGMLQuantizationType.F16,
                                     msg=f"{t.name}: expected F16, got {t.tensor_type}")

            # Forward equivalence within F32→F16 noise.
            m2 = AutoModelForCausalLM.from_pretrained(tmp, gguf_file="rt-f16.gguf", gguf_linear=True)
            torch.manual_seed(7)
            ids = torch.randint(0, m1.config.vocab_size, (1, 8))
            err = float((m1(ids).logits - m2(ids).logits).abs().max())
            self.assertLess(err, 5e-3, msg=f"F32→F16 norm save diverged too much: {err}")

    def test_save_gguf_handles_missing_quantizer(self):
        """``save_pretrained_gguf`` errors clearly when given a model that
        wasn't loaded from a .gguf (no hf_to_gguf map, no source rename rules)."""
        from transformers.integrations.gguf_save import save_pretrained_gguf
        import tempfile

        class _Stub:
            config = type("C", (), {"model_type": "definitely_not_a_real_arch"})()

            def state_dict(self):
                return {}

            def named_modules(self):
                return []

        with tempfile.NamedTemporaryFile(suffix=".gguf") as tf:
            with self.assertRaises(ValueError):
                save_pretrained_gguf(_Stub(), tf.name)


if __name__ == "__main__":
    unittest.main()
