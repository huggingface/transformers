# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fast (non-slow) tests pinning the GGUF integration's *coverage matrix* —
every model_type the loader claims to support, every quant_type the metal
kernels claim to handle, every MoE arch routed through the experts swap.

End-to-end per-arch generation tests live in `test_ggml.py` (slow CI).
The tests here only inspect the registries + the quantizer's contract with
the loading pipeline; they download nothing and run on every PR.
"""

from __future__ import annotations

import unittest

from parameterized import parameterized

from transformers.testing_utils import require_gguf, require_torch
from transformers.utils import is_torch_available


# HF model_types the public GGUF integration claims to load (mirrors the per-arch
# coverage in `test_ggml.py`). If you add a new entry to `_GGUF_ARCH_CONVERTERS`,
# add it here too so the coverage assertion catches accidental removal later.
EXPECTED_MODEL_TYPES = sorted(
    {
        # Llama / RoPE family
        "llama",
        "mistral",
        "phi3",
        "cohere",
        "qwen2",
        "qwen3",
        "deci",
        "stablelm",
        "starcoder2",
        # Norm-subtract-one variants
        "nemotron",
        "gemma2",
        "gemma3",
        "gemma3_text",
        # MoE families
        "qwen2_moe",
        "qwen3_moe",
        "minimax_m2",
        "gpt_oss",
        # Misc encoder/decoder & misc archs
        "bloom",
        "gpt2",
        "mamba",
        "lfm2",
        "falcon",
        "t5",
        "t5encoder",
        "umt5",
    }
)

# Quant types `GgufLinear` / `GgufExperts` accept on the metal fast path. The
# canonical source is `_QUANT_INFO` in `integrations.gguf_linear`; we re-list
# here so a drift between the kernel-side table and what tests expect is loud.
EXPECTED_KERNEL_QUANT_TYPES = (
    "Q4_0",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "IQ4_NL",
    "IQ4_XS",
)


@require_torch
@require_gguf
class GgufArchCoverageTests(unittest.TestCase):
    """Registry / well-formedness checks — no downloads."""

    def test_every_expected_model_type_is_registered(self):
        from transformers.modeling_gguf_pytorch_utils import _GGUF_ARCH_CONVERTERS

        missing = sorted(set(EXPECTED_MODEL_TYPES) - set(_GGUF_ARCH_CONVERTERS))
        self.assertFalse(
            missing,
            f"Model types previously supported by the GGUF loader are no longer registered "
            f"in `_GGUF_ARCH_CONVERTERS`: {missing}. Re-add the entry (or alias it onto an "
            f"existing converter list) so the loader keeps the same coverage.",
        )

    def test_no_unknown_model_types_silently_added(self):
        """Catches the reverse: a new entry sneaking into `_GGUF_ARCH_CONVERTERS`
        without being added to `EXPECTED_MODEL_TYPES`. Forces the test author of
        the new arch to acknowledge the coverage matrix."""
        from transformers.modeling_gguf_pytorch_utils import _GGUF_ARCH_CONVERTERS

        unexpected = sorted(set(_GGUF_ARCH_CONVERTERS) - set(EXPECTED_MODEL_TYPES))
        self.assertFalse(
            unexpected,
            f"New entries in `_GGUF_ARCH_CONVERTERS` that aren't in EXPECTED_MODEL_TYPES: "
            f"{unexpected}. Add them to EXPECTED_MODEL_TYPES (and to `test_ggml.py` if "
            f"they're not already covered there).",
        )

    @parameterized.expand([(m,) for m in EXPECTED_MODEL_TYPES])
    def test_arch_entry_is_well_formed(self, model_type: str):
        from transformers.core_model_loading import WeightConverter, WeightRenaming
        from transformers.modeling_gguf_pytorch_utils import get_gguf_converters

        rules = get_gguf_converters(model_type)
        self.assertGreater(len(rules), 0, f"{model_type}: empty converter list")
        for rule in rules:
            self.assertIsInstance(
                rule,
                (WeightRenaming, WeightConverter),
                f"{model_type}: every entry must be a WeightRenaming or WeightConverter, got {type(rule).__name__}",
            )


@require_torch
@require_gguf
class GgufQuantizerWiringTests(unittest.TestCase):
    """The quantizer is FP8-shaped: it owns the dequant decision in
    `load_checkpoint_state` and does NOT inject a `GGUFDequantize` op into the
    conversion chain. `update_weight_conversions` is a pure splice that puts
    the GGUF rename map in front of the model-side conversions."""

    def test_update_weight_conversions_injects_nothing(self):
        """Mirror of `Fp8Quantizer.update_weight_conversions` semantics: no
        scale-fusion / dequant op gets prepended on the GGUF side (GGUF blocks
        carry their own scale; no fusion needed)."""
        from transformers import GgufQuantizeConfig
        from transformers.core_model_loading import WeightConverter, WeightRenaming
        from transformers.gguf_conversion_ops import GGUFDequantize, Unsqueeze
        from transformers.quantizers.quantizer_gguf import GGUFQuantizer

        # Fake the rename map (no .gguf file needed for this hook). A
        # WeightConverter must carry at least one op, so use a harmless Unsqueeze.
        fake_mapping = [
            WeightRenaming(source_patterns=r"^foo$", target_patterns="bar"),
            WeightConverter(source_patterns="x", target_patterns="y", operations=[Unsqueeze(0)]),
        ]
        q = GGUFQuantizer(GgufQuantizeConfig())
        q.weight_mapping = fake_mapping
        out = q.update_weight_conversions([])
        # Output is the rename map untouched (identity splice).
        self.assertEqual(len(out), 2)
        for orig, new in zip(fake_mapping, out):
            self.assertIs(orig, new)
        # No conversion op was prepended into either rule.
        for rule in out:
            if isinstance(rule, WeightConverter):
                for op in rule.operations:
                    self.assertNotIsInstance(
                        op,
                        GGUFDequantize,
                        f"{rule}: GGUFDequantize should never be auto-prepended — "
                        f"the dequant decision lives in load_checkpoint_state.",
                    )


@require_torch
@require_gguf
class GgufExpertsRegistryTests(unittest.TestCase):
    """MoE archs with GGUF rules either ship a `GgufExperts` subclass in
    `MODEL_TYPE_TO_GGUF_EXPERTS` (byte-passthrough path) or are explicitly
    excluded with a reason in the registry source."""

    # Archs that are MoE but intentionally NOT in MODEL_TYPE_TO_GGUF_EXPERTS,
    # with the reason captured here so the exclusion is reviewable.
    _MOE_ARCHS_EXCLUDED = {
        "gpt_oss": (
            "Transposed `(in, out)` gate_up layout + per-expert biases. Ships no "
            "transposed-aware mul_mat_id kernel today; falls back to bf16 via "
            "load_checkpoint_state(dequantize=True) and the stock `GptOssExperts` "
            "+ `batched_mm_experts_forward(is_transposed=True, has_bias=True)`."
        ),
    }

    def test_moe_archs_have_experts_class_or_documented_exclusion(self):
        if not is_torch_available():
            self.skipTest("requires torch")
        from transformers.integrations.gguf_linear import MODEL_TYPE_TO_GGUF_EXPERTS

        moe_archs = sorted(m for m in EXPECTED_MODEL_TYPES if "moe" in m or m in ("minimax_m2", "gpt_oss"))
        for arch in moe_archs:
            with self.subTest(arch=arch):
                if arch in self._MOE_ARCHS_EXCLUDED:
                    self.assertNotIn(arch, MODEL_TYPE_TO_GGUF_EXPERTS)
                    continue
                self.assertIn(
                    arch,
                    MODEL_TYPE_TO_GGUF_EXPERTS,
                    f"{arch}: MoE arch has GGUF rename rules but no entry in "
                    f"MODEL_TYPE_TO_GGUF_EXPERTS. Add a class (or document the exclusion).",
                )


@require_torch
@require_gguf
class GgufQuantTypeCoverageTests(unittest.TestCase):
    """Every quant_type the kernels claim to support must have both a
    `_KERNEL_FMT` mapping (so module __init__ can resolve the metal op name)
    AND a working torch dequant kernel (so the dequant path can fall back
    cleanly on CPU / CUDA)."""

    def test_quant_info_covers_expected_types(self):
        from transformers.integrations.gguf_linear import _QUANT_INFO

        missing = sorted(set(EXPECTED_KERNEL_QUANT_TYPES) - set(_QUANT_INFO))
        self.assertFalse(missing, f"`_QUANT_INFO` is missing: {missing}")

    @parameterized.expand([(qt,) for qt in EXPECTED_KERNEL_QUANT_TYPES])
    def test_kernel_fmt_entry_exists(self, quant_type: str):
        from transformers.integrations.gguf_kernels import _KERNEL_FMT

        self.assertIn(
            quant_type,
            _KERNEL_FMT,
            f"{quant_type}: missing from `_KERNEL_FMT`; module __init__ will fail to resolve `mul_mat_<fmt>_f32`.",
        )

    @parameterized.expand([(qt,) for qt in EXPECTED_KERNEL_QUANT_TYPES])
    def test_torch_dequant_kernel_exists(self, quant_type: str):
        """The CPU dequant kernel must exist for the same set of types, so
        non-MPS hosts (and `dequantize=True`) have a path."""
        import gguf

        from transformers.integrations.gguf_dequant import supported_quant_types

        ggml_type = getattr(gguf.GGMLQuantizationType, quant_type)
        self.assertIn(
            ggml_type,
            supported_quant_types(),
            f"{quant_type}: no torch dequant kernel registered in `_DISPATCH`.",
        )


@require_torch
@require_gguf
class GgufLoadCheckpointStateDequantDecisionTests(unittest.TestCase):
    """The dequant decision in `load_checkpoint_state` is what makes the
    no-op-in-update_weight_conversions design work. Pin its branches:

    * `dequantize=True` → every quantized tensor is dequanted at load time.
    * Default + MPS available → only embedding-bound tensors are pre-dequanted;
      the rest stay as `GGUFQuantizedTensor` for the swap path.
    * Default + no MPS → falls back to full dequant.
    """

    def _run(self, monkeypatch, dequantize: bool, mps_available: bool):
        import torch

        from transformers import GgufQuantizeConfig
        from transformers.integrations.gguf_dequant import GGUFQuantizedTensor
        from transformers.quantizers.quantizer_gguf import GGUFQuantizer

        # Fake a `load_gguf_checkpoint` return: a couple of quantized tensors
        # under realistic names + one already-fp norm.
        bytes32 = torch.zeros(32, dtype=torch.uint8)

        def fake_quant_tensor():
            import gguf

            t = GGUFQuantizedTensor(bytes32.clone(), quant_type=gguf.GGMLQuantizationType.Q4_0)
            return t

        fake_tensors = {
            "token_embd.weight": fake_quant_tensor(),  # embedding → must dequant
            "blk.0.attn_q.weight": fake_quant_tensor(),  # linear → swap-path target
            "blk.0.attn_norm.weight": torch.ones(8, dtype=torch.float32),  # already fp
        }

        def fake_load_gguf_checkpoint(_path, return_tensors=True):
            return {"tensors": dict(fake_tensors), "weight_mapping": []}

        # Patch the loader + the dequant kernel so we don't need a real GGUF file
        # or torch ops.
        import transformers.integrations.gguf_dequant as _dq
        import transformers.modeling_gguf_pytorch_utils as _mod

        monkeypatch(_mod, "load_gguf_checkpoint", fake_load_gguf_checkpoint)
        called = {"n": 0}

        def fake_dequantize(t, quant_type, device=None):
            called["n"] += 1
            return torch.zeros(8, 16, dtype=torch.float32)

        monkeypatch(_dq, "dequantize_gguf_tensor", fake_dequantize)
        # `load_checkpoint_state` gates the byte-passthrough path purely on MPS
        # availability (the Metal kernels are resolved later, when GgufLinear /
        # GgufExperts are built), so simulating MPS is enough here.
        monkeypatch(torch.backends.mps, "is_available", lambda: mps_available)

        q = GGUFQuantizer(GgufQuantizeConfig(dequantize=dequantize))
        out = q.load_checkpoint_state("/does/not/exist")
        return out, called["n"]

    def _monkeypatcher(self):
        """Pure-stdlib monkeypatch: save the original, restore via addCleanup."""

        def patch(obj, name, value):
            sentinel = object()
            original = getattr(obj, name, sentinel)
            setattr(obj, name, value)
            self.addCleanup(setattr, obj, name, original) if original is not sentinel else self.addCleanup(
                delattr, obj, name
            )

        return patch

    def test_dequant_true_dequants_everything(self):
        out, n = self._run(self._monkeypatcher(), dequantize=True, mps_available=True)
        self.assertEqual(n, 2, "both quantized tensors should have been dequanted up-front")
        for key in ("token_embd.weight", "blk.0.attn_q.weight"):
            from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

            self.assertNotIsInstance(out[key], GGUFQuantizedTensor, key)

    def test_default_mps_keeps_linear_targets_as_bytes(self):
        out, n = self._run(self._monkeypatcher(), dequantize=False, mps_available=True)
        self.assertEqual(n, 1, "only token_embd should be pre-dequanted on the MPS swap path")
        from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

        self.assertNotIsInstance(out["token_embd.weight"], GGUFQuantizedTensor)
        self.assertIsInstance(out["blk.0.attn_q.weight"], GGUFQuantizedTensor)

    def test_default_no_mps_falls_back_to_full_dequant(self):
        out, n = self._run(self._monkeypatcher(), dequantize=False, mps_available=False)
        self.assertEqual(n, 2, "no MPS → dequant everything so CPU / CUDA can run as plain fp")
        from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

        for key in ("token_embd.weight", "blk.0.attn_q.weight"):
            self.assertNotIsInstance(out[key], GGUFQuantizedTensor, key)


@require_torch
@require_gguf
class GgufUniformSwapTests(unittest.TestCase):
    """The swap is uniform (no rename, no per-module plan): placeholders are sized
    from the Q4_K_M average, the kernel path is gated by a coarse all-supported
    check, and quant types are bound (or float sources reverted) post-load."""

    def test_bytes_per_row_exact_and_average(self):
        from transformers.integrations.gguf_linear import _AVG_BYTES_PER_ELEM, _bytes_per_row

        # Known quant type → exact width (Q4_0 = 18 bytes per 32 elems).
        self.assertEqual(_bytes_per_row(64, "Q4_0"), (64 // 32) * 18)
        # Unknown (uniform-swap placeholder) → Q4_K_M average.
        self.assertEqual(_bytes_per_row(256, None), max(1, round(256 * _AVG_BYTES_PER_ELEM)))

    def test_unsupported_quant_forces_full_dequant(self):
        """A file containing a quant type with no metal kernel (Q2_K) must clear
        `_all_supported`, so the whole model dequantizes even on (simulated) MPS."""
        import gguf
        import torch

        import transformers.integrations.gguf_dequant as _dq
        import transformers.modeling_gguf_pytorch_utils as _mod
        from transformers import GgufQuantizeConfig
        from transformers.integrations.gguf_dequant import GGUFQuantizedTensor
        from transformers.quantizers.quantizer_gguf import GGUFQuantizer

        def patch(obj, name, value):
            sentinel = object()
            orig = getattr(obj, name, sentinel)
            setattr(obj, name, value)
            self.addCleanup(setattr, obj, name, orig) if orig is not sentinel else self.addCleanup(
                delattr, obj, name
            )

        types = {"blk.0.attn_q.weight": gguf.GGMLQuantizationType.Q2_K}  # unsupported

        def fake_load(_path, return_tensors=True):
            tensors = {n: GGUFQuantizedTensor(torch.zeros(32, dtype=torch.uint8), quant_type=qt) for n, qt in types.items()}
            return {"tensors": tensors, "weight_mapping": [], "tensor_quant_types": dict(types)}

        patch(_mod, "load_gguf_checkpoint", fake_load)
        called = {"n": 0}

        def fake_dequant(t, quant_type, device=None):
            called["n"] += 1
            return torch.zeros(8, 16, dtype=torch.float32)

        patch(_dq, "dequantize_gguf_tensor", fake_dequant)
        patch(torch.backends.mps, "is_available", lambda: True)

        q = GGUFQuantizer(GgufQuantizeConfig())
        out = q.load_checkpoint_state("/does/not/exist")
        self.assertFalse(q._all_supported)
        self.assertEqual(called["n"], 1, "unsupported quant must be dequantized despite MPS")
        self.assertNotIsInstance(out["blk.0.attn_q.weight"], GGUFQuantizedTensor)


if __name__ == "__main__":
    unittest.main()
