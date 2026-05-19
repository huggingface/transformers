# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fast (non-slow) tests that the static GGUF→HF rule table is well-formed
and covers every model_type previously supported by the legacy
``TENSOR_PROCESSORS`` map (and every model_type exercised by ``test_ggml.py``).

These tests do not download any GGUF file — they only inspect the registry,
so they can run on every PR without RUN_SLOW.
"""

from __future__ import annotations

import unittest

from parameterized import parameterized

from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.modeling_gguf_pytorch_utils import _GGUF_ARCH_CONVERTERS, get_gguf_converters
from transformers.quantizers.quantizer_gguf import GGUFQuantizer


# Every HF model_type the public GGUF integration tests exercise. Keep this
# list in sync with ``tests/quantization/ggml/test_ggml.py``.
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
        # MoE families
        "qwen2_moe",
        "qwen3_moe",
        "minimax_m2",
        "gpt_oss",
        # Norm-subtract-one variants
        "nemotron",
        "gemma2",
        "gemma3",
        "gemma3_text",
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


class GgufArchCoverageTests(unittest.TestCase):
    def test_every_expected_model_type_is_registered(self):
        missing = sorted(set(EXPECTED_MODEL_TYPES) - set(_GGUF_ARCH_CONVERTERS))
        self.assertFalse(
            missing,
            f"Model types previously supported by the GGUF loader are no longer registered "
            f"in `_GGUF_ARCH_CONVERTERS`: {missing}. Add an entry (or alias to an existing "
            f"converter list) so the static rule table matches the legacy coverage.",
        )

    @parameterized.expand([(m,) for m in sorted(_GGUF_ARCH_CONVERTERS)])
    def test_arch_entry_is_well_formed(self, model_type: str):
        rules = get_gguf_converters(model_type)
        self.assertGreater(len(rules), 0, f"{model_type}: empty converter list")
        for rule in rules:
            self.assertIsInstance(
                rule,
                (WeightRenaming, WeightConverter),
                f"{model_type}: every entry must be a WeightRenaming or WeightConverter, got {type(rule).__name__}",
            )

    def test_quantizer_prepends_gguf_dequantize_to_every_converter(self):
        """``GGUFQuantizer.update_weight_conversions`` injects ``GGUFDequantize`` at the head
        of every ``WeightConverter`` op chain — same pattern as ``Fp8Quantizer``.
        ``WeightRenaming`` entries pass through unmodified.
        """
        from transformers.gguf_conversion_ops import GGUFDequantize

        for model_type, rules in _GGUF_ARCH_CONVERTERS.items():
            quantizer = GGUFQuantizer(weight_mapping=rules)
            out = quantizer.update_weight_conversions([])
            for rule in out:
                if isinstance(rule, WeightConverter):
                    self.assertIsInstance(
                        rule.operations[0],
                        GGUFDequantize,
                        f"{model_type}: WeightConverter not prefixed with GGUFDequantize.",
                    )
