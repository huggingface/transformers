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
from transformers.gguf_conversion_ops import GGUFDequantize
from transformers.modeling_gguf_pytorch_utils import _GGUF_ARCH_CONVERTERS, get_gguf_converters


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
            # When a rule has transform ops, the first one must be GGUFDequantize so the
            # initial source-pattern → resolved-target rename happens before any tensor
            # transform — mirrors how `Fp8Dequantize` is prepended by `Fp8Quantizer`.
            if isinstance(rule, WeightConverter):
                self.assertIsInstance(
                    rule.operations[0],
                    GGUFDequantize,
                    f"{model_type}: WeightConverter must start with GGUFDequantize, "
                    f"got {type(rule.operations[0]).__name__} (use `gguf_rename(...)` to build entries).",
                )

    def test_pure_renames_use_weight_renaming(self):
        """No-op entries (pure key renames) should use the cheaper WeightRenaming class."""
        for model_type, rules in _GGUF_ARCH_CONVERTERS.items():
            for rule in rules:
                if isinstance(rule, WeightConverter):
                    # A WeightConverter with only GGUFDequantize and nothing else is just a
                    # rename pretending to be a converter — should be a WeightRenaming.
                    if len(rule.operations) == 1 and isinstance(rule.operations[0], GGUFDequantize):
                        self.fail(
                            f"{model_type}: rule {rule.source_patterns}→{rule.target_patterns} "
                            f"is a pure rename but uses WeightConverter. Drop the empty ops list "
                            f"so `gguf_rename` produces a WeightRenaming instead."
                        )
