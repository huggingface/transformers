# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Slow tests that assert every supported GGUF model_type loads with no missing or
unexpected keys.

Any missing/unexpected key means the static rename rules in
``modeling_gguf_pytorch_utils._GGUF_ARCH_CONVERTERS`` are incomplete for that
architecture — the model still loads but parameters get newly initialized from
scratch, silently producing a degraded model. These tests fail-fast on that.

Each cell is one ``(model_type, repo, gguf_file)``. The repo + file are picked
to match the architectures already exercised by ``test_ggml.py`` and to keep
download sizes manageable. New supported model types should add a row here.
"""

from __future__ import annotations

import unittest

from parameterized import parameterized

from transformers import AutoModelForCausalLM
from transformers.modeling_gguf_pytorch_utils import _GGUF_ARCH_CONVERTERS
from transformers.testing_utils import require_gguf, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


# (model_type, gguf_repo, gguf_file). One small/representative GGUF per arch.
# Sourced from `test_ggml.py`; smallest available quant chosen to limit
# download size while still exercising all renamings.
GGUF_LOAD_CELLS: list[tuple[str, str, str]] = [
    # Llama / RoPE family
    ("llama", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"),
    ("mistral", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q4_0.gguf"),
    ("phi3", "microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4.gguf"),
    ("qwen2", "Qwen/Qwen1.5-0.5B-Chat-GGUF", "qwen1_5-0_5b-chat-q4_0.gguf"),
    ("qwen3", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
    ("deci", "Deci/DeciLM-7B-instruct-GGUF", "decilm-7b-uniform-gqa-q8_0.gguf"),
    # MoE families
    ("qwen2_moe", "gdax/Qwen1.5-MoE-A2.7B_gguf", "Qwen1.5-MoE-A2.7B_q4_k_m.gguf"),
    ("qwen3_moe", "Qwen/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf"),
    ("gpt_oss", "unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q5_K_M.gguf"),
    # Norm-subtract-one variants
    ("nemotron", "bartowski/Nemotron-Mini-4B-Instruct-GGUF", "Nemotron-Mini-4B-Instruct-Q6_K.gguf"),
    ("gemma2", "bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q3_K_L.gguf"),
    ("gemma3_text", "unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-BF16.gguf"),
    ("gemma3", "google/gemma-3-4b-it-qat-q4_0-gguf", "gemma-3-4b-it-q4_0.gguf"),
    # Misc archs
    ("bloom", "afrideva/bloom-560m-GGUF", "bloom-560m.fp16.gguf"),
    ("gpt2", "mradermacher/gpt2-GGUF", "gpt2.f16.gguf"),
    ("falcon", "medmekk/falcon-7b-gguf", "falcon-7b-fp16.gguf"),
    ("starcoder2", "brittlewis12/starcoder2-3b-GGUF", "starcoder2-3b.fp16.gguf"),
    ("stablelm", "afrideva/stablelm-3b-4e1t-GGUF", "stablelm-3b-4e1t.q4_k_m.gguf"),
    ("mamba", "jpodivin/mamba-2.8b-hf-GGUF", "ggml-model-Q6_K.gguf"),
    ("lfm2", "LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf"),
]


@require_gguf
@slow
class GgufLoadCompletenessTests(unittest.TestCase):
    """Per-arch GGUF load completeness — no missing, no unexpected keys."""

    def _assert_clean_load(self, model_type: str, repo: str, gguf_file: str):
        # `output_loading_info=True` returns the `LoadStateDictInfo` as a dict
        # alongside the model: {"missing_keys", "unexpected_keys",
        # "mismatched_keys", "error_msgs"}.
        model, info = AutoModelForCausalLM.from_pretrained(
            repo,
            gguf_file=gguf_file,
            device_map="cpu",
            dtype=torch.bfloat16,
            output_loading_info=True,
        )
        missing = sorted(info["missing_keys"])
        unexpected = sorted(info["unexpected_keys"])
        mismatched = sorted(info["mismatched_keys"])
        del model

        self.assertEqual(
            missing,
            [],
            f"{model_type} ({repo}/{gguf_file}): missing keys after GGUF load — "
            f"GGUF→HF rename rules are incomplete. Missing: {missing}",
        )
        self.assertEqual(
            unexpected,
            [],
            f"{model_type} ({repo}/{gguf_file}): unexpected keys after GGUF load — "
            f"GGUF→HF rename rules produce names the model doesn't define. Unexpected: {unexpected}",
        )
        self.assertEqual(
            mismatched,
            [],
            f"{model_type} ({repo}/{gguf_file}): shape-mismatched keys — "
            f"rename rules map to the wrong target tensor. Mismatched: {mismatched}",
        )

    @parameterized.expand(GGUF_LOAD_CELLS)
    def test_no_missing_or_unexpected_keys(self, model_type: str, repo: str, gguf_file: str):
        self._assert_clean_load(model_type, repo, gguf_file)

    def test_every_registered_model_type_has_a_load_cell(self):
        """Any model_type registered in ``_GGUF_ARCH_CONVERTERS`` should be exercised
        by a load cell above — otherwise rename regressions can land silently.

        T5 variants (``t5``, ``t5encoder``, ``umt5``) are encoder/decoder models
        and are covered separately by the seq2seq paths in ``test_ggml.py``; we
        accept them as known omissions here.
        """
        covered = {row[0] for row in GGUF_LOAD_CELLS}
        registered = set(_GGUF_ARCH_CONVERTERS)
        known_skip = {"t5", "t5encoder", "umt5", "cohere", "minimax_m2"}
        gap = sorted(registered - covered - known_skip)
        self.assertFalse(
            gap,
            f"Registered GGUF model_types lack a load-completeness test cell: {gap}. "
            f"Add a (model_type, repo, gguf_file) row to GGUF_LOAD_CELLS — small Q4 quants preferred.",
        )
