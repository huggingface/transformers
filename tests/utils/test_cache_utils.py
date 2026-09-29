# Copyright 2023 HuggingFace Inc.
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

from parameterized import parameterized

from transformers.generation.configuration_utils import ALL_CACHE_IMPLEMENTATIONS
from transformers.testing_utils import (
    backend_device_count,
    is_optimum_quanto_available,
    is_torch_available,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import pipeline


# FIXME: offloaded cache is skipped because it needs `offload_only_non_sliding=False`
# but we can't configure cache through `generate()`
TEST_CACHE_IMPLEMENTATIONS = [
    cache_name
    for cache_name in ALL_CACHE_IMPLEMENTATIONS
    if cache_name not in ["offloaded", "offloaded_hybrid", "offloaded_static", "offloaded_hybrid_chunked"]
]


def _skip_on_failed_cache_prerequisites(test, cache_implementation):
    """Function to skip tests on failed cache prerequisites, given a cache implementation"""
    if cache_implementation == "quantized" and not is_optimum_quanto_available():
        test.skipTest("Quanto is not available")
    if "offloaded" in cache_implementation:
        has_accelerator = torch_device is not None and torch_device != "cpu"
        if not has_accelerator:
            test.skipTest("Offloaded caches require an accelerator")
        if cache_implementation in ["offloaded_static", "offloaded_hybrid_chunked"]:
            if backend_device_count(torch_device) != 1:
                test.skipTest("Offloaded static caches require exactly 1 accelerator")


@require_torch_accelerator
class CacheHardIntegrationTest(unittest.TestCase):
    @require_torch_accelerator
    @parameterized.expand(TEST_CACHE_IMPLEMENTATIONS)
    @slow
    def test_cache_gptj_model(self, cache_implementation):
        """Tests caches with GPT-J model. Regression test for https://github.com/huggingface/transformers/pull/34799"""
        _skip_on_failed_cache_prerequisites(self, cache_implementation)

        # Use a dedicated safetensors repo to avoid Xet chunk corruption that affects pytorch_model.bin
        # (corruption is intermittent but causes wrong wte.weight values and wrong golden outputs)
        model_id = "hf-internal-testing/tiny-random-GPTJForCausalLM-for-CacheHardIntegrationTest"
        pipe = pipeline("text-generation", model=model_id, dtype=torch.bfloat16)
        pipe.model.config.sliding_window = (
            256 if cache_implementation in ["sliding_window", "hybrid", "hybrid_chunked"] else None
        )
        out = pipe(
            "hello world",
            cache_implementation=cache_implementation,
            max_new_tokens=10,
            do_sample=False,
            disable_compile=True,
            return_tensors=True,
        )[0]["generated_token_ids"][-10:]
        EXPECTED_OUTPUT = [879, 175, 39, 141, 1000, 975, 951, 991, 683, 441]
        self.assertListEqual(out, EXPECTED_OUTPUT)
