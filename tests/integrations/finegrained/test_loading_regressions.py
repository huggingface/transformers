# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests that load, run and save local quantized checkpoints on CPU with real model mappings."""

import unittest

import torch

from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import FineGrainedConfig, Fp8Config

from .utils import _ChainAssertions


@require_torch
class FineGrainedLoadingRegressionTest(_ChainAssertions, unittest.TestCase):
    def test_mixed_block_sizes_use_the_current_modules_grid(self):
        # Both block sizes give a 2x2 scale grid for 192x192 weights, with different boundaries
        quantizer = FineGrainedHfQuantizer(
            FineGrainedConfig(
                dtype_plan={
                    "a$": Fp8Config(weight_block_size=(96, 96)),
                    "b$": Fp8Config(weight_block_size=(128, 128)),
                },
                dequantize=True,
            )
        )
        model = torch.nn.Module()
        model.a = torch.nn.Linear(192, 192, bias=False)
        model.b = torch.nn.Linear(192, 192, bias=False)
        # the members' dequantize converters read the same keys, so the host merges them into one
        converter = next(c for c in quantizer.update_weight_conversions([], model) if c.target_patterns == ["weight"])
        weight_pattern, scale_pattern = converter.source_patterns[:2]
        scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        for name, block in (("a", 96), ("b", 128)):
            with self.subTest(module=name):
                result = converter.operations[0].convert(
                    {weight_pattern: torch.ones(192, 192).to(torch.float8_e4m3fn), scale_pattern: scales},
                    full_layer_name=f"{name}.weight",
                    model=model,
                )["weight"]
                expected = scales.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:192, :192]
                torch.testing.assert_close(result, expected)
