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

from transformers.integrations.finegrained.conversions import FineGrainedDequantize
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import FineGrainedConfig, Fp8Config

from .utils import _ChainAssertions, _load_and_save, _per_expert


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
        scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        for name, block in (("a", 96), ("b", 128)):
            with self.subTest(module=name):
                result = FineGrainedDequantize(quantizer).convert(
                    {"weight": torch.ones(192, 192).to(torch.float8_e4m3fn), "weight_scale_inv": scales},
                    full_layer_name=f"{name}.weight",
                    model=model,
                )[f"{name}.weight"]
                expected = scales.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:192, :192]
                torch.testing.assert_close(result, expected)

    def test_modelopt_static_fp8_checkpoint_loads_and_saves_dense_and_experts(self):
        checkpoint = _per_expert(
            lambda rows, cols: {
                "weight": torch.full((rows, cols), 2, dtype=torch.float8_e4m3fn),
                "weight_scale": torch.tensor(1 / 128),
                "input_scale": torch.tensor(1 / 256),
            }
        )
        for name in checkpoint:
            if name.endswith(".up_proj.weight_scale"):
                checkpoint[name] = torch.tensor(1 / 64)
        config = AutoQuantizationConfig.from_dict({"quant_method": "modelopt", "quant_algo": "FP8"})
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        q_proj = model.model.layers[0].self_attn.q_proj
        self.assertEqual(q_proj.weight.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(q_proj.weight_scale_inv, torch.tensor(1 / 128))
        torch.testing.assert_close(q_proj.activation_scale, torch.tensor(1 / 256))
        experts = model.model.layers[0].mlp.experts
        torch.testing.assert_close(
            experts.gate_up_proj_scale_inv,
            torch.tensor([[[1 / 128], [1 / 64]], [[1 / 128], [1 / 64]]]),
        )
        self.assertSavesBack(saved, checkpoint)
