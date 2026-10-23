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

import os
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    NVFP4Config,
    Qwen2MoeConfig,
)
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import FineGrainedConfig, Fp8Config

from .utils import _ChainAssertions


def _quantized_weights(model):
    """Return the dequantized `q_proj` and expert weights, the ones the tiny checkpoints ship quantized."""
    suffixes = ("q_proj.weight", "experts.gate_up_proj", "experts.down_proj")
    weights = {name: tensor for name, tensor in model.state_dict().items() if name.endswith(suffixes)}
    assert len(weights) == len(suffixes), sorted(weights)
    return weights


@require_torch
class FineGrainedLoadingRegressionTest(_ChainAssertions, unittest.TestCase):
    def _checkpoint(self, directory, moe=False):
        config_class = Qwen2MoeConfig if moe else LlamaConfig
        config = config_class(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            **({"num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 32} if moe else {}),
        )
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
        # Saving the model first gives the checkpoint key layout, including split experts
        model.save_pretrained(directory)
        filename = os.path.join(directory, "model.safetensors")
        state = load_file(filename)
        quantized = {}
        # Every weight dequantizes to exactly 1/64, so all tests compare to one constant.
        # E2M1 code 2 is 1.0, the block scale is 1, and the global is 1/64.
        for name, tensor in state.items():
            if tensor.ndim != 2 or name.endswith(("embed_tokens.weight", "lm_head.weight", "mlp.gate.weight")):
                quantized[name] = tensor
                continue
            stem = name.removesuffix("weight")
            quantized[name] = torch.full((tensor.shape[0], tensor.shape[1] // 2), 0x22, dtype=torch.uint8)
            quantized[stem + "weight_scale"] = torch.ones(
                tensor.shape[0], tensor.shape[1] // 16, dtype=torch.float8_e4m3fn
            )
            quantized[stem + "weight_scale_2"] = torch.tensor(1 / 64)
        config.quantization_config = NVFP4Config(modules_to_not_convert=["lm_head", "mlp.gate"])
        config.save_pretrained(directory)
        save_file(quantized, filename, metadata={"format": "pt"})

    def test_dequantized_dense_model_respects_requested_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            self._checkpoint(directory)
            model = AutoModelForCausalLM.from_pretrained(
                directory, quantization_config=NVFP4Config(dequantize=True), dtype=torch.float16, device_map="cpu"
            ).eval()
            self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float16})
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(torch.tensor([[1, 2, 3]])).logits).all())

    def test_dequantized_moe_can_save_and_reload(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as saved:
            self._checkpoint(directory, moe=True)
            model = AutoModelForCausalLM.from_pretrained(
                directory, quantization_config=NVFP4Config(dequantize=True), dtype=torch.float32, device_map="cpu"
            ).eval()
            model.save_pretrained(saved)
            reloaded, info = AutoModelForCausalLM.from_pretrained(saved, output_loading_info=True)
            self.assertEqual(info["missing_keys"], set())
            self.assertEqual(info["unexpected_keys"], set())
            self.assertFalse(hasattr(reloaded.config, "quantization_config"))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(reloaded.state_dict()[key], value)
            with torch.no_grad():
                inputs = torch.tensor([[1, 2, 3]])
                torch.testing.assert_close(model(inputs).logits, reloaded(inputs).logits)

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
