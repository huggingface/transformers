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
from unittest import mock

import torch
from safetensors.torch import load_file, save_file

import transformers.conversion_mapping as conversion_mapping
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    NVFP4Config,
    Qwen2MoeConfig,
)
from transformers.core_model_loading import WeightRenaming
from transformers.integrations.finegrained.conversions import FineGrainedDequantize
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.quantizers.quantizer_finegrained import FineGrainedHfQuantizer
from transformers.testing_utils import require_torch
from transformers.utils.quantization_config import FineGrainedConfig, Fp8Config, Mxfp4Config, Mxfp8Config

from .utils import _ChainAssertions, _load_and_save, _per_expert


def _quantized_weights(model):
    """Return the dequantized `q_proj` and expert weights, the ones the tiny checkpoints ship quantized."""
    suffixes = ("q_proj.weight", "experts.gate_up_proj", "experts.down_proj")
    weights = {name: tensor for name, tensor in model.state_dict().items() if name.endswith(suffixes)}
    assert len(weights) == len(suffixes), sorted(weights)
    return weights


@require_torch
class FineGrainedLoadingRegressionTest(_ChainAssertions, unittest.TestCase):
    def _checkpoint(self, directory, moe=False, mixed=False, modelopt_mixed=False):
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
        # Every format dequantizes to exactly 1/64, so all tests compare to one constant.
        # E2M1 code 2 is 1.0, the block scale is 1, and the global is 1/64. compressed-tensors ships it as 64.
        for name, tensor in state.items():
            if tensor.ndim != 2 or name.endswith(("embed_tokens.weight", "lm_head.weight", "mlp.gate.weight")):
                quantized[name] = tensor
                continue
            if modelopt_mixed and ".q_proj." not in name and ".mlp.experts." not in name:
                quantized[name] = tensor
                continue
            stem = name.removesuffix("weight")
            if modelopt_mixed and ".q_proj." in name:
                quantized[name] = torch.full_like(tensor, 2, dtype=torch.float8_e4m3fn)
                quantized[stem + "weight_scale"] = torch.tensor(1 / 128)
                quantized[stem + "input_scale"] = torch.tensor(1.0)
                continue
            ct = mixed and ".q_proj." in name
            quantized[stem + ("weight_packed" if ct else "weight")] = torch.full(
                (tensor.shape[0], tensor.shape[1] // 2), 0x22, dtype=torch.uint8
            )
            quantized[stem + "weight_scale"] = torch.ones(
                tensor.shape[0], tensor.shape[1] // 16, dtype=torch.float8_e4m3fn
            )
            quantized[stem + ("weight_global_scale" if ct else "weight_scale_2")] = torch.tensor(
                64.0 if ct else 1 / 64
            )
        config.quantization_config = (
            FineGrainedConfig(
                dtype_plan={r"\.q_proj$": NVFP4Config(checkpoint_format="compressed-tensors"), ".*": NVFP4Config()},
                modules_to_not_convert=["lm_head", "mlp.gate"],
            )
            if mixed
            else NVFP4Config(modules_to_not_convert=["lm_head", "mlp.gate"])
        )
        if modelopt_mixed:
            config.quantization_config = {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                    "model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
                },
            }
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

    def test_mixed_checkpoint_global_conventions_are_scoped_per_module(self):
        with tempfile.TemporaryDirectory() as modelopt, tempfile.TemporaryDirectory() as mixed:
            self._checkpoint(modelopt, moe=True)
            self._checkpoint(mixed, moe=True, mixed=True)
            plain = AutoModelForCausalLM.from_pretrained(
                modelopt, quantization_config=NVFP4Config(dequantize=True), dtype=torch.float32, device_map="cpu"
            )
            loaded = AutoModelForCausalLM.from_pretrained(
                mixed,
                quantization_config=FineGrainedConfig(
                    dtype_plan={
                        r"\.q_proj$": NVFP4Config(checkpoint_format="compressed-tensors"),
                        ".*": NVFP4Config(),
                    },
                    modules_to_not_convert=["lm_head", "mlp.gate"],
                    dequantize=True,
                ),
                dtype=torch.float32,
                device_map="cpu",
            )
            quantized = _quantized_weights(plain)
            for name, tensor in quantized.items():
                torch.testing.assert_close(loaded.state_dict()[name], tensor)
                torch.testing.assert_close(tensor, torch.full_like(tensor, 1 / 64))

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

    def test_modelopt_mixed_precision_checkpoint_loads_and_saves(self):
        for rename_namespace in (False, True):
            with (
                self.subTest(rename_namespace=rename_namespace),
                tempfile.TemporaryDirectory() as directory,
                tempfile.TemporaryDirectory() as saved,
            ):
                self._checkpoint(directory, moe=True, modelopt_mixed=True)
                mapping = conversion_mapping.get_checkpoint_conversion_mapping("qwen2_moe")
                if rename_namespace:
                    config = Qwen2MoeConfig.from_pretrained(directory)
                    layers = config.quantization_config["quantized_layers"]
                    config.quantization_config["quantized_layers"] = {
                        name.replace("model.layers.", "decoder.blocks."): spec for name, spec in layers.items()
                    }
                    config.quantization_config = AutoQuantizationConfig.from_dict(config.quantization_config).to_dict()
                    config.save_pretrained(directory)
                    filename = os.path.join(directory, "model.safetensors")
                    save_file(
                        {
                            name.replace("model.layers.", "decoder.blocks."): value
                            for name, value in load_file(filename).items()
                        },
                        filename,
                        metadata={"format": "pt"},
                    )
                    mapping = [WeightRenaming(r"^decoder\.blocks\.", "model.layers."), *mapping]
                with mock.patch.dict(conversion_mapping._checkpoint_conversion_mapping_cache, {"qwen2_moe": mapping}):
                    model, info = AutoModelForCausalLM.from_pretrained(
                        directory,
                        quantization_config=FineGrainedConfig(dequantize=True),
                        dtype=torch.float16,
                        device_map="cpu",
                        output_loading_info=True,
                    )
                    self.assertEqual(info["missing_keys"], set())
                    self.assertEqual(info["unexpected_keys"], set())
                    self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float16})
                    state = model.state_dict()
                    for tensor in _quantized_weights(model).values():
                        torch.testing.assert_close(tensor, torch.full_like(tensor, 1 / 64))
                    original = load_file(os.path.join(directory, "model.safetensors"))
                    unlisted = "model.layers.0.self_attn.k_proj.weight"
                    checkpoint_key = (
                        unlisted.replace("model.layers.", "decoder.blocks.") if rename_namespace else unlisted
                    )
                    torch.testing.assert_close(state[unlisted], original[checkpoint_key].to(torch.float16))
                    with torch.no_grad():
                        inputs = torch.tensor([[1, 2, 3]])
                        expected = model(inputs).logits
                        self.assertTrue(torch.isfinite(expected).all())
                    model.save_pretrained(saved)
                    reloaded = AutoModelForCausalLM.from_pretrained(saved).eval()
                    with torch.no_grad():
                        torch.testing.assert_close(reloaded(inputs).logits, expected)

    def test_modelopt_mx_checkpoint_loads_and_saves_dense_and_expert_weights(self):
        for format_name, config_class in (("mxfp8", Mxfp8Config), ("mxfp4", Mxfp4Config)):
            with self.subTest(format=format_name):
                packed = format_name == "mxfp4"
                checkpoint = _per_expert(
                    lambda rows, cols: {
                        "weight": (
                            torch.full((rows, cols // 2), 0xB2, dtype=torch.uint8)
                            if packed
                            else torch.full((rows, cols), 2, dtype=torch.float8_e4m3fn)
                        ),
                        "weight_scale": torch.full((rows, cols // 32), 121, dtype=torch.uint8),
                    }
                )
                model, info, saved = _load_and_save(checkpoint, config_class(checkpoint_format="modelopt"))
                self.assertLoadsCleanly(info)
                q_proj = model.model.layers[0].self_attn.q_proj
                experts = model.model.layers[0].mlp.experts
                if packed:
                    self.assertEqual(q_proj.weight.dtype, torch.int8)
                    self.assertEqual(experts.gate_up_proj.dtype, torch.int8)
                    torch.testing.assert_close(
                        q_proj.weight.view(torch.uint8), checkpoint["model.layers.0.self_attn.q_proj.weight"]
                    )
                self.assertSavesBack(saved, checkpoint)

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

    def test_compressed_tensors_mxfp8_scales_reach_dense_and_expert_modules(self):
        config = AutoQuantizationConfig.from_dict(
            {
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {"num_bits": 8, "type": "float", "strategy": "group", "group_size": 32},
                        "input_activations": {
                            "num_bits": 8,
                            "type": "float",
                            "strategy": "group",
                            "group_size": 32,
                            "dynamic": True,
                        },
                    }
                },
            }
        )
        self.assertIsInstance(config, Mxfp8Config)
        checkpoint = _per_expert(
            lambda rows, cols: {
                "weight": torch.full((rows, cols), 2, dtype=torch.float8_e4m3fn),
                "weight_scale": torch.full((rows, cols // 32), 121, dtype=torch.uint8),
            }
        )
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        torch.testing.assert_close(
            model.model.layers[0].self_attn.q_proj.weight_scale_inv,
            checkpoint["model.layers.0.self_attn.q_proj.weight_scale"],
        )
        self.assertSavesBack(saved, checkpoint)
