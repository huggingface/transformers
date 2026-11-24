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

import gc
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoModel, AutoModelForCausalLM, CompressedTensorsConfig, Qwen3Config
from transformers.testing_utils import (
    require_compressed_tensors,
    require_torch,
    require_torch_accelerator,
    require_torchao_version_greater_or_equal,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.quantizers.auto import AutoHfQuantizer
    from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer


def nvfp4_config():
    weights = {
        "num_bits": 4,
        "type": "float",
        "strategy": "tensor_group",
        "group_size": 16,
        "symmetric": True,
        "dynamic": False,
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "nvfp4": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": {**weights, "dynamic": "local"},
            }
        },
        "ignore": ["lm_head"],
    }


@require_torch
@require_compressed_tensors
class NVFP4ConfigTest(unittest.TestCase):
    def test_loading_options(self):
        config = AutoHfQuantizer.merge_quantization_configs(nvfp4_config(), CompressedTensorsConfig(weight_only=True))
        config.post_init()
        self.assertTrue(config.get_loading_attributes()["weight_only"])
        self.assertNotIn("weight_only", config.to_dict())
        self.assertNotIn("_additional_compression", config.to_dict())
        self.assertFalse(CompressedTensorsConfig.from_dict(config.to_dict()).weight_only)
        for option in ("dequantize", "use_optimized_inference"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "cannot be combined"):
                CompressedTensorsConfig(**nvfp4_config(), weight_only=True, **{option: True}).post_init()
        invalid = nvfp4_config() | {"format": "pack-quantized"}
        with self.assertRaisesRegex(ValueError, "nvfp4-pack-quantized"):
            CompressedTensorsConfig(**invalid, weight_only=True).post_init()
        default_quantizer = CompressedTensorsHfQuantizer(CompressedTensorsConfig(**nvfp4_config()))
        self.assertIs(default_quantizer.is_qat_trainable, False)

    def test_old_torchao_is_an_error(self):
        from transformers.utils.import_utils import is_torchao_available

        quantizer = CompressedTensorsHfQuantizer(CompressedTensorsConfig(**nvfp4_config(), weight_only=True))
        is_torchao_available.cache_clear()
        self.addCleanup(is_torchao_available.cache_clear)
        with (
            patch("transformers.utils.import_utils.is_torch_available", return_value=True),
            patch("transformers.utils.import_utils._is_package_available", return_value=(True, "0.15.0")),
            self.assertRaisesRegex(ImportError, r"torchao>=0\.16\.0"),
        ):
            quantizer.validate_environment(device_map={"": "cpu"})


@require_torch
@require_compressed_tensors
@require_torchao_version_greater_or_equal("0.16.0")
class NVFP4LoadingTest(unittest.TestCase):
    @staticmethod
    def make_checkpoint(directory, model_class=AutoModelForCausalLM, input_global_scale=True):
        from safetensors.torch import save_file

        torch.manual_seed(17)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            max_position_embeddings=64,
            attention_bias=True,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        reference = model_class.from_config(config, dtype=torch.bfloat16, attn_implementation="eager").eval()
        tensors = {name: value.clone() for name, value in reference.state_dict().items()}
        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        for name, module in reference.named_modules():
            if not isinstance(module, torch.nn.Linear) or name == "lm_head":
                continue
            n, k = module.weight.shape
            packed = torch.arange(n * k // 2).to(torch.uint8).reshape(n, k // 2)
            scales = (torch.randint(1, 9, (n, k // 16)).float() / 128).to(torch.float8_e4m3fn)
            del tensors[f"{name}.weight"]
            tensors.update(
                {
                    f"{name}.weight_packed": packed,
                    f"{name}.weight_scale": scales,
                    f"{name}.weight_global_scale": torch.tensor([2.0]),
                }
            )
            if input_global_scale:
                tensors[f"{name}.input_global_scale"] = torch.tensor([3.0])
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, k).long()
            module.weight = torch.nn.Parameter(
                (values[codes] * scales.float().repeat_interleave(16, dim=1) / 2).to(torch.bfloat16),
                requires_grad=False,
            )
        reference.config.quantization_config = nvfp4_config()
        reference.config.save_pretrained(directory)
        weight_map = {}
        # Separate packed weights from their scales to exercise cross-shard collection.
        for index in range(2):
            shard = {key: value for key, value in tensors.items() if key.endswith("weight_packed") == bool(index)}
            filename = f"model-{index + 1:05d}-of-00002.safetensors"
            save_file(shard, os.path.join(directory, filename), metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(shard, filename))
        with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
            json.dump({"metadata": {}, "weight_map": weight_map}, handle)
        return reference, tensors

    @staticmethod
    def load_checkpoint(directory, model_class=AutoModelForCausalLM, device="cpu"):
        return model_class.from_pretrained(
            directory,
            dtype=torch.bfloat16,
            device_map=device,
            quantization_config=CompressedTensorsConfig(weight_only=True),
            attn_implementation="eager",
            local_files_only=True,
            output_loading_info=True,
        )

    def assert_backbone_parity(self, model, reference, tensors):
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        reference = reference.to(model.device)
        actual_base, expected_base = model.base_model, reference.base_model
        for name, module in expected_base.named_modules():
            actual = actual_base.get_submodule(name)
            if isinstance(module, torch.nn.Linear):
                source_name = f"model.{name}" if hasattr(reference, "lm_head") else name
                self.assertIs(type(actual), torch.nn.Linear)
                self.assertIsInstance(actual.weight, NVFP4Tensor)
                self.assertIsNone(actual.weight.act_quant_kwargs)
                self.assertEqual(actual.weight.per_tensor_scale.item(), 0.5)
                torch.testing.assert_close(
                    actual.weight.qdata.cpu(), tensors[f"{source_name}.weight_packed"], rtol=0, atol=0
                )
                torch.testing.assert_close(actual.weight.dequantize(), module.weight, rtol=0, atol=0)
        self.assertNotIsInstance(actual_base.embed_tokens.weight, NVFP4Tensor)
        self.assertEqual(actual_base.rotary_emb.inv_freq.dtype, torch.float32)
        if hasattr(model, "lm_head"):
            self.assertNotIsInstance(model.lm_head.weight, NVFP4Tensor)
        inputs = torch.tensor([[1, 5, 8, 11]], device=model.device)
        with torch.no_grad():
            torch.testing.assert_close(
                actual_base(inputs).last_hidden_state, expected_base(inputs).last_hidden_state, rtol=0, atol=0
            )
            if hasattr(model, "lm_head") and hasattr(reference, "lm_head"):
                torch.testing.assert_close(
                    model.generate(inputs, max_new_tokens=3, do_sample=False),
                    reference.generate(inputs, max_new_tokens=3, do_sample=False),
                    rtol=0,
                    atol=0,
                )

    def test_prefix_loading_and_optional_activation_scale(self):
        for source in (AutoModel, AutoModelForCausalLM):
            with tempfile.TemporaryDirectory() as directory:
                reference, tensors = self.make_checkpoint(
                    directory, source, input_global_scale=source is AutoModelForCausalLM
                )
                for target in (AutoModel, AutoModelForCausalLM):
                    with self.subTest(source=source.__name__, target=target.__name__):
                        model, info = self.load_checkpoint(directory, target)
                        missing = (
                            {"lm_head.weight"} if source is AutoModel and target is AutoModelForCausalLM else set()
                        )
                        extra = {"lm_head.weight"} if source is AutoModelForCausalLM and target is AutoModel else set()
                        self.assertEqual(set(info["missing_keys"]), missing)
                        self.assertEqual(set(info["unexpected_keys"]), extra)
                        self.assertFalse(info["mismatched_keys"] or info["error_msgs"])
                        self.assert_backbone_parity(model, reference, tensors)

    @require_torch_accelerator
    def test_accelerator_loading(self):
        if torch_device not in ("cuda", "xpu"):
            self.skipTest("NVFP4 weight-only loading supports CUDA and XPU accelerators.")
        with tempfile.TemporaryDirectory() as directory:
            reference, tensors = self.make_checkpoint(directory, input_global_scale=False)
            model, info = self.load_checkpoint(directory, device=torch_device)
            self.assertFalse(any(info.values()), info)
            self.assert_backbone_parity(model, reference, tensors)

    def test_invalid_checkpoint_tensors(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            reference, original = self.make_checkpoint(directory)
            name = "model.layers.0.self_attn.q_proj"
            for suffix, value in (
                ("weight_packed", None),
                ("weight_scale", torch.ones(1)),
                ("weight_global_scale", torch.tensor([1e-40])),
                ("input_global_scale", torch.tensor([float("nan")])),
                ("weight_zero_point", torch.tensor([0.0])),
            ):
                with self.subTest(suffix=suffix):
                    # Distinct files avoid rewriting mmap-backed tensors on Windows.
                    case = os.path.join(directory, suffix)
                    reference.config.save_pretrained(case)
                    tensors = dict(original)
                    if value is None:
                        del tensors[f"{name}.{suffix}"]
                    else:
                        tensors[f"{name}.{suffix}"] = value
                    save_file(tensors, os.path.join(case, "model.safetensors"), metadata={"format": "pt"})
                    with self.assertRaisesRegex(RuntimeError, "conversion"):
                        self.load_checkpoint(case)
            gc.collect()

    def test_memory_estimate_and_scope_guards(self):
        from transformers.core_model_loading import WeightConverter, _IdentityOp
        from transformers.integrations.accelerate import compute_module_sizes

        with tempfile.TemporaryDirectory() as directory:
            model, _ = self.make_checkpoint(directory)
            quantizer = CompressedTensorsHfQuantizer(CompressedTensorsConfig(**nvfp4_config(), weight_only=True))
            name = "model.layers.0.self_attn.q_proj"
            weight = model.get_submodule(name).weight
            self.assertEqual(quantizer.param_element_size(model, f"{name}.weight", weight), 2)
            quantizer._process_model_before_weight_loading(model)
            sizes, _ = compute_module_sizes(model, quantizer, only_modules=False)
            self.assertEqual(sizes[f"{name}.weight"], weight.numel() // 2 + weight.numel() // 16 + 4)
            for converter in quantizer.get_weight_conversions():
                key = "other.layers.0.self_attn.q_proj.weight_packed"
                self.assertEqual(converter.rename_source_key(key), (key, None))
            with self.assertRaisesRegex(ValueError, "model-level WeightConverters"):
                quantizer.update_weight_conversions([WeightConverter("weight", "weight", [_IdentityOp()])], model)
            with self.assertRaisesRegex(ValueError, "bfloat16"):
                quantizer.update_dtype(torch.float32)
            self.assertFalse(quantizer.is_trainable or quantizer.is_qat_trainable or quantizer.is_serializable())
