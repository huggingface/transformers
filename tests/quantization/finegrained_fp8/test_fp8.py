# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import copy
import gc
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from parameterized import parameterized

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config, OPTForCausalLM
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.testing_utils import (
    backend_empty_cache,
    get_device_properties,
    require_accelerate,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


@contextmanager
def _patch_no_accelerator():
    with ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.is_available", return_value=False))
        if hasattr(torch, "xpu"):
            stack.enter_context(patch("torch.xpu.is_available", return_value=False))
            stack.enter_context(
                patch("transformers.quantizers.quantizer_finegrained_fp8.is_torch_xpu_available", return_value=False)
            )
        yield


@require_torch_accelerator
class FineGrainedFP8ConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Simple test that checks if one uses a config and converts it to a dict, the dict is the same as the config object
        """
        quantization_config = FineGrainedFP8Config()
        config_to_dict = quantization_config.to_dict()

        for key in config_to_dict:
            self.assertEqual(getattr(quantization_config, key), config_to_dict[key])

    def test_from_dict(self):
        """
        Simple test that checks if one uses a dict and converts it to a config object, the config object is the same as the dict
        """
        dict = {"modules_to_not_convert": ["lm_head.weight"], "quant_method": "fp8"}
        quantization_config = FineGrainedFP8Config.from_dict(dict)

        self.assertEqual(dict["modules_to_not_convert"], quantization_config.modules_to_not_convert)
        self.assertEqual(dict["quant_method"], quantization_config.quant_method)


def _quantize_static_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.clamp(tensor.float().abs().max() / 448.0, min=1e-6).reshape(1).to(torch.float32)
    quantized = torch.clamp(tensor.float() / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scale


def _dequantize_static_fp8(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tensor.float() * scale.float()


class _ReferenceInputQuantKernel:
    def channel_scale_quantize_fp8_static_bf16(self, input, channel_scale, scale):
        scaled = input.float() * channel_scale.float().reshape(1, -1)
        return torch.clamp(scaled / scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)


class _ReferenceMLPKernel:
    def fp8_swiglu_mlp_bf16(
        self,
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
    ):
        x = _dequantize_static_fp8(input, input_scale)
        gate_up = F.linear(x, _dequantize_static_fp8(gate_up_weight, gate_up_weight_scale))
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden_fp8 = torch.clamp(hidden / hidden_scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)
        out = F.linear(
            _dequantize_static_fp8(hidden_fp8, hidden_scale), _dequantize_static_fp8(down_weight, down_weight_scale)
        )
        return out.to(torch.bfloat16)

    def fp8_geglu_mlp_bf16(
        self,
        input,
        gate_up_weight,
        down_weight,
        input_scale,
        gate_up_weight_scale,
        hidden_scale,
        down_weight_scale,
    ):
        x = _dequantize_static_fp8(input, input_scale)
        gate_up = F.linear(x, _dequantize_static_fp8(gate_up_weight, gate_up_weight_scale))
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.gelu(gate, approximate="tanh") * up
        hidden_fp8 = torch.clamp(hidden / hidden_scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)
        out = F.linear(
            _dequantize_static_fp8(hidden_fp8, hidden_scale), _dequantize_static_fp8(down_weight, down_weight_scale)
        )
        return out.to(torch.bfloat16)

    def fp8_gelu_mlp_bf16(
        self,
        input,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        input_scale,
        up_weight_scale,
        hidden_scale,
        down_weight_scale,
    ):
        x = _dequantize_static_fp8(input, input_scale)
        up = F.linear(x, _dequantize_static_fp8(up_weight, up_weight_scale), up_bias.float())
        hidden = F.gelu(up, approximate="tanh")
        hidden_fp8 = torch.clamp(hidden / hidden_scale.float(), -448.0, 448.0).to(torch.float8_e4m3fn)
        out = F.linear(
            _dequantize_static_fp8(hidden_fp8, hidden_scale),
            _dequantize_static_fp8(down_weight, down_weight_scale),
            down_bias.float(),
        )
        return out.to(torch.bfloat16)


def _static_fp8_linear(in_features: int, out_features: int, *, has_bias: bool = False) -> nn.Module:
    from transformers.integrations.finegrained_fp8 import FP8Linear

    linear = FP8Linear(
        in_features,
        out_features,
        block_size=None,
        activation_scheme="static",
        has_bias=has_bias,
    )
    weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.1
    weight_fp8, weight_scale = _quantize_static_fp8(weight)
    linear.weight = nn.Parameter(weight_fp8, requires_grad=False)
    linear.weight_scale_inv = nn.Parameter(weight_scale, requires_grad=False)
    linear.activation_scale = nn.Parameter(torch.tensor([0.05], dtype=torch.float32), requires_grad=False)
    if has_bias:
        linear.bias = nn.Parameter(torch.randn(out_features, dtype=torch.bfloat16) * 0.01, requires_grad=False)
    return linear


class _TinyStaticFP8GatedMLP(nn.Module):
    def __init__(self, *, activation="gelu_pytorch_tanh"):
        super().__init__()
        from transformers.activations import ACT2FN

        self.gate_proj = _static_fp8_linear(8, 16)
        self.up_proj = _static_fp8_linear(8, 16)
        self.down_proj = _static_fp8_linear(16, 8)
        self.up_proj.activation_scale = nn.Parameter(
            self.gate_proj.activation_scale.detach().clone(), requires_grad=False
        )
        self.up_proj.weight_scale_inv = nn.Parameter(
            self.gate_proj.weight_scale_inv.detach().clone(), requires_grad=False
        )
        self.act_fn = ACT2FN[activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _TinyStaticFP8DenseGELUMLP(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers.activations import ACT2FN

        self.fc1 = _static_fp8_linear(8, 16, has_bias=True)
        self.fc2 = _static_fp8_linear(16, 8, has_bias=True)
        self.activation_fn = ACT2FN["gelu_pytorch_tanh"]

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))


class _TinyStaticFP8Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gated = _TinyStaticFP8GatedMLP()
        self.dense = _TinyStaticFP8DenseGELUMLP()
        self.config = SimpleNamespace()

    def forward(self, x):
        return self.gated(x), self.dense(x)


class StaticQuantizedMLPFusionTest(unittest.TestCase):
    fusion_config = {
        "input_quant": {"repo_id": "reference/input-quant"},
        "gated_mlp": {"repo_id": "reference/gated"},
        "dense_gelu_mlp": {"repo_id": "reference/dense"},
    }

    def _kernel_loader(self, repo_id, version=None, revision=None, trust_remote_code=False):
        del version, revision, trust_remote_code
        if repo_id == "reference/input-quant":
            return _ReferenceInputQuantKernel()
        if repo_id in {"reference/gated", "reference/dense"}:
            return _ReferenceMLPKernel()
        raise AssertionError(f"Unexpected repo id {repo_id}")

    def test_fusion_config_parsing(self):
        from transformers.integrations.static_quantized_mlp import get_static_quantized_mlp_fusion_spec

        spec = get_static_quantized_mlp_fusion_spec(
            {
                "input_quant": {"repo_id": "reference/input-quant", "trust_remote_code": True},
                "gated_mlp": {"repo_id": "reference/gated", "version": 2},
            }
        )
        self.assertEqual(spec.input_quant.kernel.repo_id, "reference/input-quant")
        self.assertTrue(spec.input_quant.kernel.trust_remote_code)
        self.assertEqual(spec.gated_mlp.kernel.version, 2)
        self.assertIsNone(spec.dense_gelu_mlp)

    def test_fusion_config_rejects_unknown_pattern(self):
        from transformers.fusion_mapping import register_fusion_patches

        with self.assertRaisesRegex(ValueError, "Unknown static quantized MLP fusion option"):
            register_fusion_patches(object, SimpleNamespace(), {"static_quantized_mlp": {"unknown_mlp": True}})

    def test_static_quantized_mlp_requires_explicit_kernel_config(self):
        from transformers.fusion_mapping import register_fusion_patches

        with self.assertRaisesRegex(ValueError, "requires explicit Hub kernel configurations"):
            register_fusion_patches(object, SimpleNamespace(), {"static_quantized_mlp": True})

    def test_static_fp8_gated_and_dense_gelu_mlp_replacement(self):
        from transformers.integrations.static_quantized_mlp import (
            StaticFP8DenseGELUMLP,
            StaticFP8GatedMLP,
            replace_with_static_quantized_mlp,
        )

        torch.manual_seed(0)
        model = _TinyStaticFP8Model()
        reference = copy.deepcopy(model)
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

        with patch("transformers.integrations.static_quantized_mlp._load_hub_kernel", side_effect=self._kernel_loader):
            model = replace_with_static_quantized_mlp(model, self.fusion_config)
            reference = replace_with_static_quantized_mlp(reference, self.fusion_config)

        self.assertIsInstance(model.gated, StaticFP8GatedMLP)
        self.assertIsInstance(model.dense, StaticFP8DenseGELUMLP)
        torch.testing.assert_close(model.gated(x), reference.gated(x))
        torch.testing.assert_close(model.dense(x), reference.dense(x))

        state_dict = model.state_dict()
        self.assertIn("gated.gate_proj.weight", state_dict)
        self.assertIn("dense.fc1.weight", state_dict)
        self.assertNotIn("gated.gate_up_weight", state_dict)
        self.assertNotIn("dense.input_channel_scale", state_dict)

    def test_static_fp8_dense_gelu_unsupported_activation_falls_back(self):
        from transformers.activations import ACT2FN
        from transformers.integrations.static_quantized_mlp import (
            StaticFP8DenseGELUMLP,
            replace_with_static_quantized_mlp,
        )

        model = _TinyStaticFP8Model()
        model.dense.activation_fn = ACT2FN["relu"]

        with patch("transformers.integrations.static_quantized_mlp._load_hub_kernel", side_effect=self._kernel_loader):
            model = replace_with_static_quantized_mlp(
                model,
                {
                    "input_quant": self.fusion_config["input_quant"],
                    "dense_gelu_mlp": self.fusion_config["dense_gelu_mlp"],
                },
            )

        self.assertNotIsInstance(model.dense, StaticFP8DenseGELUMLP)

    def test_quantizer_after_load_applies_static_quantized_mlp_fusion(self):
        from transformers.integrations.static_quantized_mlp import StaticFP8DenseGELUMLP, StaticFP8GatedMLP

        model = _TinyStaticFP8Model()
        model.config.fusion_config = {"static_quantized_mlp": self.fusion_config}
        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(activation_scheme="static", weight_block_size=None))

        with patch("transformers.integrations.static_quantized_mlp._load_hub_kernel", side_effect=self._kernel_loader):
            model = quantizer._process_model_after_weight_loading(model)

        self.assertIsInstance(model.gated, StaticFP8GatedMLP)
        self.assertIsInstance(model.dense, StaticFP8DenseGELUMLP)


@slow
@require_accelerate
@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping FP8QuantizerTest because it is not supported on GPU with capability < 8.9",
)
class FP8QuantizerTest(unittest.TestCase):
    model_name = "meta-llama/Llama-3.2-1B"
    quantized_model_name = "hf-internal-testing/Llama-3.2-1B-Instruct-fp8"
    input_text = "Once upon a time"
    max_new_tokens = 10
    EXPECTED_OUTPUTS = {
        "Once upon a time, there was a little girl who loved to play",
        "Once upon a time, there was a man who was very rich.",
    }
    EXPECTED_DEQUANTIZED_OUTPUT = "Once upon a time, in a small village nestled in the rolling hills"
    device_map = torch_device
    offload_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": "cpu",
        "model.layers.8": "cpu",
        "model.layers.9": "cpu",
        "model.layers.10": "cpu",
        "model.layers.11": "cpu",
        "model.layers.12": "cpu",
        "model.layers.13": "cpu",
        "model.layers.14": "cpu",
        "model.layers.15": "cpu",
        "model.rotary_emb": "cpu",
        "model.norm": "cpu",
        "lm_head": 0,
    }

    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.quantization_config = FineGrainedFP8Config()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=cls.quantization_config
        )

    def setup(self):
        """
        Clear also on each setup (e.g. if a different model is used than the base cls one)
        """
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    @parameterized.expand(
        [
            "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",
            "hf-internal-testing/tiny-random-MixtralForCausalLM",
        ]
    )
    def test_moe_conversion_doesnt_raise(self, model_id):
        quantization_config = FineGrainedFP8Config(weight_block_size=(32, 32))
        AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    def test_quantized_model_conversion(self):
        """
        Simple test that checks if the quantized model has been converted properly
        """

        from transformers.integrations import FP8Linear, replace_with_fp8_linear

        model_id = "facebook/opt-350m"
        config = AutoConfig.from_pretrained(model_id, revision="cb32f77e905cccbca1d970436fb0f5e6b58ee3c5")
        quantization_config = FineGrainedFP8Config()

        with torch.device("meta"):
            model = OPTForCausalLM(config)

        nb_linears = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                nb_linears += 1
        model = replace_with_fp8_linear(model, quantization_config=quantization_config)
        nb_fp8_linear = 0
        for module in model.modules():
            if isinstance(module, FP8Linear):
                nb_fp8_linear += 1
        self.assertEqual(nb_linears, nb_fp8_linear)
        with torch.device("meta"):
            model = OPTForCausalLM(config)
        quantization_config = FineGrainedFP8Config()
        model = replace_with_fp8_linear(model, modules_to_not_convert=["fc1"], quantization_config=quantization_config)
        nb_fp8_linear = 0
        for module in model.modules():
            if isinstance(module, FP8Linear):
                nb_fp8_linear += 1
        self.assertEqual(nb_linears - 24, nb_fp8_linear)

    def test_quantizer_validation_no_accelerator(self):
        """Test quantizer validation when CUDA/XPU is not available"""
        with _patch_no_accelerator():
            config = FineGrainedFP8Config()
            quantizer = FineGrainedFP8HfQuantizer(config)
            quantizer.pre_quantized = False

            with self.assertRaises(RuntimeError):
                quantizer.validate_environment()

    def test_dequantization_no_accelerator(self):
        """Test dequantization when CUDA/XPU is not available"""
        with _patch_no_accelerator():
            config = FineGrainedFP8Config()
            quantizer = FineGrainedFP8HfQuantizer(config)
            quantizer.pre_quantized = True
            quantizer.validate_environment()
            self.assertTrue(quantizer.quantization_config.dequantize)

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertIn(output_tokens, self.EXPECTED_OUTPUTS)

    def test_dequantized_model(self):
        """
        Simple test that checks if the dequantized model is working properly
        """
        quantization_config = FineGrainedFP8Config(dequantize=True)
        dequantized_model = AutoModelForCausalLM.from_pretrained(
            self.quantized_model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
        del dequantized_model

    def test_dequantize_when_no_accelerator(self):
        """
        Simple test that checks if the dequantized model is working properly when no accelerator is available
        """
        with _patch_no_accelerator():
            dequantized_model = AutoModelForCausalLM.from_pretrained(self.quantized_model_name, device_map="cpu")
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to("cpu")
            output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
            del dequantized_model

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_weight_and_weight_scale_inv(self):
        """
        Simple test that checks if the weight and weight_scale_inv are working properly
        """
        weight = self.quantized_model.model.layers[0].self_attn.q_proj.weight
        weight_scale_inv = self.quantized_model.model.layers[0].self_attn.q_proj.weight_scale_inv
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(weight_scale_inv.dtype, torch.float32)
        self.assertEqual(weight.shape, (weight_scale_inv.shape[0] * 128, weight_scale_inv.shape[1] * 128))

    def test_block_size(self):
        """
        Simple test that checks if the block size is working properly
        """
        self.assertEqual(self.quantized_model.config.quantization_config.weight_block_size, (128, 128))
        quantization_config = FineGrainedFP8Config(weight_block_size=(32, 32))
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        self.assertEqual(quantized_model.config.quantization_config.weight_block_size, (32, 32))

    @require_torch_multi_accelerator
    def test_quantized_model_multi_accelerators(self):
        """
        Simple test that checks if the quantized model is working properly with multiple accelerators
        set CUDA_VISIBLE_DEVICES=0,1 if you have more than 2 GPUs; or set ZE_AFFINITY_MASK=0,1 if you
        have more than 2 XPUs.
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        quantization_config = FineGrainedFP8Config()
        # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
        # this will lead to put the entire model to device 0.
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory={0: "1GB", 1: "10GB"},
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    @require_torch_multi_accelerator
    def test_save_pretrained_multi_accelerators(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
            # this will lead to put the entire model to device 0.
            model = AutoModelForCausalLM.from_pretrained(
                tmpdirname, device_map="auto", max_memory={0: "1GB", 1: "10GB"}
            )
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_quantized_model_offload(self):
        """
        Simple test that checks if the quantized model returns an error when loading with cpu/disk offloaded
        """
        with self.assertRaisesRegex(
            ValueError, "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."
        ):
            AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.offload_device_map, quantization_config=self.quantization_config
            )

    def test_save_pretrained_offload(self):
        """
        Simple test that checks if the saved quantized model is working properly cpu/disk offload
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            quantized_model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.offload_device_map)
            output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_compute_module_sizes(self):
        r"""
        Test if we compute the right module sizes needed to generate the device map.
        Also test if we get the right values for `total_byte_count` in `caching_allocator_warmup`.
        """
        from transformers.integrations import FP8Linear
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.modeling_utils import expand_device_map, get_total_byte_count
        from transformers.quantizers import AutoHfQuantizer

        # we need to preprocess the model like that because device_map calculation happens before we load the weights inside the model.
        # For normal wieghts, it's fine but for quantized weights, the tensors dtype might change during loading.
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
            model_size, _ = compute_module_sizes(model, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            total_byte_count = list(get_total_byte_count(model, expanded_device_map).values())[0]

            # testing prequantized = False should be enough, the shape should be the same whether it is pre-quantized or not
            hf_quantizer = AutoHfQuantizer.from_config(FineGrainedFP8Config(), pre_quantized=False)
            hf_quantizer.preprocess_model(model=model, config=model.config)
            quantized_model_size, _ = compute_module_sizes(model, hf_quantizer, only_modules=False)

            expected_keys = [name for name, _ in model.named_parameters()] + [
                name for name, _ in model.named_buffers()
            ]
            expanded_device_map = expand_device_map({"": torch_device}, expected_keys)
            quantized_total_byte_count = list(get_total_byte_count(model, expanded_device_map, hf_quantizer).values())[
                0
            ]

        for name, module in model.named_modules():
            if isinstance(module, FP8Linear):
                # from 16 bits to 8 bits
                assert int(model_size[f"{name}.weight"] // 2) == int(quantized_model_size[f"{name}.weight"])

        # check that we get the same value, as we use `compute_module_sizes` in `get_total_byte_count`
        assert total_byte_count == model_size[""]
        assert quantized_total_byte_count == quantized_model_size[""]

        # we should at least have 1.5 times memory reduction in total
        assert model_size[""] > quantized_model_size[""] * 1.5

    @parameterized.expand(["eager", "batched_mm", "grouped_mm", "deepgemm"])
    def test_quantized_moe_forward(self, experts_implementation):
        """
        Checks implicitly if the moe implementation is correct, i.e. it does not crash for cases
        where the indices go over `top_k` as shown within the Minimax M2 model
        """
        # deepgemm only has CUDA kernels, skip on other devices
        if experts_implementation == "deepgemm" and torch_device != "cuda":
            self.skipTest("deepgemm is only supported on CUDA")

        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/MiniMax-M2-Tiny-FP8",  # single layer version
            experts_implementation=experts_implementation,
            device_map=self.device_map,
        )
        assert model.config._experts_implementation == experts_implementation

        tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is your favourite condiment?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Do you have mayonnaise recipes?"}]},
        ]
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
            self.device_map
        )

        # Only caring about this not crashing
        _ = model.generate(**model_inputs, max_new_tokens=24)


@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping FP8LinearTest because it is not supported on GPU with capability < 8.9",
)
class FP8LinearTest(unittest.TestCase):
    device = torch_device

    def test_linear_preserves_shape(self):
        """
        Test that FP8Linear preserves shape when in_features == out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(256, 256, block_size=(128, 128)).to(self.device)
        x = torch.rand((1, 5, 256)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, x.shape)

    def test_linear_with_diff_feature_size_preserves_shape(self):
        """
        Test that FP8Linear generates the correct shape when in_features != out_features.
        """
        from transformers.integrations import FP8Linear

        linear = FP8Linear(128, 256, block_size=(128, 128)).to(self.device)
        x = torch.rand((1, 5, 128)).to(self.device)

        x_ = linear(x)
        self.assertEqual(x_.shape, (1, 5, 256))
