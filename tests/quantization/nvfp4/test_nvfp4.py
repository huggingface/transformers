# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers import MixtralConfig, MixtralForCausalLM, NVFP4Config
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.integrations.mxfp4 import FP4_VALUES
from transformers.integrations.nvfp4 import (
    NVFP4Dequantize,
    NVFP4Deserialize,
    NVFP4Experts,
    NVFP4ExpertsDeserialize,
    NVFP4Linear,
    NVFP4Quantize,
    swizzle_sf,
)
from transformers.quantizers.quantizer_nvfp4 import NVFP4HfQuantizer
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import QuantizationMethod


if is_torch_available():
    import torch
    from safetensors.torch import save_file
    from torch import nn


class _PackedWeight:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_kernel():
    def pack(weight, device):
        out_features, in_features = weight.shape
        return SimpleNamespace(
            qweight=torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            sf=torch.empty(out_features, in_features // 64, dtype=torch.int32, device=device),
            sf_rowmajor=torch.empty(out_features, in_features // 16, dtype=torch.uint8, device=device),
            global_scale=torch.ones(1, dtype=torch.float32, device=device),
        )

    def scaled_fp4_quant(input, global_scale, swizzled):
        return input.new_zeros(input.shape[0], input.shape[1] // 2, dtype=torch.uint8), input.new_zeros(
            input.shape[0], input.shape[1] // 64, dtype=torch.int32
        )

    def cutlass_scaled_fp4_mm(qinput, weight, input_sf, weight_sf, alpha):
        return qinput.new_zeros(qinput.shape[0], weight.shape[0], dtype=torch.bfloat16)

    return SimpleNamespace(
        PackedWeight=_PackedWeight,
        gemm=lambda packed_weight, input: input.new_zeros((*input.shape[:-1], packed_weight.n)),
        pack=pack,
        swizzled_sf_shape=lambda rows, columns: (rows, columns // 64),
        ops=SimpleNamespace(scaled_fp4_quant=scaled_fp4_quant, cutlass_scaled_fp4_mm=cutlass_scaled_fp4_mm),
    )


def _modelopt_quantize(weight):
    """ModelOpt's NVFP4 (`NVFP4QTensor.quantize`): `weight ~= fp4 * weight_scale * weight_scale_2`."""
    weight_scale_2 = weight.abs().amax() / (6 * 448)
    blocks = weight.view(weight.shape[0], -1, 16)
    weight_scale = (blocks.abs().amax(-1) / (6 * weight_scale_2)).to(torch.float8_e4m3fn)
    scaled = blocks / (weight_scale.float() * weight_scale_2).unsqueeze(-1)
    codes = (scaled.unsqueeze(-1) - torch.tensor(FP4_VALUES)).abs().argmin(-1).flatten(1)  # nearest e2m1
    return (codes[:, 0::2] | codes[:, 1::2] << 4).to(torch.uint8), weight_scale, weight_scale_2


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4ConfigTest(unittest.TestCase):
    def test_config_uses_quantization_method_enum(self):
        config = NVFP4Config(modules_to_not_convert=["lm_head"])

        self.assertEqual(config.quant_method, QuantizationMethod.NVFP4)
        self.assertEqual(config.to_dict()["modules_to_not_convert"], ["lm_head"])


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4IntegrationTest(unittest.TestCase):
    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_replaces_only_eligible_linears(self, _):
        model = nn.ModuleDict(
            {
                "convert": nn.Linear(64, 32, bias=False),
                "skip": nn.Linear(64, 32, bias=False),
                "biased": nn.Linear(64, 32, bias=True),
                "unaligned": nn.Linear(63, 32, bias=False),
            }
        )

        quantizer = NVFP4HfQuantizer(NVFP4Config(), pre_quantized=False)
        quantizer.modules_to_not_convert = ["skip"]
        quantizer.replace_modules(model)

        self.assertIsInstance(model["convert"], NVFP4Linear)
        self.assertIsInstance(model["skip"], nn.Linear)
        self.assertIsInstance(model["biased"], nn.Linear)
        self.assertIsInstance(model["unaligned"], nn.Linear)
        self.assertEqual(model["convert"].weight_sf.dtype, torch.int32)
        self.assertEqual(model["convert"].weight_sf_rowmajor.dtype, torch.uint8)
        self.assertEqual(model["convert"].weight_global_scale.dtype, torch.float32)

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_linear_forward_uses_normal_function_entry_point(self, _):
        layer = NVFP4Linear(64, 32)
        output = layer(torch.randn(2, 3, 64))

        self.assertEqual(output.shape, (2, 3, 32))

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_packed_tensors_are_registered_in_state_dict(self, _):
        layer = NVFP4Linear(64, 32)

        buffer_names = {"weight", "weight_sf", "weight_sf_rowmajor", "weight_global_scale", "input_scale"}
        for buffer_name in buffer_names:
            self.assertIsInstance(getattr(layer, buffer_name), nn.Buffer)
        self.assertEqual(set(layer.state_dict()), buffer_names)
        self.assertNotIn("nvfp4_kernel", layer.state_dict())

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_conversion_produces_all_registered_buffers(self, _):
        conversion = NVFP4Quantize(torch.device("cpu"))

        output = conversion.convert({"model.proj.weight": torch.randn(32, 64)})

        self.assertEqual(
            set(output),
            {
                "model.proj.weight",
                "model.proj.weight_sf",
                "model.proj.weight_sf_rowmajor",
                "model.proj.weight_global_scale",
                "model.proj.input_scale",
            },
        )
        self.assertEqual(output["model.proj.weight"].shape, (32, 32))

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_modelopt_checkpoint_round_trip(self, _):
        weight = torch.randn(128, 64)
        packed, weight_scale, weight_scale_2 = _modelopt_quantize(weight)
        checkpoint = {
            "weight": packed,
            "weight_scale": weight_scale,
            "weight_scale_2": weight_scale_2,
            "input_scale": torch.tensor(0.01),
        }

        buffers = NVFP4Deserialize(None).convert(dict(checkpoint))
        layer = NVFP4Linear(64, 128)
        self.assertEqual(set(buffers), set(layer.state_dict()))
        for name in ("weight_sf", "weight_sf_rowmajor", "weight_global_scale"):  # `weight` is a bf16 placeholder
            self.assertEqual(
                (name, buffers[name].dtype, buffers[name].shape),
                (name, getattr(layer, name).dtype, getattr(layer, name).shape),
            )
        torch.testing.assert_close(buffers["weight_sf_rowmajor"], weight_scale.view(torch.uint8))
        torch.testing.assert_close(
            buffers["weight_sf"].view(-1), swizzle_sf(weight_scale.view(torch.uint8)).view(torch.int32)
        )
        torch.testing.assert_close(buffers["weight_global_scale"], (1 / weight_scale_2).reshape(1))
        torch.testing.assert_close(buffers["input_scale"], torch.tensor([0.01]))

        model = SimpleNamespace(get_parameter_or_buffer=lambda name: torch.empty(0))
        dense = NVFP4Dequantize(None).convert(checkpoint, model=model, full_layer_name="proj.weight")["weight"][0]
        # Within half the widest e2m1 step (4 -> 6) of each block
        bound = (weight_scale.float() * weight_scale_2).unsqueeze(-1)
        self.assertTrue(((dense - weight).abs().view(128, -1, 16) <= bound + 1e-6).all())
        # The transformers layout of the same scales dequantizes to the same weight
        ours = {key: buffers[key] for key in ("weight", "weight_sf_rowmajor", "weight_global_scale")}
        torch.testing.assert_close(
            NVFP4Dequantize(None).convert(ours, model=model, full_layer_name="proj.weight")["weight"][0], dense
        )

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel")
    def test_static_input_scale_replaces_the_dynamic_one(self, load_kernel):
        kernel = _fake_kernel()
        kernel.ops = SimpleNamespace(
            scaled_fp4_quant=Mock(wraps=kernel.ops.scaled_fp4_quant),
            cutlass_scaled_fp4_mm=Mock(wraps=kernel.ops.cutlass_scaled_fp4_mm),
        )
        kernel.gemm = Mock(wraps=kernel.gemm)
        load_kernel.return_value = kernel
        layer = NVFP4Linear(64, 32)
        layer.weight_global_scale = torch.tensor([2.0])

        layer(torch.randn(3, 64))  # no static scale: the kernel's dynamic GEMM
        self.assertEqual((kernel.gemm.call_count, kernel.ops.scaled_fp4_quant.call_count), (1, 0))

        layer.input_scale = torch.tensor([0.5])
        self.assertEqual(layer(torch.randn(2, 3, 64)).shape, (2, 3, 32))
        torch.testing.assert_close(kernel.ops.scaled_fp4_quant.call_args.args[1], torch.tensor([2.0]))
        torch.testing.assert_close(kernel.ops.cutlass_scaled_fp4_mm.call_args.args[4], torch.tensor([0.25]))
        layer(torch.randn(1, 64))  # decode keeps the W4A16 GEMV
        self.assertEqual((kernel.gemm.call_count, kernel.ops.scaled_fp4_quant.call_count), (2, 1))


def _tiny_mixtral_config():
    # 128-row, 64-column aligned projections: the fake kernel's `swizzled_sf_shape` is unpadded
    return MixtralConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        # Only the experts are quantized in these tests
        quantization_config=NVFP4Config(modules_to_not_convert=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]),
    )


def _modelopt_experts_checkpoint(model):
    """The state dict of `model` as a ModelOpt NVFP4 export: per-expert `w1`/`w2`/`w3` with their scales."""
    checkpoint = {}
    for key, value in model.state_dict().items():
        if ".experts." not in key:
            checkpoint[key.replace(".mlp.gate.", ".block_sparse_moe.gate.")] = value.contiguous()
            continue
        prefix = key.rsplit(".mlp.", 1)[0] + ".block_sparse_moe.experts"
        for expert, weight in enumerate(value):
            gate, up = weight.chunk(2)
            projections = {"w1": gate, "w3": up} if key.endswith("gate_up_proj") else {"w2": weight}
            for name, projection in projections.items():
                packed, weight_scale, weight_scale_2 = _modelopt_quantize(projection.float())
                checkpoint[f"{prefix}.{expert}.{name}.weight"] = packed
                checkpoint[f"{prefix}.{expert}.{name}.weight_scale"] = weight_scale
                checkpoint[f"{prefix}.{expert}.{name}.weight_scale_2"] = weight_scale_2
                checkpoint[f"{prefix}.{expert}.{name}.input_scale"] = torch.tensor(0.01 * (expert + 1))
    return checkpoint


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4ExpertsTest(unittest.TestCase):
    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_experts_are_swapped_and_their_converters_target_the_nvfp4_buffers(self, _):
        with torch.device("meta"):
            model = MixtralForCausalLM(_tiny_mixtral_config())
        quantizer = NVFP4HfQuantizer(model.config.quantization_config, pre_quantized=True)
        quantizer.modules_to_not_convert = model.config.quantization_config.modules_to_not_convert
        quantizer.replace_modules(model)

        experts = model.model.layers[0].mlp.experts
        self.assertIsInstance(experts, NVFP4Experts)
        self.assertEqual(experts.gate_up_proj.shape, (4, 128, 64))
        self.assertEqual(experts.down_proj_sf_rowmajor.shape, (4, 128, 4))

        converters = [
            converter
            for converter in get_model_conversion_mapping(model, hf_quantizer=quantizer)
            if any(isinstance(op, NVFP4ExpertsDeserialize) for op in getattr(converter, "operations", []))
        ]
        self.assertEqual(
            {converter._original_target_patterns[0] for converter in converters},
            {".experts.gate_up_proj", ".experts.down_proj"},
        )
        sources = {source for converter in converters for source in converter._original_source_patterns}
        for name in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            self.assertIn(f".experts.*.w1.{name}", sources)

    @patch.object(NVFP4HfQuantizer, "validate_environment")  # no CUDA GPU needed with the fake kernel
    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_load_modelopt_experts(self, _, __):
        config = _tiny_mixtral_config()
        model = MixtralForCausalLM(config)
        checkpoint = _modelopt_experts_checkpoint(model)
        prefix = "model.layers.0.block_sparse_moe.experts"
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            save_file(checkpoint, f"{tmp}/model.safetensors", metadata={"format": "pt"})
            quantized, info = MixtralForCausalLM.from_pretrained(tmp, output_loading_info=True)
            dense, dense_info = MixtralForCausalLM.from_pretrained(
                tmp, output_loading_info=True, quantization_config=NVFP4Config(dequantize=True)
            )

        for loading_info in (info, dense_info):
            self.assertFalse(any(loading_info.values()), loading_info)

        experts = quantized.model.layers[0].mlp.experts
        self.assertIsInstance(experts, NVFP4Experts)
        expert = 2
        w1, w3 = (checkpoint[f"{prefix}.{expert}.{name}.weight"] for name in ("w1", "w3"))
        torch.testing.assert_close(experts.gate_up_proj[expert], torch.cat([w1, w3]))
        weight_scale = checkpoint[f"{prefix}.{expert}.w2.weight_scale"].view(torch.uint8)
        torch.testing.assert_close(experts.down_proj_sf_rowmajor[expert], weight_scale)
        torch.testing.assert_close(experts.down_proj_sf[expert].view(-1), swizzle_sf(weight_scale).view(torch.int32))
        global_scale = 1 / checkpoint[f"{prefix}.{expert}.w2.weight_scale_2"]
        torch.testing.assert_close(experts.down_proj_global_scale[expert], global_scale)
        torch.testing.assert_close(experts.gate_up_proj_input_scale, torch.tensor([0.01, 0.02, 0.03, 0.04]))
        self.assertEqual(quantized(torch.tensor([[1, 2, 3]])).logits.shape, (1, 3, 64))

        # `dequantize=True`: dense experts, within half the widest e2m1 step of each block
        reference = model.model.layers[0].mlp.experts.down_proj[expert]
        weight_scale_2 = checkpoint[f"{prefix}.{expert}.w2.weight_scale_2"]
        bound = (checkpoint[f"{prefix}.{expert}.w2.weight_scale"].float() * weight_scale_2).unsqueeze(-1)
        error = (dense.model.layers[0].mlp.experts.down_proj[expert] - reference).abs().view(128, -1, 16)
        self.assertTrue((error <= bound + 1e-6).all())

    def test_concatenated_projections_share_the_largest_global_scale(self):
        model = SimpleNamespace(get_parameter_or_buffer=lambda name: torch.empty(0))
        tensors, references = {}, []
        for name, weight in (("w1", torch.randn(128, 64)), ("w3", 4 * torch.randn(128, 64))):
            packed, weight_scale, weight_scale_2 = _modelopt_quantize(weight)
            tensors |= {f"{name}.weight": [packed], f"{name}.weight_scale": [weight_scale]}
            tensors |= {f"{name}.weight_scale_2": [weight_scale_2]}
            modelopt = {"weight": packed, "weight_scale": weight_scale, "weight_scale_2": weight_scale_2}
            references.append(NVFP4Dequantize(None).convert(modelopt, model=model, full_layer_name="w")["weight"][0])
        with patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel):
            buffers = NVFP4ExpertsDeserialize(None).convert(tensors, ["w1.weight", "w3.weight"], ["gate_up_proj"])

        layer = {
            "weight": buffers["gate_up_proj"][0],
            "weight_sf_rowmajor": buffers["gate_up_proj_sf_rowmajor"][0],
            "weight_global_scale": buffers["gate_up_proj_global_scale"],
        }
        gate, up = NVFP4Dequantize(None).convert(layer, model=model, full_layer_name="w")["weight"][0].chunk(2)
        torch.testing.assert_close(up, references[1])  # `w3` has the largest global scale: unchanged
        # `w1`'s block scales are rescaled to it, within the fp8 e4m3 rounding (half of a 2**-3 mantissa step)
        self.assertTrue(((gate - references[0]).abs() <= references[0].abs() / 16 + 1e-6).all())
        self.assertEqual(buffers["gate_up_proj_input_scale"].numel(), 0)


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4QuantizerValidationTest(unittest.TestCase):
    def setUp(self):
        self.quantizer = NVFP4HfQuantizer(NVFP4Config(), pre_quantized=False)

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(10, 0))
    def test_accepts_one_blackwell_cuda_device(self, _, __, ___):
        self.quantizer.validate_environment(device_map={"": "cuda:1"})

        self.assertEqual(self.quantizer.quantization_device, torch.device("cuda:1"))

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_rejects_multi_device_map_until_tp_metadata_is_supported(self, _, __):
        with self.assertRaisesRegex(ValueError, "Tensor parallelism"):
            self.quantizer.validate_environment(device_map={"model.layers.0": 0, "model.layers.1": 1})

    def test_rejects_native_tp_until_scale_metadata_is_supported(self):
        model = SimpleNamespace(config=SimpleNamespace(distributed_config=SimpleNamespace(tp_size=2)))

        with self.assertRaisesRegex(ValueError, "scale metadata does not have a sharding plan"):
            self.quantizer.update_parallel_plan(model, {})

    def test_update_parallel_plan_keeps_non_tp_configs(self):
        configs = (
            SimpleNamespace(),
            SimpleNamespace(distributed_config=SimpleNamespace(tp_size=1)),
        )

        for config in configs:
            with self.subTest(config=config):
                self.quantizer.update_parallel_plan(SimpleNamespace(config=config), {})

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_rejects_cpu_offload(self, _, __):
        with self.assertRaisesRegex(ValueError, "CPU or disk offload"):
            self.quantizer.validate_environment(device_map={"model": "cuda", "lm_head": "cpu"})
